import os
import datetime

import torch
import torch_geometric
import time

from tqdm import tqdm

from apps.data import get_dataloader, graph_to_matrix
from Scalable_NeuralIF import NeuralIF, LearnedPreconditioner

from neuralif.utils import count_parameters, save_dict_to_file, condition_number, eigenval_distribution, gershgorin_norm
from neuralif.logger import TrainResults, TestResults
from neuralif.loss import loss

from krylov.cg import preconditioned_conjugate_gradient
from krylov.gmres import gmres

# In[]: Validate Functiom
@torch.no_grad()
def validate(model, validation_loader, solve=False, solver="cg", **kwargs):
    r"""
    Evaluate the model on validation set.
    1. If solve = False, then compute the Frobenius norm of preconditioner and A.
    2. If solve = True, then compute the average iterations  of CG.
    :param model: The trained NeuralIF model
    :param validation_loader: The Dataloader
    :param solve: bool, True or False, whether compute F_norm or iterations
    :param solver: "cg" or "gmres"
    :return: average F_norm or iteration numbers
    """
    model.eval()
    acc_loss = 0.0
    num_loss = 0
    acc_solver_iters = 0.0

    for i, data in enumerate(validation_loader):
        data = data.to(device)
        # construct problem data
        A, b = graph_to_matrix(data)

        # run conjugate gradient method
        # this requires the learned preconditioner to be reasonably good!
        if solve:
            # run CG on CPU
            with torch.inference_mode():
                preconditioner = LearnedPreconditioner(data, model)

            A = A.to("cpu").to(torch.float64)
            b = b.to("cpu").to(torch.float64)
            x_init = None

            solver_start = time.time()

            if solver == "cg":
                l, x_hat = preconditioned_conjugate_gradient(A.to("cpu"), b.to("cpu"), M=preconditioner,
                                                             x0=x_init, rtol=1e-6, max_iter=1_000)
            elif solver == "gmres":
                l, x_hat = gmres(A, b, M=preconditioner, x0=x_init, atol=1e-6, max_iter=1_000, left=False)
            else:
                raise NotImplementedError("Solver not implemented choose between CG and GMRES!")

            solver_stop = time.time()

            # Measure preconditioning performance
            solver_time = (solver_stop - solver_start)
            acc_solver_iters += len(l) - 1

        else:
            r"""
            Here model is NeuralIF, the outputs are the matrix L, we compute the loss
                \| LL^T - A \|^2_F
            """
            output, _, _ = model(data)
            # Here, we compute the loss using the full forbenius norm (no estimator)
            # l = frobenius_loss(output, A)
            l = loss(data, output, config="frobenius")
            acc_loss += l.item()
            num_loss += 1

    if solve:
        # print(f"Smallest eigenvalue: {dist[0]}")
        print(f"Validation\t iterations:\t{acc_solver_iters / len(validation_loader):.2f}")
        return acc_solver_iters / len(validation_loader)

    else:
        print(f"Validation loss:\t{acc_loss / num_loss:.2f}")
        return acc_loss / len(validation_loader)

# In[]: Config Definition
config = {
    "name": "experiment_train",
    "save": True,
    "seed": 42,
    "n": 0,
    "batch_size": 4,
    "num_epochs": 100,
    "dataset": "random",
    "loss": None,
    "gradient_clipping": 1.0,
    "regularizer": 0.0,
    "scheduler": False,
    "model": "neuralif",
    "normalize": False,
    "latent_size": 8,
    "message_passing_steps": 3,
    "decode_nodes": False,
    "normalize_diag": False,
    "aggregate": ["mean", "sum"],
    "activation": "relu",
    "skip_connections": True,
    "augment_nodes": False,
    "global_features": 0,
    "edge_features": 1,
    "graph_norm": False,
    "two_hop": False,
    "num_neighbors": [15, 10],  # number of neighbours to sample in each hop (GraphSAGE sampling)
    "device": "cuda"
}

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare output folder
if config["name"]:
    folder = f"results/{config['name']}"
else:
    folder = f"results/" + datetime.datetime.now().strftime("results/%Y-%m-%d_%H-%M-%S")
if config["save"]:
    os.makedirs(folder, exist_ok=True)
    save_dict_to_file(config, os.path.join(folder, "config.json"))


# global seed-ish
torch_geometric.seed_everything(config["seed"])

# args for the model
model_args = {k: config[k] for k in ["latent_size", "message_passing_steps", "skip_connections",
                                     "augment_nodes", "global_features", "decode_nodes",
                                     "normalize_diag", "activation", "aggregate", "graph_norm",
                                     "two_hop", "edge_features", "normalize"]
              if k in config}

# run the GMRES algorithm instead of CG (?)
gmres = False

model = NeuralIF(**model_args)
model.to(device)

print(f"Number params in model: {count_parameters(model)}/n")

optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

# Setup datasets
train_loader = get_dataloader(config["dataset"], config["n"], config["batch_size"],
                              spd=not gmres, mode="train")

validation_loader = get_dataloader(config["dataset"], config["n"], 1, spd=(not gmres), mode="val")

best_val = float("inf")
logger = TrainResults(folder)

# todo: compile the model
# compiled_model = torch.compile(model, mode="reduce-overhead")
# model = torch_geometric.compile(model, mode="reduce-overhead")

total_it = 0

# Train loop
for epoch in range(config["num_epochs"]):
    running_loss = 0.0
    grad_norm = 0.0

    start_epoch = time.perf_counter()

    print(f"Epoch: {epoch}")

    for it, data in  enumerate(tqdm(train_loader, desc="Training", total=len(train_loader))):
        # increase iteration count
        total_it += 1

        # enable training mode
        model.train()

        start = time.perf_counter()
        data = data.to(device)

        output, reg, _ = model(data)
        l = loss(output, data, c=reg, config=config["loss"])

        #  if reg:
        #    l = l + config["regularizer"] * reg

        l.backward()
        running_loss += l.item()

        # track the gradient norm
        if "gradient_clipping" in config and config["gradient_clipping"]:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])

        else:
            total_norm = 0.0

            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2

            grad_norm = total_norm ** 0.5 / config["batch_size"]

        # update network parameters
        optimizer.step()
        optimizer.zero_grad()

        logger.log(l.item(), grad_norm, time.perf_counter() - start)

    # Do validation after 100 updates (to support big datasets)
    # convergence is expected to be pretty fast...
    if (epoch + 1) % 10 == 0:

        # start with cg-checks after 5 iterations
        val_its = validate(model, validation_loader, solve=True,
                           solver="gmres" if gmres else "cg")

        # use scheduler
        # if config["scheduler"]:
        #    scheduler.step(val_loss)

        logger.log_val(None, val_its)

        # val_perf = val_cgits if val_cgits > 0 else val_loss
        val_perf = val_its

        if val_perf < best_val:
            if config["save"]:
                torch.save(model.state_dict(), f"{folder}/best_model.pt")
            best_val = val_perf

    epoch_time = time.perf_counter() - start_epoch

    # save model every epoch for analysis...
    if config["save"]:
        torch.save(model.state_dict(), f"{folder}/model_epoch{epoch + 1}.pt")

    print(f"Epoch {epoch + 1} \t loss: {1 / len(train_loader) * running_loss} \t time: {epoch_time}")

# save fully trained model
if config["save"]:
    logger.save_results()
    torch.save(model.to(torch.float).state_dict(), f"{folder}/final_model.pt")

# Test the model
# wandb.run.summary["validation_chol"] = best_val
print()
print("Best validation loss:", best_val)


