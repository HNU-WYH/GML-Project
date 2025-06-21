import os
import datetime

import torch
import torch_geometric
import time

from torch_geometric.data import Batch
from tqdm import tqdm

from apps.data import get_dataloader, graph_to_matrix, FolderDataset
from Scalable_NeuralIF import NeuralIF, LearnedPreconditioner, ToLowerTriangular

from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.loader import GraphSAINTNodeSampler
from neuralif.utils import count_parameters, save_dict_to_file, condition_number, eigenval_distribution, gershgorin_norm
from neuralif.logger import TrainResults, TestResults
from neuralif.loss import loss

from krylov.cg import preconditioned_conjugate_gradient
from krylov.gmres import gmres

# # In[]: Print the average degree in val data
# mode = "train"
# file_list = os.listdir(f"dataset/" + mode)
#
# deg_sum = 0
# for i, file_path in tqdm(enumerate(file_list), desc="File Index", total=len(file_list)):
#     # only the out-degree is computed here
#     data  = torch.load(os.path.join(f"dataset/" + mode, file_path), weights_only=False)
#     deg = torch_geometric.utils.degree(data.edge_index[0], num_nodes=data.x.size(0))
#     avg_deg = deg.mean().item()
#     deg_sum += avg_deg
# avg_deg = deg_sum / len(file_list)
#
# print(f"The average degree in {mode} dataset is: {avg_deg}")

# In[]:
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

    for i, data in enumerate(tqdm(validation_loader, desc="Validation",  total = len(validation_loader))):
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

# In[]:
# Configuration
config = {
    "name": "saint_graph",
    "sample_size": 4000,
    "save": True,
    "seed": 42,
    "n": 0,
    "num_epochs": 81,
    "sample_coverage": 2,
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
    "device": "cuda"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed
torch_geometric.seed_everything(config["seed"])

# In[] Self-define sage sampler
class SubgraphSampler(IterableDataset):
    def __init__(self, graph_paths, sample_size, sample_coverage = 2, save_dir = r".\results\saint"):
        super().__init__()
        self.graph_paths   = graph_paths
        self.sample_size   = sample_size
        self.sample_coverage = sample_coverage
        self.cache_dir = save_dir

    def __iter__(self):
        for idx, path in enumerate(self.graph_paths):
            full_graph = torch.load(path, weights_only=False)

            num_steps = full_graph.num_nodes // self.sample_size + 1
            cache_dir = os.path.join(self.cache_dir, f'graph_{idx}')
            os.makedirs(cache_dir, exist_ok=True)

            loader = GraphSAINTNodeSampler(
                data=full_graph,
                batch_size=self.sample_size,    # 每次采 sample_size 个 seed
                shuffle=True,
                sample_coverage=self.sample_coverage,
                num_steps=num_steps,
                save_dir=cache_dir,
            )
            for sub in loader:
                yield sub

    def __len__(self):
        return sum(
            (torch.load(p, weights_only=False).num_nodes // self.sample_size + 1)
            for p in self.graph_paths
        )

class ToSymmetric(torch_geometric.transforms.BaseTransform):
    """
    把 data.edge_index / edge_attr 复制到上三角，使子图对称。
    """
    def __call__(self, lower):
        ei, ea = torch_geometric.utils.to_undirected(
            lower.edge_index,
            lower.edge_attr,
            reduce='mean'          # 重复边取平均；如需保留同值可用 'max'
        )
        lower.edge_index, lower.edge_attr = ei, ea
        return lower

# In[]: local to global matrix

def local_to_global_sparse(local_L: torch.Tensor, batch_subgraph) -> torch.Tensor:
    """Convert a sparse matrix whose indices are *local* (0..tilde_N-1)
    to global indices (0..N-1).

    Args:
        local_L (torch.sparse_coo_tensor): output of model on the subgraph.
        batch_subgraph (torch_geometric.data.Data): the subgraph that contains
            attributes:
              - global_id : LongTensor[tilde_N] mapping local → global id
              - full_n    : int, total #nodes in original graph.
    Returns:
        torch.sparse_coo_tensor of shape (full_n, full_n) with indices mapped
        back to the original graph coordinate system.
    """
    # local row/col indices
    local_L = local_L.coalesce()
    row_loc, col_loc = local_L.indices()
    # map to global
    glob = batch_subgraph.global_id
    row_glob = glob[row_loc]
    col_glob = glob[col_loc]
    edge_idx_glob = torch.stack([row_glob, col_glob], dim=0)
    # rebuild sparse tensor in global space (values shared)
    full_n = int(batch_subgraph.full_n)
    return torch.sparse_coo_tensor(edge_idx_glob, local_L.values(), (full_n, full_n)).coalesce()

# In[]: Validation Data
train_folder = "./dataset/train/"
val_folder   = "./dataset/val/"
train_paths  = FolderDataset(train_folder, n=config["n"], graph=True).files
val_graphs   = FolderDataset(val_folder,   n=config["n"], graph=True)
val_data     = Batch.from_data_list([torch.load(p, weights_only=False) for p in val_graphs.files])
validation_loader = get_dataloader(config["dataset"], config["n"], 1, spd=(not gmres), mode="val")

# In[]: Parameter Analysis
def run_experiment(batch_size):
    tag = f'bs{batch_size}'
    cfg = config.copy()
    cfg['sample_size'] = batch_size
    cfg['name'] = f"{config['name']}_{tag}"

    folder = f"results/{cfg['name']}"
    os.makedirs(folder, exist_ok=True)

    logger = TrainResults(folder)
    save_dict_to_file(cfg, os.path.join(folder, "config.json"))

    # ==== Dataset & loader ====
    subgraph_dataset = SubgraphSampler(
        graph_paths=train_paths,
        sample_size=cfg["sample_size"],
        sample_coverage=cfg["sample_coverage"],
        save_dir=folder + '/saint_cache',
    )
    train_loader = DataLoader(subgraph_dataset, batch_size=None)

    model = NeuralIF(
        latent_size=cfg["latent_size"],
        message_passing_steps=cfg["message_passing_steps"],
        skip_connections=cfg["skip_connections"],
        augment_nodes=cfg["augment_nodes"],
        global_features=cfg["global_features"],
        decode_nodes=cfg["decode_nodes"],
        normalize_diag=cfg["normalize_diag"],
        activation=cfg["activation"],
        aggregate=cfg["aggregate"],
        two_hop=cfg["two_hop"],
        edge_features=cfg["edge_features"],
        graph_norm=cfg["graph_norm"],
    ).to(device)
    print("#Params:", count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    best_val = float("inf")
    for epoch in range(cfg["num_epochs"]):
        running_loss = 0.0
        grad_norm = 0.0

        start_epoch = time.perf_counter()

        model.train()

        for sub in tqdm(train_loader, desc=f"Epoch {epoch+1}", total=len(train_loader)):
            sub = sub.to(device)
            out, reg, _ = model(sub)

            l = loss(out, sub, c=reg, config=cfg["loss"], node_norm = sub.node_norm)
            l.backward()
            running_loss += l.item()

            # track the gradient norm
            if "gradient_clipping" in cfg and cfg["gradient_clipping"]:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clipping"])

            else:
                total_norm = 0.0

                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2

                grad_norm = total_norm ** 0.5 / cfg["sample_size"]

            # update network parameters
            optimizer.step()
            optimizer.zero_grad()

            logger.log(l.item(), grad_norm, time.perf_counter() - start_epoch)

        if (epoch+1) % 10 == 0:
            val_its = validate(model, validation_loader, solve=True,
                               solver="gmres" if gmres else "cg")
            logger.log_val(None, val_its)
            val_perf = val_its
            print(f"The optimal required iterations in CG is {val_its}")

            if val_perf < best_val:
                if cfg["save"]:
                    os.makedirs(folder, exist_ok=True)
                    torch.save(model.state_dict(), f"{folder}/best_model.pt")
                best_val = val_perf

        epoch_time = time.perf_counter() - start_epoch

        # save model every epoch for analysis...
        if cfg["save"]:
            torch.save(model.state_dict(), f"{folder}/model_epoch{epoch + 1}.pt")

        print(f"Epoch {epoch + 1} \t loss: {1 / len(train_loader) * running_loss} \t time: {epoch_time}")

    # save fully trained model
    if cfg["save"]:
        logger.save_results()
        torch.save(model.to(torch.float).state_dict(), f"{folder}/final_model.pt")

    print("Best validation loss:", best_val)
    return best_val

# In[] Main:
BATCH_LIST = [256, 512, 1024, 2048, 4096]

results = {}
for bs in BATCH_LIST:
    results[bs] = run_experiment(bs)

print("\n=== Summary ===")
for bs, iters in results.items():
    print(f"batch={bs:<5d}  best_iters={iters:.1f}")