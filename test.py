import os
import datetime

import numpy as np
import scipy
import scipy.sparse
import torch
import json

from tqdm import tqdm

from krylov.cg import conjugate_gradient, preconditioned_conjugate_gradient
from krylov.gmres import gmres
from krylov.preconditioner import get_preconditioner

from Scalable_NeuralIF import NeuralIF, LearnedPreconditioner
from neuralif.utils import torch_sparse_to_scipy, time_function
from neuralif.logger import TestResults

from apps.data import matrix_to_graph_sparse, get_dataloader, graph_to_matrix

# In[] help function:
def load_checkpoint(model:NeuralIF, config, device):
    """
    :param model: model for loading
    :param config: dict of parameters (formerly args)
    :param device: "cuda" or "cpu"
    :return: loaded model
    """
    # 1. checkpoint latest -> find the newest path
    # 2. other str -> the path of weight file
    # 3. None -> not load any weight file
    checkpoint = config["checkpoint"]

    if checkpoint == "latest":
        # list all the directories in the results folder

        d = os.listdir("./results/")
        d.sort()

        loaded_cfg = None

        # find the latest checkpoint in the newest experiment
        for i in range(len(d)):
            if os.path.isdir("./results/" + d[-i - 1]):
                dir_contents = os.listdir("./results/" + d[-i - 1])

                # looking for a directory with both config and model weights
                if "config.json" in dir_contents and "final_model.pt" in dir_contents:
                    # load the config.json file
                    with open("./results/" + d[-i - 1] + "/config.json") as f:
                        loaded_cfg = json.load(f)

                        # check if the model in test is same to mode in training
                        # for example, train model is Neuralif, test model should also be neuralif
                        if loaded_cfg["model"] != config["model"]:
                            loaded_cfg = None
                            continue

                        # best model or last model
                        if "best_model.pt" in dir_contents:
                            checkpoint = "./results/" + d[-i - 1] + "/best_model.pt"
                            break
                        else:
                            checkpoint = "./results/" + d[-i - 1] + "/final_model.pt"
                            break
        if loaded_cfg is None:
            print("Checkpoint not found...")

        # neuralif has optional drop tolerance...
        # Inference stage, we can drop some small values
        # threshold defined in the config at test.py
        if config["model"] == "neuralif":
            loaded_cfg["drop_tol"] = config["drop_tol"]

        # intialize model and hyper-parameters in orginal cfg with minor revision
        model = model(**loaded_cfg)
        print(f"load checkpoint from : {checkpoint}")

        model.load_state_dict(
            torch.load(checkpoint, weights_only=False, map_location=torch.device(device))
        )

    elif checkpoint is not None:
        # in this case, checkpoint is a path of the experiment folder
        # we also need to use config["weights"] to find the model file path
        # config["weights"] = "best_model" or "final_model"
        with open(checkpoint + "/config.json") as f:
            loaded_cfg = json.load(f)

        if config["model"] == "neuralif":
            loaded_cfg["drop_tol"] = config["drop_tol"]

        model = model(**loaded_cfg)
        model.load_state_dict(
            torch.load(
                checkpoint + f"/{config['weights']}.pt",
                map_location=torch.device(device)
            )
        )
        print(f"load checkpoint: {checkpoint}" + f"/{config['weights']}.pt")
    else:
        # do not load any weight file
        model = model(**{
            "global_features": 0,
            "latent_size": 8,
            "augment_nodes": False,
            "message_passing_steps": 3,
            "skip_connections": True,
            "activation": "relu",
            "aggregate": None,
            "edge_features": 1,
            "decode_nodes": False,
        })

        print("No checkpoint provided, using random weights")

    return model

def warmup(model, device):
    """avoid overhead in the first forward pass"""
    # set testing parameters
    model.to(device)
    model.eval()

    # run model warmup
    test_size = 1_000
    matrix = scipy.sparse.coo_matrix((np.ones(test_size), (np.arange(test_size), np.arange(test_size))))
    data = matrix_to_graph_sparse(matrix, torch.ones(test_size)) # identity matrix
    data.to(device)
    _ = model(data)

    print("Model warmup done...")


# inference model
@torch.inference_mode()
def test(model, test_loader, device, folder, save_results=False, dataset="random", solver="cg"):
    """
    test function
    :param model: trained model
    :param test_loader: PyTorch DataLoader
    :param device: "cuda" or "cpu"
    :param folder: the path of folder to store testing results
    :param save_results: whether store results of every sample to disk
    :param dataset: the method for data generation, typicall "random"
    :param solver: "cg" or "gmres" or "direct
    :return:
    """
    if save_results:
        os.makedirs(folder, exist_ok=True)

    print(f"\nTest:\t{len(test_loader.dataset)} samples")
    print(f"Solver:\t{solver} solver\n")

    # Two modes: either test baselines or the learned preconditioner
    if model is None:
        methods = ["ilu"] #["baseline", "jacobi", "ilu"]
    else:
        assert solver in ["cg", "gmres"], "Data-driven method only works with CG or GMRES"
        methods = ["learned"]

    # using direct solver
    if solver == "direct":
        methods = ["direct"]

    for method in methods:
        print(f"Testing {method} preconditioner")

        test_results = TestResults(method, dataset, folder,
                                   model_name=f"\n{model.__class__.__name__}" if method == "learned" else "",
                                   target=1e-6,
                                   solver=solver)

        for sample, data in tqdm(enumerate(test_loader), desc="Testing", total = test_loader.__len__()):
            plot = save_results and sample == (len(test_loader.dataset) - 1)

            # Getting the preconditioners
            start = time_function()

            data = data.to(device)

            with torch.inference_mode():
                prec = LearnedPreconditioner(data, model)
            # prec = get_preconditioner(data, method, model=model)

            # Get properties...
            p_time = prec.time
            breakdown = prec.breakdown
            nnzL = prec.nnz

            stop = time_function()

            A, b = graph_to_matrix(data)
            A = A.to("cpu").to(torch.float64)
            b = b.to("cpu").to(torch.float64)

            # A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(),
            #                             dtype=torch.float64,
            #                             requires_grad=False).to("cpu").to_sparse_csr()
            # b = data.x[:, 0].squeeze().to("cpu").to(torch.float64)

            b_norm = torch.linalg.norm(b)

            # we assume that b is unit norm wlog
            b = b / b_norm
            solution = data.s.to("cpu").to(torch.float64).squeeze() / b_norm if hasattr(data, "s") else None

            overhead = (stop - start) - (p_time)

            # RUN CONJUGATE GRADIENT
            start_solver = time_function()

            solver_settings = {
                "max_iter": 10_000,
                "x0": None
            }

            if breakdown:
                res = []

            elif solver == "direct":

                # convert to sparse matrix (scipy)
                A_ = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(),
                                             dtype=torch.float64, requires_grad=False)

                # scipy sparse...
                A_s = torch_sparse_to_scipy(A_).tocsr()

                # override start time
                start_solver = time_function()

                dense = False

                if dense:
                    _ = scipy.linalg.solve(A_.to_dense().numpy(), b.numpy(), assume_a='pos')
                else:
                    _ = scipy.sparse.linalg.spsolve(A_s, b.numpy())

                # dummy values...
                res = [(torch.Tensor([0]), torch.Tensor([0]))] * 2

            elif solver == "cg" and method == "baseline":
                # no preconditioner required when using baseline method
                res, _ = conjugate_gradient(A, b, x_true=solution,
                                            rtol=test_results.target, **solver_settings)

            elif solver == "cg":
                res, _ = preconditioned_conjugate_gradient(A, b, M=prec, x_true=solution,
                                                           rtol=test_results.target, **solver_settings)

            elif solver == "gmres":

                res, _ = gmres(A, b, M=prec, x_true=solution,
                               **solver_settings, plot=plot,
                               atol=test_results.target,
                               left=False)

            stop_solver = time_function()
            solver_time = (stop_solver - start_solver)

            # LOGGING
            test_results.log_solve(A.shape[0], solver_time, len(res) - 1,
                                   np.array([r[0].item() for r in res]),
                                   np.array([r[1].item() for r in res]),
                                   p_time, overhead)

            # ANALYSIS of the preconditioner and its effects!
            nnzA = A._nnz()

            test_results.log(nnzA, nnzL, plot=plot)

            svd = False
            if svd:
                # compute largest and smallest singular value
                Pinv = prec.get_inverse()
                APinv = A.to_dense() @ Pinv

                # compute the singular values of the preconditioned matrix
                S = torch.linalg.svdvals(APinv)

                # print the smallest and largest singular value
                test_results.log_eigenval_dist(S, plot=plot)

                # compute the loss of the preconditioner
                p = prec.get_p_matrix()
                loss1 = torch.linalg.norm(p.to_dense() - A.to_dense(), ord="fro")

                a_inv = torch.linalg.inv(A.to_dense())
                loss2 = torch.linalg.norm(p.to_dense() @ a_inv - torch.eye(a_inv.shape[0]), ord="fro")

                test_results.log_loss(loss1, loss2, plot=False)

                print(
                    f"Smallest singular value: {S[-1]} | Largest singular value: {S[0]} | Condition number: {S[0] / S[-1]}")
                print(f"Loss Lmax: {loss1}\tLoss Lmin: {loss2}")
                print()

        if save_results:
            test_results.save_results()

        test_results.print_summary()

base_config_path = "results/benchmark/config.json"
with open(base_config_path) as f:
    base_config = json.load(f)
base_config = {**base_config}

test_loader = get_dataloader(base_config["dataset"], base_config["n"], 1, spd=True, mode="test")
model_name_list = [None, "benchmark", "saint_graph_bs500", "saint_graph_bs1000", "saint_graph_bs2000", "saint_graph_bs4000", "saint_graph_bs6000"]

def run_test(base_config, model_name):
    config = base_config.copy()
    config["name"] = model_name

    device = "cpu" #torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if config.get("name") is not None:
        folder = "results/" + config.get("name")
        model = "neuralif"
        print(f"Using device: {device}, Using model: {model}")
    else:
        folder = "results/IC"
        print(f"Using device: {device}, Using model: Incomplete Cholesky")

    config["drop_tol"] = 0
    config["checkpoint"] = folder
    config["weights"] = "final_model"
    if model is not None:
        model = load_checkpoint(NeuralIF, config, device)
        warmup(model, device)

    test(model, test_loader, device, folder,
         save_results=True, dataset="random", solver="cg")

for model_name in model_name_list:
    run_test(base_config, model_name)