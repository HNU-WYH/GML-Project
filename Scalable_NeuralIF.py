#!/usr/bin/env python
# coding: utf-8

# # Scalable GNN–Based Preconditioners for Conjugate Gradient Methods
# **Authors: Nicholas Tan Yun Yu, Low Jun Yu, Yuhan Wu**
# 
# This project was inspired by [NeuralIF](https://arxiv.org/abs/2305.16368).
# 
# **Summary**: The authors come up with a novel message-passing GNN block that is used by the network to predict efficient preconditioners to solve sparse linear systems. These preconditioners are tested using the preconditioned conjugate gradient (CG) method, which make the algorithm converge faster than other state-of-the-art preconditioners.
# 
# **Motivation**: Modern data-driven and physics-based applications frequently force us to deal with dense matrices. Therefore, we hope to show that the message-passing GNN block can learn effective preconditioners for these scaled up fields. An example of a machine learning problem that could benefit from this is Gaussian Processes, which makes use of a dense kernel function as such: (some image)
# 
# **The problem**: Scaling the problem to dense matrices is nontrivial. The Coates graph representation has 1 node per row/column, and one edge only for each nonzero entry in A. For a dense n*n matrix, that graph becomes complete – with n^2 edges – so both memory and compute blow up to O(n^2).
# 
# **Research direction**: Implement an edge-regression GNN that can work on dense matrices. We can achieve this using sampling techniques such as GraphSAGE and Cluster-GCN.

# # 1. Installation & Setup

# ## Load files from google drive

# In[1]:
import os
import datetime
import pprint
import time

import numpy as np
import scipy
import torch
import torch_geometric
import torch.nn as nn
import torch_geometric.nn as pyg
from torch_geometric.nn import aggr
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Batch
from scipy.sparse import tril, coo_matrix

from apps.synthetic import create_dataset
from apps.data import get_dataloader, FolderDataset
from apps.data import graph_to_matrix, matrix_to_graph

from neuralif.utils import (
    count_parameters, save_dict_to_file,
    condition_number, eigenval_distribution, gershgorin_norm,
    TwoHop
)
from neuralif.logger import TrainResults, TestResults
from neuralif.loss import loss

from krylov.cg import preconditioned_conjugate_gradient
from krylov.gmres import gmres

# import from self-curated numml file
# from numml import SparseCSRTensor
# ## Set GPU
# In[2]: Device Setting Up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# In[3]: Data Generation
n = 10_000
alpha = 10e-4

create_dataset(n, 1000, alpha=alpha, mode='train', rs=0, graph=True, solution=True)
create_dataset(n, 10, alpha=alpha, mode='val', rs=10000, graph=True, solution=True)
create_dataset(n, 100, alpha=alpha, mode='test', rs=103600, graph=True, solution=True)

# In[4]: Model
class GraphNet(nn.Module):
    # Follows roughly the outline of torch_geometric.nn.MessagePassing()
    # As shown in https://github.com/deepmind/graph_nets
    # Here is a helpful python implementation:
    # https://github.com/NVIDIA/GraphQSat/blob/main/gqsat/models.py
    # Also allows multirgaph GNN via edge_2_features
    def __init__(self, node_features, edge_features, global_features=0, hidden_size=0,
                 aggregate="mean", activation="relu", skip_connection=False, edge_features_out=None):

        super().__init__()

        # different aggregation functions
        if aggregate == "sum":
            self.aggregate = aggr.SumAggregation()
        elif aggregate == "mean":
            self.aggregate = aggr.MeanAggregation()
        elif aggregate == "max":
            self.aggregate = aggr.MaxAggregation()
        elif aggregate == "softmax":
            self.aggregate = aggr.SoftmaxAggregation(learn=True)
        else:
            raise NotImplementedError(f"Aggregation '{aggregate}' not implemented")

        self.global_aggregate = aggr.MeanAggregation()

        add_edge_fs = 1 if skip_connection else 0
        edge_features_out = edge_features if edge_features_out is None else edge_features_out

        # Graph Net Blocks (see https://arxiv.org/pdf/1806.01261.pdf)
        self.edge_block = MLP([global_features + (edge_features + add_edge_fs) + (2 * node_features),
                               hidden_size,
                               edge_features_out],
                              activation=activation)

        self.node_block = MLP([global_features + edge_features_out + node_features,
                               hidden_size,
                               node_features],
                              activation=activation)

        # optional set of blocks for global GNN
        self.global_block = None
        if global_features > 0:
            self.global_block = MLP([edge_features_out + node_features + global_features,
                                     hidden_size,
                                     global_features],
                                    activation=activation)

    def forward(self, x, edge_index, edge_attr, g=None):
        row, col = edge_index

        if self.global_block is not None:
            assert g is not None, "Need global features for global block"

            # run the edge update and aggregate features
            edge_embedding = self.edge_block(torch.cat([torch.ones(x[row].shape[0], 1, device=x.device) * g,
                                                        x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)


            agg_features = torch.cat([torch.ones(x.shape[0], 1, device=x.device) * g, x, aggregation], dim=1)
            node_embeddings = self.node_block(agg_features)

            # aggregate over all edges and nodes (always mean)
            mp_global_aggr = g
            edge_aggregation_global = self.global_aggregate(edge_embedding)
            node_aggregation_global = self.global_aggregate(node_embeddings)

            # compute the new global embedding
            # the old global feature is part of mp_global_aggr
            global_embeddings = self.global_block(torch.cat([node_aggregation_global,
                                                             edge_aggregation_global,
                                                             mp_global_aggr], dim=1))

            return edge_embedding, node_embeddings, global_embeddings

        else:
            # update edge features and aggregate
            edge_embedding = self.edge_block(torch.cat([x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)
            agg_features = torch.cat([x, aggregation], dim=1)
            # update node features
            node_embeddings = self.node_block(agg_features)
            return edge_embedding, node_embeddings, None


class MLP(nn.Module):
    def __init__(self, width, layer_norm=False, activation="relu", activate_final=False):
        super().__init__()
        width = list(filter(lambda x: x > 0, width))
        assert len(width) >= 2, "Need at least one layer in the network!"

        lls = nn.ModuleList()
        for k in range(len(width)-1):
            lls.append(nn.Linear(width[k], width[k+1], bias=True))
            if k != (len(width)-2) or activate_final:
                if activation == "relu":
                    lls.append(nn.ReLU())
                elif activation == "tanh":
                    lls.append(nn.Tanh())
                elif activation == "leakyrelu":
                    lls.append(nn.LeakyReLU())
                elif activation == "sigmoid":
                    lls.append(nn.Sigmoid())
                else:
                    raise NotImplementedError(f"Activation '{activation}' not implemented")

        if layer_norm:
            lls.append(nn.LayerNorm(width[-1]))

        self.m = nn.Sequential(*lls)

    def forward(self, x):
        return self.m(x)


class MP_Block(nn.Module):
    # L@L.T matrix multiplication graph layer
    # Aligns the computation of L@L.T - A with the learned updates
    def __init__(self, skip_connections, first, last, edge_features, node_features, global_features, hidden_size, **kwargs) -> None:
        super().__init__()

        # first and second aggregation
        if "aggregate" in kwargs and kwargs["aggregate"] is not None:
            aggr = kwargs["aggregate"] if len(kwargs["aggregate"]) == 2 else kwargs["aggregate"] * 2
        else:
            aggr = ["mean", "sum"]

        act = kwargs["activation"] if "activation" in kwargs else "relu"

        edge_features_in = 1 if first else edge_features
        edge_features_out = 1 if last else edge_features

        # We use 2 graph nets in order to operate on the upper and lower triangular parts of the matrix
        self.l1 = GraphNet(node_features=node_features, edge_features=edge_features_in, global_features=global_features,
                           hidden_size=hidden_size, skip_connection=(not first and skip_connections),
                           aggregate=aggr[0], activation=act, edge_features_out=edge_features)

        self.l2 = GraphNet(node_features=node_features, edge_features=edge_features, global_features=global_features,
                           hidden_size=hidden_size, aggregate=aggr[1], activation=act, edge_features_out=edge_features_out)

    def forward(self, x, edge_index, edge_attr, global_features):
        edge_embedding, node_embeddings, global_features = self.l1(x, edge_index, edge_attr, g=global_features)

        # flip row and column indices
        edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_embedding, node_embeddings, global_features = self.l2(node_embeddings, edge_index, edge_embedding, g=global_features)

        return edge_embedding, node_embeddings, global_features

############################
#         HELPERS          #
############################
def augment_features(data, skip_rhs=False):
    # transform nodes to include more features

    if skip_rhs:
        # use instead notde position as an input feature!
        data.x = torch.arange(data.x.size()[0], device=data.x.device).unsqueeze(1)

    data = torch_geometric.transforms.LocalDegreeProfile()(data)

    # diagonal dominance and diagonal decay from the paper
    row, col = data.edge_index
    diag = (row == col)
    diag_elem = torch.abs(data.edge_attr[diag])
    # remove diagonal elements by setting them to zero
    non_diag_elem = data.edge_attr.clone()
    non_diag_elem[diag] = 0

    row_sums = aggr.SumAggregation()(torch.abs(non_diag_elem), row)
    alpha = diag_elem / row_sums
    row_dominance_feature = alpha / (alpha + 1)
    row_dominance_feature = torch.nan_to_num(row_dominance_feature, nan=1.0)

    # compute diagonal decay features
    row_max = aggr.MaxAggregation()(torch.abs(non_diag_elem), row)
    alpha = diag_elem / row_max
    row_decay_feature = alpha / (alpha + 1)
    row_decay_feature = torch.nan_to_num(row_decay_feature, nan=1.0)

    data.x = torch.cat([data.x, row_dominance_feature, row_decay_feature], dim=1)

    return data


class ToLowerTriangular(torch_geometric.transforms.BaseTransform):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, data, order=None):
        if not self.inplace:
            data = data.clone()

        # TODO: if order is given use that one instead
        if order is not None:
            raise NotImplementedError("Custom ordering not yet implemented...")

        # transform the data into lower triag graph
        # this should be a data transformation (maybe?)
        rows, cols = data.edge_index[0], data.edge_index[1]
        fil = cols <= rows
        l_index = data.edge_index[:, fil]
        edge_embedding = data.edge_attr[fil]

        data.edge_index, data.edge_attr = l_index, edge_embedding
        return data


# In[11]:


class Preconditioner:
    def __init__(self, A, **kwargs):
        self.breakdown = False
        self.nnz = 0
        self.time = 0
        self.n = kwargs.get("n", 0)

    def timed_setup(self, A, **kwargs):
        start = time_function()
        self.setup(A, **kwargs)
        stop = time_function()
        self.time = stop - start

    def get_inverse(self):
        ones = torch.ones(self.n)
        offset = torch.zeros(1).to(torch.int64)

        I = torch.sparse.spdiags(ones, offset, (self.n, self.n))
        I = I.to(torch.float64)

        return I

    def get_p_matrix(self):
        return self.get_inverse()

    def check_breakdown(self, P):
        if np.isnan(np.min(P)):
            self.breakdown = True

    def __call__(self, x):
        return x

class LearnedPreconditioner(Preconditioner):
    def __init__(self, data, model, **kwargs):
        super().__init__(data, **kwargs)

        self.model = model
        self.spd = isinstance(model, NeuralIF)

        self.timed_setup(data, **kwargs)

        # # count non‐zeros in a torch Tensor / sparse tensor
        # def count_nnz(tensor):
        #     if tensor.layout == torch.sparse_coo:
        #         # number of nonzero entries in a sparse_coo_tensor
        #         return int(tensor._values().numel())
        #     else:
        #         # dense tensor: count != 0
        #         return int((tensor != 0).sum().item())

        # if self.spd:
        #     self.nnz = count_nnz(self.L)
        # else:
        #     nnz_L = count_nnz(self.L)
        #     nnz_U = count_nnz(self.U)
        #     # subtract the diagonal entries once:
        #     self.nnz = nnz_L + nnz_U - int(data.x.size(0))

        if self.spd:
            self.nnz = self.L.nnz
        else:
            self.nnz = self.L.nnz + self.U.nnz - data.x.shape[0]

    def setup(self, data, **kwargs):
        L, U, _ = self.model(data)

        self.L = L.to("cpu").to(torch.float64)
        self.U = U.to("cpu").to(torch.float64)

    def get_inverse(self):
        L_inv = torch.inverse(self.L.to_dense())
        U_inv = torch.inverse(self.U.to_dense())

        return U_inv@L_inv

    def get_p_matrix(self):
        return self.L@self.U

    def __call__(self, x):
        return fb_solve(self.L, self.U, x, unit_upper=not self.spd)


def fb_solve(L, U, r, unit_lower=False, unit_upper=False):
    print(L)
    y = L.solve_triangular(upper=False, unit=unit_lower, b=r)
    z = U.solve_triangular(upper=True, unit=unit_upper, b=y)
    return z

time_function = lambda: time.perf_counter()


# In[12]:


# helper functions
@torch.no_grad()
def validate(model, validation_loader, solve=False, solver="cg"):
    model.eval()
    acc_loss = 0.0
    num_loss = 0
    acc_solver_iters = 0.0

    for data in validation_loader:
        data = data.to(device)
        A, b = graph_to_matrix(data)

        if solve:
            preconditioner = LearnedPreconditioner(data, model)
            print(preconditioner)
            A_cpu = A.cpu().double()
            b_cpu = b.cpu().double()
            x0 = None

            start = time.time()
            if solver == "cg":
                iters, x_hat = preconditioned_conjugate_gradient(
                    A_cpu, b_cpu, M=preconditioner, x0=x0,
                    rtol=1e-6, max_iter=1000
                )
            else:
                iters, x_hat = gmres(
                    A_cpu, b_cpu, M=preconditioner, x0=x0,
                    atol=1e-6, max_iter=1000, left=False
                )
            acc_solver_iters += len(iters) - 1
        else:
            output, _, _ = model(data)
            # l = frobenius_loss(output, A)
            l = loss(data, output, config="frobenius")
            acc_loss += l.item()
            num_loss += 1

    if solve:
        avg_iters = acc_solver_iters / len(validation_loader)
        print(f"Validation iterations: {avg_iters:.2f}")
        return avg_iters
    else:
        avg_loss = acc_loss / num_loss
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss


# ## NeuralIF model

# In[13]:


class NeuralIF(nn.Module):
    # Neural Incomplete factorization
    def __init__(self, drop_tol=0, **kwargs) -> None:
        super().__init__()

        self.global_features = kwargs["global_features"]
        self.latent_size = kwargs["latent_size"]
        # node features are augmented with local degree profile
        self.augment_node_features = kwargs["augment_nodes"]

        num_node_features = 8 if self.augment_node_features else 1
        message_passing_steps = kwargs["message_passing_steps"]

        # edge feature representation in the latent layers
        edge_features = kwargs.get("edge_features", 1)

        self.skip_connections = kwargs["skip_connections"]

        self.mps = torch.nn.ModuleList()
        for l in range(message_passing_steps):
            # skip connections are added to all layers except the first one
            self.mps.append(MP_Block(skip_connections=self.skip_connections,
                                     first=l==0,
                                     last=l==(message_passing_steps-1),
                                     edge_features=edge_features,
                                     node_features=num_node_features,
                                     global_features=self.global_features,
                                     hidden_size=self.latent_size,
                                     activation=kwargs["activation"],
                                     aggregate=kwargs["aggregate"]))

        # node decodings
        self.node_decoder = MLP([num_node_features, self.latent_size, 1]) if kwargs["decode_nodes"] else None

        # diag-aggregation for normalization of rows
        self.normalize_diag = kwargs["normalize_diag"] if "normalize_diag" in kwargs else False
        self.diag_aggregate = aggr.SumAggregation()

        # normalization
        self.graph_norm = pyg.norm.GraphNorm(num_node_features) if ("graph_norm" in kwargs and kwargs["graph_norm"]) else None

        # drop tolerance and additional fill-ins and more sparsity
        self.tau = drop_tol
        self.two = kwargs.get("two_hop", False)

    def forward(self, data):
        # ! data could be batched here...(not implemented)

        if self.augment_node_features:
            data = augment_features(data, skip_rhs=True)

        # add additional edges to the data
        if self.two:
            data = TwoHop()(data)

        # * in principle it is possible to integrate reordering here.

        data = ToLowerTriangular()(data)

        # get the input data
        edge_embedding = data.edge_attr
        l_index = data.edge_index

        if self.graph_norm is not None:
            node_embedding = self.graph_norm(data.x, batch=data.batch)
        else:
            node_embedding = data.x

        # copy the input data (only edges of original matrix A)
        a_edges = edge_embedding.clone()

        if self.global_features > 0:
            global_features = torch.zeros((1, self.global_features), device=data.x.device, requires_grad=False)
            # feature ideas: nnz, 1-norm, inf-norm col/row var, min/max variability, avg distances to nnz
        else:
            global_features = None

        # compute the output of the network
        for i, layer in enumerate(self.mps):
            if i != 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)

            edge_embedding, node_embedding, global_features = layer(node_embedding, l_index, edge_embedding, global_features)

        # transform the output into a matrix
        return self.transform_output_matrix(node_embedding, l_index, edge_embedding, a_edges)

    def transform_output_matrix(self, node_x, edge_index, edge_values, a_edges):
        # force diagonal to be positive
        diag = edge_index[0] == edge_index[1]

        # normalize diag such that it has zero residual
        if self.normalize_diag:
            # copy the diag of matrix A
            a_diag = a_edges[diag]

            # compute the row norm
            square_values = torch.pow(edge_values, 2)
            aggregated = self.diag_aggregate(square_values, edge_index[0])

            # now, we renormalize the edge values such that they are the square root of the original value...
            edge_values = torch.sqrt(a_diag[edge_index[0]]) * edge_values / torch.sqrt(aggregated[edge_index[0]])

        else:
            # otherwise, just take the edge values as they are...
            # but take the square root as it is numerically better
            # edge_values[diag] = torch.exp(edge_values[diag])
            edge_values[diag] = torch.sqrt(torch.exp(edge_values[diag]))

        # node decoder
        node_output = self.node_decoder(node_x).squeeze() if self.node_decoder is not None else None

        # ! this if should only be activated when the model is in production!!
        if torch.is_inference_mode_enabled():

            # we can decide to remove small elements during inference from the preconditioner matrix
            if self.tau != 0:
                small_value = (torch.abs(edge_values) <= self.tau).squeeze()

                # small value and not diagonal
                elems = torch.logical_and(small_value, torch.logical_not(diag))

                # might be able to do this easily!
                edge_values[elems] = 0

                # remove zeros from the sparse representation
                filt = (edge_values != 0).squeeze()
                edge_values = edge_values[filt]
                edge_index = edge_index[:, filt]

            # ! this is the way to go!!
            # Doing pytorch -> scipy -> numml is a lot faster than pytorch -> numml on CPU
            # On GPU it is faster to go to pytorch -> numml -> CPU

            # convert to scipy sparse matrix
            # m = to_scipy_sparse_matrix(edge_index, matrix_values)
            m = torch.sparse_coo_tensor(edge_index, edge_values.squeeze(),
                                        size=(node_x.size()[0], node_x.size()[0]))
                                        # type=torch.double)

            # produce L and U seperatly
            l = SparseCSRTensor(m)
            u = SparseCSRTensor(m.T)

            return l, u, node_output

        else:
            # For training and testing (computing regular losses for examples.)
            # does not need to be performance optimized!
            # use torch sparse directly
            t = torch.sparse_coo_tensor(edge_index, edge_values.squeeze(),
                                        size=(node_x.size()[0], node_x.size()[0]))

            # normalized l1 norm is best computed here!
            # l2_nn = torch.linalg.norm(edge_values, ord=2)
            l1_penalty = torch.sum(torch.abs(edge_values)) / len(edge_values)

            return t, l1_penalty, node_output


# # 4. Training

# ## Set Training Configuration

# In[14]:


config = {
    "name": "experiment_1",
    "save": True,
    "seed": 42,
    "n": 0,
    "batch_size": 1,
    "num_epochs": 100,
    "dataset": "random",
    "loss": "frobenius",
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
    "num_neighbors": [15, 10]  # number of neighbours to sample in each hop (GraphSAGE sampling)
}

# Prepare output folder
if config["name"]:
    folder = f"results/{config['name']}"
else:
    folder = datetime.datetime.now().strftime("results/%Y-%m-%d_%H-%M-%S")
if config["save"]:
    os.makedirs(folder, exist_ok=True)
    save_dict_to_file(config, os.path.join(folder, "config.json"))


# In[16]:


# Seed for reproducibility
torch_geometric.seed_everything(config["seed"])

# Select model
model_args = {k: config[k] for k in [
    "latent_size", "message_passing_steps", "skip_connections",
    "augment_nodes", "global_features", "decode_nodes",
    "normalize_diag", "activation", "aggregate", "graph_norm",
    "two_hop", "edge_features", "normalize"
] if k in config}

use_gmres = False
if config["model"] in ("nif", "neuralif", "inf"):
    model = NeuralIF(**model_args)
else:
    raise ValueError("Unknown model type")

model.to(device)
print("Number of parameters:", count_parameters(model))

optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20
)


# ## Set DataLoader
# Run only 1 cell from the following

# Dataloader using NeighborSampler (GraphSAGE-inspired)
# - Random sampling of nodes at each layer to form a node's local neighbourhood

# In[17]:


from torch_geometric.loader import NeighborSampler
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import torch_sparse, torch_scatter

# get list of all files from directory in .pt format. each file represents 1 graph
train_dataset = FolderDataset(f"./data/Random/train/", n=config["n"], graph=True, size=None)
train_graphs   = [torch.load(path, weights_only=False) for path in train_dataset.files]
big_train_data = Batch.from_data_list(train_graphs)

# transform each sampler batch into a Data object
def to_data(batch_size, n_id, adjs):
    # Collect the node features for all participating nodes:
    x_sub = big_train_data.x[n_id]

    # Gather edges & edge‐attrs from each hop:
    rows, cols, eids = [], [], []
    for edge_index, e_id, size in adjs:
        rows.append(edge_index[0])
        cols.append(edge_index[1])
        eids.append(e_id)
    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    eid = torch.cat(eids, dim=0)

    edge_index_sub = torch.stack([row, col], dim=0)
    edge_attr_sub  = big_train_data.edge_attr[eid]

    data = Data(
        x          = x_sub,
        edge_index = edge_index_sub,
        edge_attr  = edge_attr_sub,
    )
    data.n = x_sub.size(0)            # if your model reads data.n
    return data

# build neighbour sampler while transforming all training entries to the Data object
train_loader = NeighborSampler(               # implementation of abstract class DataLoader
    edge_index   = big_train_data.edge_index,
    sizes        = config["num_neighbors"],
    node_idx     = None,                      # sample seeds from all nodes
    num_nodes    = big_train_data.num_nodes,
    return_e_id  = True,                      # we need the e_id to slice edge_attr
    transform    = to_data,                   # turn each sample into Data
    batch_size   = config["batch_size"],      # # of seeds per iteration
    shuffle      = True,
    num_workers  = 4,
    pin_memory   = True,
)

first_sample = next(iter(train_loader))
print(type(first_sample))    # torch_geometric.data.Data
print(first_sample)          # Data object with format x, edge_index, edge_attr, n, etc.
print(len(train_loader))

# load the full graph for the validation dataloader
validation_loader = get_dataloader(config["dataset"], config["n"], batch_size=1, spd=(not gmres), mode="val")


# Dataloader for full graph (Original dataloader)

# In[19]:


# 1. retrieve the FolderDataset object for each of train and validation loader based on .pt files
# 2. pass the FolderDataset object into torch.util's DataLoader as the dataset field
# 3. return this DataLoader object

# the loader below passes train_dataset directly to torch.util's DataLoader class

train_loader = get_dataloader(config["dataset"], config["n"], config["batch_size"],
                                  spd=not gmres, mode="train")

print(train_loader.dataset[0])
print(len(train_loader))

validation_loader = get_dataloader(config["dataset"], config["n"], 1, spd=(not gmres), mode="val")


# ## Training Loop

# In[20]:


from torch_geometric.utils import add_self_loops


# training loop
best_val = float("inf")
logger = TrainResults(folder)
total_it = 0

for epoch in range(config["num_epochs"]):
    running_loss = 0.0
    start_epoch = time.perf_counter()

    for data in train_loader:
        total_it += 1
        model.train()

        start = time.perf_counter()
        data = data.to(device)

        ### resolving the unmatching dimension bug for NeighborSampler class
        # 1) override n properly
        data.n = int(data.x.size(0))

        # 2) add self-loops so each node has at least one incoming edge
        data.edge_index, data.edge_attr = add_self_loops(
            data.edge_index,
            data.edge_attr,
            fill_value=0.0,
            num_nodes=data.n
        )
        ###

        # print(f"Input training data to the model is {data}")
        output, reg, _ = model(data)
        # print(f"Output from the model is {output} and reg term is {reg}")

        l = loss(output, data, c=reg, config=config["loss"])
        l.backward()

        # gradient clipping or manual norm
        if config["gradient_clipping"]:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clipping"]
            )
        else:
            total_norm = sum(
                p.grad.detach().data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            )
            grad_norm = (total_norm ** 0.5) / config["batch_size"]

        optimizer.step()
        optimizer.zero_grad()

        running_loss += l.item()
        logger.log(l.item(), grad_norm, time.perf_counter() - start)

        # periodic validation
        if total_it % 1000 == 0:
            val_metric = validate(
                model, validation_loader, solve=True,
                solver="gmres" if use_gmres else "cg"
            )
            logger.log_val(None, val_metric)
            if val_metric < best_val:
                best_val = val_metric
                if config["save"]:
                    torch.save(model.state_dict(), f"{folder}/best_model.pt")

    epoch_time = time.perf_counter() - start_epoch
    print(f"Epoch {epoch+1} — loss: {running_loss/len(train_loader):.4f}, time: {epoch_time:.1f}s")
    if config["save"]:
        torch.save(model.state_dict(), f"{folder}/model_epoch{epoch+1}.pt")


# In[33]:


# save results
if config["save"]:
    logger.save_results()
    torch.save(model.to(torch.float).state_dict(), f"{folder}/final_model.pt")


# In[ ]:


# test printout
print("Best validation performance:", best_val)

