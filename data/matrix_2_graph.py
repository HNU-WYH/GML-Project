"""
In the original work, they use the 8 dimensional node feature.
But I found that the time requires to construct such a graph with 8d node features is more than doing an incomplete cholesky
Another paper (Li et al) used rhs vector b as the graph signal (node features),
and even some researchers give up node feature use 1 as the node feature for all nodes.
I uploaded this two paper to github repo also.
Thus I think use rhs vector b is enough for our work
"""

import torch
import numpy as np

from typing import Union
from scipy.sparse import coo_matrix
from torch_geometric.data import Data


def matrix_to_graph_sparse(A: coo_matrix, b: Union[list, np.ndarray]):
    """
    Convert a sparse matrix A (n×n) in COO Format, and right hand side vector b (n×1) into graph g = (V,E)
    where
    - each edge e_ij represents a nonzero entry a_ij in matrix A
    - each node v_i represents the column/row index, the vector entry b_i is used as the node feature (Li et al.)

    :param A: a sparse matrix with size = (n, n) in the form of scipy.sparse.coo_matrix (A.row, A.col, A.data)
    :param b: list of ndarray with size = (n, ), denoting the rhs vector
    :return: data: torch_geometric.data.Data，containing the following members:
        - x = [n, 1] # graph signal & node feature, representing rhs vector b (n×1)
        - edge_index = [2, nnz] # index for edges (source, target)
        - edge_attr = [nnz, 1] # edge feature (values of non-zero entries)
        - num_edges = nnz # the non-zero entries in the sparse matrix A
    """
    # compute all features required to define the graph data
    edge_index = torch.tensor(list(map(lambda x: [x[0], x[1]], zip(A.row, A.col))), dtype=torch.long)
    edge_features = torch.tensor(list(map(lambda x: [x], A.data)), dtype=torch.float)
    node_features = torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    # edge_index.t().contiguous() transpose and make the content contiguous in memory
    return data

def matrix_to_graph(A: np.ndarray, b: Union[list, np.ndarray]):
    """
    similar to matrix_to_graph_sparse(A,b), but for the case when A is in the format of np.ndarray
    """
    return matrix_to_graph_sparse(coo_matrix(A), b)


def graph_to_matrix(data:Data, normalize=False):
    """
    convert a graph data into the sparse matrix A (torch_geometric.data.Data) and rhs vector b (np.ndarray)
    :param data: a torch_geometric.data.Data object containing data.edge_index [2, nnz], data.edge_attr [nnz, 1], data_x [n, 1]
    :param normalize: bool, default false, if true then normalize the rhs vector b
    :return: torch.sparse_coo_tensor A and torch.tensor b
    """
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0].squeeze(), requires_grad=False)
    b = data.x[:, 0].squeeze()

    if normalize:
        b = b / torch.linalg.norm(b)

    return A, b