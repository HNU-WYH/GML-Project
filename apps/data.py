from glob import glob
from typing import Union

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def matrix_to_graph_sparse(A: coo_matrix, b: Union[list, np.ndarray]):
    """
    Convert a sparse matrix A (n×n) in COO Format, and right hand side vector b (n×1) into graph g = (V,E)
    Here A is a full matrix, not a lower or upper triangular matrix
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


def get_dataloader(dataset, n=0, batch_size=1, spd=True, mode="train", size=None, graph=True):
    # Setup datasets
    
    if dataset == "random":
        data = FolderDataset(f"./dataset/{mode}/", n, size=size, graph=graph)
    
    else:
        raise NotImplementedError("Dataset not implemented, Available: random")
    
    # Data Loaders
    if mode == "train":
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
    
    return dataloader


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, n, graph=True, size=None) -> None:
        super().__init__()
        
        self.graph = True
        assert self.graph, "Graph keyword is depracated, only graph=True is supported."
                
        if n != 0:
            if self.graph:
                self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder+'*.pt')))
            else:
                self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder+'*.npz')))
        else:
            file_ending = "pt" if self.graph else "npz"
            self.files = list(glob(folder+f'*.{file_ending}'))
        
        if size is not None:
            assert len(self.files) >= size, f"Only {len(self.files)} files found in {folder} with n={n}"
            self.files = self.files[:size]
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found in {folder} with n={n}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.graph:
            g = torch.load(self.files[idx], weights_only=False)
        
        else:
            # deprecated...
            d = np.load(self.files[idx], allow_pickle=True)
            g = matrix_to_graph(d["A"], d["b"])
        
        return g
