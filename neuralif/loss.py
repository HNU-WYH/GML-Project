import warnings
import torch

from apps.data import graph_to_matrix


warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def frobenius_loss(L, A, sparse=True):
    # * Cholesky decomposition style loss
    
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    
    if sparse:
        # Not directly supported in pyotrch:
        # https://github.com/pytorch/pytorch/issues/95169
        # https://github.com/rusty1s/pytorch_sparse/issues/45
        r = L@U - A
        return torch.norm(r)
        
    else:
        A = A.cpu().to_dense().squeeze()
        L = L.cpu().to_dense().squeeze()
        U = U.cpu().to_dense().squeeze()
        
        return torch.linalg.norm(L@U - A, ord="fro")


def sketched_loss(L, A, c=None, normalized=False, node_norm = None):
    # both cholesky and LU decomposition
    A = A.to(L.device)

    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    
    eps = 1e-8
    
    z = torch.randn((A.shape[0], 1), device=L.device)

    # if normalized:
    # z = z / torch.linalg.norm(z) # z-vector also should have unit length
    Mz = L @ (U @ z) - A @ z

    if node_norm is not None:
        norm = (Mz.squeeze().pow(2) * node_norm.squeeze()).sum()
    else:
        norm = torch.linalg.vector_norm(Mz, ord=2) # vector norm
    
    if normalized and c is None:
        norm = norm / torch.linalg.vector_norm(A@z, ord=2)
    elif normalized:
        norm = norm / (c + eps)
    
    return norm


def supervised_loss(L, A, x):
    # Note: Ax = b
    
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    
    if x is None:
        # if x is None, we recompute the solution in every time step.
        with torch.no_grad():
            b = torch.randn((A.shape[0], 1), device=L.device)
            x = torch.linalg.solve(A.to_dense(), b)
    else:
        b = A@x
        
    res = L@(U@x) - b
    return torch.linalg.vector_norm(res, ord=2)


def dircet_min_loss(L, A, x):
    # get L and U factors
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
        
    if x is None:
        # if x is None, we recompute the solution in every time step.
        with torch.no_grad():
            b = torch.randn((A.shape[0], 1), device=L.device)
            x = torch.linalg.solve(A.to_dense(), b)
    else:
        b = A@x
        
    res = L@(U@x)
    return torch.linalg.vector_norm(res, ord=2)


def combined_loss(L, A, x, w=1):
    # combined loss
    loss1 = sketched_loss(L, A)
    loss2 = supervised_loss(L, A, x)
    
    return w * loss1 + loss2


def loss(output, data, config=None, **kwargs):
    
    # load the data
    with torch.no_grad():
        A, b = graph_to_matrix(data)
    
    # compute loss
    if config is None:
        # this is the regular loss used to train NeuralIF
        if "node_norm" in kwargs:
            l = sketched_loss(output, A, normalized=False, node_norm = kwargs["node_norm"])
        else:
            l = sketched_loss(output, A, normalized=False)
    
    elif config == "normalized":
        l = sketched_loss(output, A, kwargs.get("c", None), normalized=True)
    
    elif config == "supervised":
        l = supervised_loss(output, A, data.s.squeeze())
        
    elif config == "inverted":
        l = supervised_loss(output, A, None)
    
    elif config == "combined":
        l = combined_loss(output, A, data.s.squeeze())
        
    elif config == "combined-supervised":
        l = combined_loss(output, A, None)
    
    elif config == "frobenius":
        l = frobenius_loss(output, A, sparse=False)
    
    else:
        raise ValueError("Invalid loss configuration")
    
            
    return l
