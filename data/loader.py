import torch
import numpy as np

from glob import glob
from torch_geometric.loader import DataLoader

from data.matrix_2_graph import matrix_to_graph

"""
copy from the repo, if we want to use mini-batching SGD to partition graph data, then we need to revise these two functions by:
1. NeighborSampler
2. ClusterData
3. GraphSAINT

I left this unchanged for further revision
"""
def get_dataloader(dataset, n=0, batch_size=1, spd=True, mode="train", size=None, graph=True):
    # Setup datasets

    if dataset == "random":
        data = FolderDataset(f"./data/Random/{mode}/", n, size=size, graph=graph)

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
                self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder + '*.pt')))
            else:
                self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder + '*.npz')))
        else:
            file_ending = "pt" if self.graph else "npz"
            self.files = list(glob(folder + f'*.{file_ending}'))

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