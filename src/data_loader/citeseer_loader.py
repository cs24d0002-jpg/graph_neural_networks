import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_citeseer(root='data/', normalize_features=True):
    """Load CiteSeer dataset.
    Returns:
        dataset (Planetoid): Dataset object (has .num_classes, .num_features)
        data (Data): Single graph object
    """
    transform = NormalizeFeatures() if normalize_features else None
    dataset = Planetoid(root=root, name='CiteSeer', transform=transform)
    data = dataset[0]
    return dataset, data