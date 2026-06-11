import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_cora(root='data/', normalize_features=True):
    """Load Cora dataset.
    Returns:
        dataset (Planetoid): The dataset object (has .num_classes, .num_features)
        data (Data): The single graph object (has .x, .edge_index, .y, masks)
    """
    transform = NormalizeFeatures() if normalize_features else None
    dataset = Planetoid(root=root, name='Cora', transform=transform)
    data = dataset[0]   # Cora is a single graph
    return dataset, data