import torch
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader

def load_reddit(root='data/Reddit'):
    """Load Reddit dataset and create train + inference loaders."""
    print("Loading Reddit dataset...")
    dataset = Reddit(root=root)
    data = dataset[0]   # Keep on CPU, loaders will sample from it

    kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
    
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[25, 10],
        shuffle=True,
        **kwargs,
    )

    # Inference loader: sample all neighbors for full-graph evaluation
    subgraph_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[-1],
        shuffle=False,
        **kwargs,
    )

    # No deletion! Both loaders now safely share the original data.
    return dataset, data, train_loader, subgraph_loader