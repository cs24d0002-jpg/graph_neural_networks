import os
from torch_geometric.datasets import Planetoid, PPI, Reddit
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader,NeighborLoader

def load_data(dataset_name, root='/home/dell/IIITDM/My Research/Implementation/graph_neural_networks/data'):
    """
    Standardizes loading to prevent redownloads.
    Note: PyG creates a subfolder named {dataset_name} inside {root}.
    """
    
    
    # 1. Handle Planetoid (Cora, CiteSeer, PubMed)
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(
            root=root, 
            name=dataset_name, 
            transform=T.NormalizeFeatures()
        )
        print(f"--- {dataset_name} loaded from {dataset.root} ---")
        return dataset
    # 2. Handle PPI
    elif dataset_name.lower() == 'ppi':
        
        # dataset = PPI(root=os.path.join(root, 'PPI'))
        train_dataset = PPI(root=os.path.join(root, 'PPI'), split='train')
        val_dataset = PPI(root=os.path.join(root, 'PPI'), split='val')
        test_dataset = PPI(root=os.path.join(root, 'PPI'), split='test')

        # Use a small batch_size for your T400 (e.g., 2 graphs at a time)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        in_channels = train_dataset.num_features
        out_channels = train_dataset.num_classes
        
        return train_loader, val_loader, test_loader, in_channels, out_channels
        
    # 3. Handle Reddit
    elif dataset_name.lower() == 'reddit':
        dataset = Reddit(root=os.path.join(root, 'Reddit'))
        data = dataset[0]
        train_loader = NeighborLoader(
            data,
            num_neighbors=[25, 10], # Sampling sizes per hop
            batch_size=128,        # Number of target nodes per batch
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=0           # Disable multi-processing for stability
        )

        # For validation/testing, we also use a loader to avoid OOM
        val_loader = NeighborLoader(
            data,
            num_neighbors=[25, 10],
            batch_size=128,
            input_nodes=data.val_mask
        )
        # Define the Test Loader
        test_loader = NeighborLoader(
            data,
            num_neighbors=[25, 10], # Same sampling hops as training
            batch_size=512,         # Can be slightly higher than training since no gradients are stored
            input_nodes=data.test_mask,
            shuffle=False,          # No need to shuffle for testing
            num_workers=0           # Keep at 0 for T400 stability
        )
        return train_loader, val_loader, test_loader, dataset.num_features, dataset.num_classes
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet.")

    