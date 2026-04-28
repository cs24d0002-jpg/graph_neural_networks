import torch

from src.evaluate.metrics import compute_metrics

def train_one_epoch(model, data, optimizer, criterion,loader=None):
    model.train()
    if loader == None:
        optimizer.zero_grad()
        # Handle both Homogeneous (data.x) and Heterogeneous data
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    else:
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            if hasattr(batch, 'batch_size'): # For NeighborLoader
                loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            else: # For PPI
                loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Pass the specific mask to your metrics calculator
    # This assumes compute_metrics is defined as we discussed earlier
    return compute_metrics(out, data.y, mask)