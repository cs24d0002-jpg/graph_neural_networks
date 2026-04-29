import torch

from src.evaluate.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, optimizer, criterion, loader=None, data=None):
    model.train()
    if loader is None:
        optimizer.zero_grad()
        # Handle both Homogeneous (data.x) and Heterogeneous data
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    else:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(loader)

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Pass the specific mask to your metrics calculator
    # This assumes compute_metrics is defined as we discussed earlier
    return compute_metrics(out, data.y, mask)