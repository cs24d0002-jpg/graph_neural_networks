import torch
import torch.nn.functional as F

def train_full_batch(model, data, optimizer, mask, device):
    """Single training epoch for a full‑batch graph."""
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()