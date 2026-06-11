import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate_reddit(model, data, subgraph_loader, mask, device):
    """Evaluates the model on Reddit."""
    model.eval()

    # Ensure data.x is on the same device as the model's parameters
    # We keep it on CPU and let the inference method move pieces as needed.
    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y[mask].cpu()
    y_pred = out.argmax(dim=-1)[mask].cpu()
    acc = (y_pred == y_true).sum().item() / mask.sum().item()
    return acc