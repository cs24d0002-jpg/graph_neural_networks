import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_reddit(model, loader, optimizer, device):
    """Single training epoch for Reddit using neighbor sampling."""
    model.train()
    total_loss = total_correct = total_examples = 0

    pbar = tqdm(total=len(loader.dataset), desc='Training', leave=False)
    for batch in loader:
        # Move the entire batch to GPU
        batch = batch.to(device)
        optimizer.zero_grad()
        y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == batch.y[:batch.batch_size]).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)

    pbar.close()
    return total_loss / total_examples, total_correct / total_examples