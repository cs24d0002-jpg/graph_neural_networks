from sklearn.metrics import f1_score
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
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y.float()) # PPI labels are already floats
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, mask = None, data= None, loader = None):
    model.eval()
    if(loader is None):
        out = model(data.x, data.edge_index)
        return compute_metrics(out, data.y, mask)
    else:
        all_preds, all_targets = [], []
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            # Threshold at 0.5 for multi-label prediction
            pred = (logits > 0).float() 
            all_preds.append(pred.cpu())
            all_targets.append(batch.y.cpu())
        
        y_true = torch.cat(all_targets, dim=0).numpy()
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_true = y_true.flatten() 
        y_pred = y_pred.flatten()
        return {'f1': f1_score(y_true, y_pred, average='micro')}