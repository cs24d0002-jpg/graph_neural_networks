import torch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE_PPI(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE_PPI, self).__init__()
        # Layer 1: Aggregates features from neighbors
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        # Layer 2: Final projection to 121 classes
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
# 1. Load Data
train_dataset = PPI(root='data/PPI', split='train')
val_dataset = PPI(root='data/PPI', split='val')
test_dataset = PPI(root='data/PPI', split='test')

# Low batch size (2) to ensure 4GB VRAM stability
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

# 2. Initialize Model, Optimizer, and Loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE_PPI(in_channels=50, hidden_channels=256, out_channels=121).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss() # Essential for Multi-label

# 3. Training Function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y) # PPI labels are already floats
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 4. Evaluation Function (Micro-F1)
@torch.no_grad()
def test(loader):
    model.eval()
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
    return f1_score(y_true, y_pred, average='micro')

# 5. Execution Loop
best_val_f1 = 0
for epoch in range(1, 201):
    loss = train()
    val_f1 = test(val_loader)
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'sage_ppi_best.pt')
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}')

# Final Test Accuracy
model.load_state_dict(torch.load('sage_ppi_best.pt'))
final_f1 = test(test_loader)
print(f'Final Test Micro-F1 Score: {final_f1:.4f}')