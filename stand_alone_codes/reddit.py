import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader

class RedditSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RedditSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


dataset = Reddit(root='data/Reddit')
data = dataset[0]

# T400 Optimization: small batch_size and limited neighbors
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RedditSAGE(in_channels=dataset.num_features, hidden_channels=256, out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # In NeighborLoader, we only care about the predictions of the 
        # original 'batch_size' nodes, which are at the start of the batch
        out = model(batch.x, batch.edge_index)
        batch_size = batch.batch_size
        loss = criterion(out[:batch_size], batch.y[:batch_size])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        batch_size = batch.batch_size
        
        pred = out[:batch_size].argmax(dim=-1)
        total_correct += (pred == batch.y[:batch_size]).sum().item()
        total_nodes += batch_size
    return total_correct / total_nodes


# --- Execution Loop ---
best_val_acc = 0
epochs = 100 # Reddit usually needs fewer epochs than Cora due to more updates per epoch

for epoch in range(1, epochs + 1):
    # 1. Run Training
    loss = train() 
    
    # 2. Run Validation (Using the sampled test function)
    val_acc = test(val_loader)
    
    # 3. Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'reddit_sage_best.pt')
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} (Saved!)")
    else:
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

# 4. Final Test
model.load_state_dict(torch.load('reddit_sage_best.pt'))
test_acc = test(test_loader)
print(f"\nFinal Reddit Test Accuracy: {test_acc:.4f}")