import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,LayerNorm

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1: Conv -> Activation -> Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2: Final Prediction
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

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
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=8, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        # GAT is sensitive to heads; hidden_channels here is per-head
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x) # GAT papers usually use ELU
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT_PPI(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Reduced from 4 heads to 2; hidden units from 256 to 64
        self.conv1 = GATConv(in_channels, 64, heads=2, concat=True)
        self.ln1 = LayerNorm(64 * 2)
        
        self.conv2 = GATConv(64 * 2, 64, heads=2, concat=True)
        self.ln2 = LayerNorm(64 * 2)
        
        # Final layer: 2 heads averaged (concat=False)
        self.conv3 = GATConv(64 * 2, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.elu(x)
        
        # No normalization on the final output layer
        x = self.conv3(x, edge_index)
        return x