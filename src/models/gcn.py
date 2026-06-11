import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        """Layer-wise inference for large graphs to prevent OOM."""
        # x_all is the full node features (on CPU)
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                # Move needed subgraph data to GPU
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                    x = F.dropout(x, p=self.dropout, training=False)
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all