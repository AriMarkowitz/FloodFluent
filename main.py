from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn import to_hetero
import torch

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False)
        self.conv2 = GATv2Conv((-1,-1), out_channels,add_self_loops=False)

    def forward(self, x, edge_idx):
        x = torch.relu(self.conv1(x, edge_idx))
        x = self.conv2(x, edge_idx)
        return x

model1 = GNN(hidden_channels=64, out_channels=1)
model1 = to_hetero(model1, metadata, aggr="sum")