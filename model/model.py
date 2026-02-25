import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool, AttentionalAggregation

class HybridQueryGNN(torch.nn.Module):
    def __init__(self, input_dim=26): # 21 one-hot + 5 numeric (Total Cost is excluded from GNN)
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, 32, heads=2)
        self.bn1 = BatchNorm(64)
        self.conv2 = GATv2Conv(64, 32, heads=1)
        self.bn2 = BatchNorm(32)
        self.dropout = torch.nn.Dropout(0.4)

        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
        self.att_pool = AttentionalAggregation(gate_nn=gate_nn)

        self.final_head = torch.nn.Sequential(
            torch.nn.Linear(32 + 1, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Slice to hide 'Total Cost' (index 24) from message passing to prevent leakage
        x_no_cost = torch.cat([x[:, :24], x[:, 25:]], dim=1)

        h = F.elu(self.bn1(self.conv1(x_no_cost, edge_index)))
        h = self.dropout(h)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))

        h_pooled = self.att_pool(h, batch)

        # Re-inject the shortcut: Total Cost (Index 24)
        root_costs = global_mean_pool(x[:, 24].unsqueeze(1), batch)

        combined = torch.cat([h_pooled, root_costs], dim=1)
        return self.final_head(combined)