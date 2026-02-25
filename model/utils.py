import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

OP_LIST = [
    "Seq Scan", "Hash Join", "Index Scan", "Hash", "Materialize",
    "Nested Loop", "ModifyTable", "Sort", "Merge Join", "Index Only Scan",
    "Aggregate", "GroupAggregate", "HashAggregate", "Limit", "Append",
    "Subquery Scan", "CTE Scan", "Result", "Unique", "WindowAgg"
]

def get_one_hot_op(op_name):
    one_hot = [0.0] * (len(OP_LIST) + 1)
    if op_name in OP_LIST:
        one_hot[OP_LIST.index(op_name)] = 1.0
    else:
        one_hot[-1] = 1.0
    return one_hot

def plan_to_graph(plan):
    nodes, edge_index = [], []
    def walk(node, parent_idx=None):
        node_idx = len(nodes)
        op_features = get_one_hot_op(node.get("Node Type"))
        numeric_features = [
            float(np.log1p(node.get("Plan Rows", 0))),
            float(np.log1p(node.get("Plan Width", 0))),
            float(np.log1p(node.get("Startup Cost", 0))),
            float(np.log1p(node.get("Total Cost", 0))),
            float(np.log1p(node.get("Shared Hit Blocks", 0))),
            float(np.log1p(node.get("Shared Read Blocks", 0)))
        ]
        nodes.append(op_features + numeric_features)
        if parent_idx is not None:
            edge_index.append([node_idx, parent_idx])
        if "Plans" in node:
            for subplan in node["Plans"]: walk(subplan, node_idx)

    walk(plan)
    x = torch.tensor(nodes, dtype=torch.float)
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    return x, ei

def get_prepared_data(raw_data):
    """Returns the dataset and the fitted scaler."""
    log_runtimes = np.log1p([item['runtime'] for item in raw_data]).reshape(-1, 1)
    scaler = StandardScaler()
    scaled_runtimes = scaler.fit_transform(log_runtimes)

    dataset = []
    for i, item in enumerate(raw_data):
        x, ei = plan_to_graph(item['plan'])
        y = torch.tensor([[scaled_runtimes[i][0]]], dtype=torch.float)
        dataset.append(Data(x=x, edge_index=ei, y=y))
    return dataset, scaler