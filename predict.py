# predict.py
import torch
import json
import joblib
import numpy as np
from model.model import HybridQueryGNN
from model.utils import plan_to_graph
from torch_geometric.data import Batch, Data

def predict(json_file):
    # 1. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridQueryGNN(input_dim=27)
    model.load_state_dict(torch.load("query_gnn.pth", map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Scaler
    scaler = joblib.load("scaler.pkl")

    # 3. Process Plan
    with open(json_file, 'r') as f:
        plan = json.load(f)

    x, edge_index = plan_to_graph(plan)
    data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
    data = data.to(device)

    # 4. Infer
    with torch.no_grad():
        out = model(data).cpu().numpy()
        log_ms = scaler.inverse_transform(out)
        ms = np.expm1(log_ms)

    print(f"\nTarget Query: {json_file}")
    print(f"Predicted Runtime: {ms[0][0]:.3f} ms")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_plan_json>")