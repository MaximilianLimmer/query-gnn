import torch
import json
import joblib
import numpy as np
import os
from model.model import HybridQueryGNN
from model.utils import plan_to_graph
from torch_geometric.data import Batch, Data

# Ground truth values for comparison
ground_truth = {
    "fast_plan.json": 0.08,
    "medium_plan.json": 0.28,
    "slow_plan.json": 0.70,
}

def predict(json_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and weights
    model = HybridQueryGNN(input_dim=27)
    model.load_state_dict(torch.load("query_gnn.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load preprocessing scaler
    scaler = joblib.load("scaler.pkl")

    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    with open(json_file, 'r') as f:
        plan = json.load(f)

    # Convert JSON to Graph
    x, edge_index = plan_to_graph(plan)
    data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
    data = data.to(device)

    # Inference
    with torch.no_grad():
        out = model(data).cpu().numpy()
        log_ms = scaler.inverse_transform(out)
        pred_ms = np.expm1(log_ms)[0][0]

    # Output
    filename = os.path.basename(json_file)
    print(f"\n--- Analysis: {filename} ---")
    print(f"Prediction: {pred_ms:.3f} ms")

    if filename in ground_truth:
        actual_ms = ground_truth[filename]
        q_error = max(pred_ms / actual_ms, actual_ms / pred_ms)
        print(f"Actual:     {actual_ms:.3f} ms")
        print(f"Q-Error:    {q_error:.2f}x")
    print("-" * 30)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_plan_json>")