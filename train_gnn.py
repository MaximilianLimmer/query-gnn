import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GlobalAttention
from sklearn.metrics import r2_score
from torch_geometric.nn import BatchNorm, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.nn import AttentionalAggregation

OP_LIST = [
    "Seq Scan", "Hash Join", "Index Scan", "Hash", "Materialize",
    "Nested Loop", "ModifyTable", "Sort", "Merge Join", "Index Only Scan",
    "Aggregate", "GroupAggregate", "HashAggregate", "Limit", "Append",
    "Subquery Scan", "CTE Scan", "Result", "Unique", "WindowAgg"
]

def run_error_analysis(model, test_loader, target_scaler, raw_data):
    model.eval()
    results = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Get predictions and actuals back to real ms
            out = model(batch).cpu().numpy()
            log_preds = target_scaler.inverse_transform(out)
            log_actuals = target_scaler.inverse_transform(batch.y.cpu().numpy())

            preds = np.expm1(log_preds).flatten()
            actuals = np.expm1(log_actuals).flatten()

            # Calculate metrics for each query in the batch
            for j in range(len(preds)):
                error = preds[j] - actuals[j]
                abs_error = abs(error)
                results.append({
                    'actual': actuals[j],
                    'predicted': preds[j],
                    'error': error,
                    'abs_error': abs_error,
                    'pct_error': (abs_error / actuals[j]) * 100 if actuals[j] > 0 else 0
                })

    df_err = pd.DataFrame(results)

    print("\n--- 🔍 ERROR ANALYSIS REPORT ---")
    print(f"Overall MAE: {df_err['abs_error'].mean():.4f} ms")
    print(f"Median Absolute Error: {df_err['abs_error'].median():.4f} ms")
    print(f"95th Percentile Error: {df_err['abs_error'].quantile(0.95):.4f} ms")

    # 1. THE TOP 5 OFFENDERS (Outlier Analysis)
    print("\n🚀 Top 5 Worst Predictions (Highest Absolute Error):")
    top_offenders = df_err.sort_values(by='abs_error', ascending=False).head(5)
    print(top_offenders[['actual', 'predicted', 'abs_error', 'pct_error']])

    # 2. BIAS CHECK (Under vs Over predicting)
    under_pred = len(df_err[df_err['error'] < 0])
    over_pred = len(df_err[df_err['error'] > 0])
    print(f"\n⚖️ Bias: Under-predicted {under_pred} times, Over-predicted {over_pred} times.")

    # 3. RESIDUAL PLOT
    plt.figure(figsize=(10, 6))
    plt.scatter(df_err['actual'], df_err['error'], alpha=0.4, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot: Where is the model drifting?")
    plt.xlabel("Actual Runtime (ms)")
    plt.ylabel("Prediction Error (ms)")
    plt.grid(True, alpha=0.3)
    plt.savefig("residuals.png")

    return df_err

def get_one_hot_op(op_name):
    # Length is 20 (OP_LIST) + 1 (Other) = 21
    one_hot = [0.0] * (len(OP_LIST) + 1)
    if op_name in OP_LIST:
        one_hot[OP_LIST.index(op_name)] = 1.0
    else:
        one_hot[-1] = 1.0 # The "catch-all" for anything else
    return one_hot

class HybridQueryGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 21 (one-hot) + 5 (numeric features) = 26
        self.conv1 = GATv2Conv(26, 32, heads=2)
        self.bn1 = BatchNorm(64)
        self.conv2 = GATv2Conv(64, 32, heads=1)
        self.bn2 = BatchNorm(32)
        self.dropout = torch.nn.Dropout(0.4)

        # UPDATED: Using AttentionalAggregation instead of GlobalAttention
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

        # Hide index 24 (Total Cost) from GNN message passing
        x_no_cost = torch.cat([x[:, :24], x[:, 25:]], dim=1)

        h = F.elu(self.bn1(self.conv1(x_no_cost, edge_index)))
        h = self.dropout(h)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))

        # UPDATED: New pooling call
        h_pooled = self.att_pool(h, batch)

        # Shortcut: Grab Total Cost (Index 24)
        root_costs = global_mean_pool(x[:, 24].unsqueeze(1), batch)

        combined = torch.cat([h_pooled, root_costs], dim=1)
        return self.final_head(combined)

# --- Keep your plan_to_graph function exactly as it is ---
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
            # KEEP THIS: Child points to parent (Bottom-up flow)
            edge_index.append([node_idx, parent_idx])

        if "Plans" in node:
            for subplan in node["Plans"]: walk(subplan, node_idx)

    walk(plan)
    x = torch.tensor(nodes, dtype=torch.float)
    if not edge_index:
        ei = torch.empty((2, 0), dtype=torch.long)
    else:
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return x, ei

def final_evaluation(model, test_loader, scaler):
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch).cpu().numpy()

            # Step 1: Reverse the Standard Scaling (e.g., 0.5 -> 6.2)
            log_preds = scaler.inverse_transform(out)
            log_actuals = scaler.inverse_transform(batch.y.cpu().numpy())

            # Step 2: Reverse the Log transform (e.g., 6.2 -> 500ms)
            all_preds.extend(np.expm1(log_preds).flatten())
            all_actuals.extend(np.expm1(log_actuals).flatten())

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Metrics
    mae = np.mean(np.abs(all_preds - all_actuals))
    r2 = r2_score(all_actuals, all_preds)

    print(f"\n🏆 CORRECTED GRADE 🏆")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.4f} ms")

    # Re-plot using real MS values
    plt.figure(figsize=(8,8))
    plt.scatter(all_actuals, all_preds, alpha=0.3)
    plt.plot([0, max(all_actuals)], [0, max(all_actuals)], color='red')
    plt.title("Actual vs Predicted (Standardized GNN)")
    plt.savefig("final_performance.png")

if __name__ == "__main__":
    # 1. Load Data
    with open("query_data.json", "r") as f:
        raw_data = json.load(f)

    # 2. TARGET SCALING (Crucial for Batch Norm stability)
    # We take log first, then StandardScale (Mean=0, Std=1)
    log_runtimes = np.log1p([item['runtime'] for item in raw_data]).reshape(-1, 1)
    target_scaler = StandardScaler()
    scaled_runtimes = target_scaler.fit_transform(log_runtimes)

    dataset = []
    for i, item in enumerate(raw_data):
        x, ei = plan_to_graph(item['plan'])
        # y is now a tiny, centered number (e.g., -0.5 or 1.2)
        y = torch.tensor([[scaled_runtimes[i][0]]], dtype=torch.float)
        dataset.append(Data(x=x, edge_index=ei, y=y))

    # 3. SPLIT & LOAD
    random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:train_size], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset[train_size:], batch_size=32, shuffle=False)

    # 4. INITIALIZE
    model = HybridQueryGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    # ADD THIS: Lowers the LR if Val MAE stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = torch.nn.MSELoss()

    # 5. TRAINING LOOP
    print("🧠 Training with Input Batch Norm + Target Scaling...")
    for epoch in range(300):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            model.eval()
            val_errors_ms = []
            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch).cpu().numpy()

                    # Un-scale the prediction to get back to milliseconds
                    log_preds = target_scaler.inverse_transform(out)
                    real_preds = np.expm1(log_preds)

                    # Un-scale the actual target
                    log_actuals = target_scaler.inverse_transform(batch.y.cpu().numpy())
                    real_actuals = np.expm1(log_actuals)

                    val_errors_ms.extend(np.abs(real_preds - real_actuals).flatten())

            avg_mae_ms = np.mean(val_errors_ms)

            print(f"Epoch {epoch:03d} | Train Loss: {total_loss/len(train_loader):.4f} | Val MAE: {avg_mae_ms:.2f} ms")

    # 6. EVALUATE
    torch.save(model.state_dict(), "query_gnn_model.pth")
    # Pass the scaler to the evaluation function to get real ms back
    final_evaluation(model, test_loader, target_scaler)
    error_df = run_error_analysis(model, test_loader, target_scaler, raw_data)



