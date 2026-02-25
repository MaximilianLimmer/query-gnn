import json
import random
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from model.evaluate import run_full_analysis
from model.model import HybridQueryGNN
from model.utils import get_prepared_data

def run_evaluation(model, loader, scaler):
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch).cpu().numpy()
            log_preds = scaler.inverse_transform(out)
            log_actuals = scaler.inverse_transform(batch.y.cpu().numpy())
            all_preds.extend(np.expm1(log_preds).flatten())
            all_actuals.extend(np.expm1(log_actuals).flatten())
    return np.array(all_preds), np.array(all_actuals)



def run_training_pipeline(epochs=300, data_path="query_data.json"):
    # 1. Load & Prepare
    with open(data_path, "r") as f:
        raw_data = json.load(f)

    dataset, target_scaler = get_prepared_data(raw_data)
    random.shuffle(dataset)

    split = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:split], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset[split:], batch_size=32)

    # 2. Setup
    model = HybridQueryGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 3. Train
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            preds, actuals = run_evaluation(model, test_loader, target_scaler)
            mae = np.mean(np.abs(preds - actuals))
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Val MAE: {mae:.2f} ms")
            scheduler.step(mae)

    # 4. Final Save & Stats
    torch.save(model.state_dict(), "query_gnn.pth")
    analysis_df = run_full_analysis(
        model=model,
        loader=test_loader,
        scaler=target_scaler,
        raw_data=raw_data
    )
    analysis_df.to_csv("test_results.csv", index=False)
    print("Training and Evaluation complete.")