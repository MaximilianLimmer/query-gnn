import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def run_full_analysis(model, loader, scaler, raw_data, output_prefix="gnn_report"):
    model.eval()
    results = []

    # Map raw_data to a lookup for SQL strings if needed
    # (Assuming raw_data order matches or you store an ID)

    with torch.no_grad():
        for batch in loader:
            out = model(batch).cpu().numpy()
            log_preds = scaler.inverse_transform(out)
            log_actuals = scaler.inverse_transform(batch.y.cpu().numpy())

            preds = np.expm1(log_preds).flatten()
            actuals = np.expm1(log_actuals).flatten()

            for p, a in zip(preds, actuals):
                q_err = max(p/a, a/p) if a > 0 and p > 0 else 1.0
                results.append({
                    'actual': a,
                    'predicted': p,
                    'q_error': q_err,
                    'abs_error': abs(p - a)
                })

    df = pd.DataFrame(results)

    # Global Stats
    print("\n" + "="*30)
    print("FINAL REPORT")
    print("="*30)
    print(f"Median Q-Error: {df['q_error'].median():.2f}")
    print(f"90th Percentile Q: {df['q_error'].quantile(0.90):.2f}")
    print(f"R² Score: {r2_score(df['actual'], df['predicted']):.4f}")

    _plot_visuals(df, output_prefix)
    return df

def _plot_visuals(df, prefix):
    plt.figure(figsize=(12, 5))

    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(df['actual'], df['predicted'], alpha=0.4)
    plt.plot([df['actual'].min(), df['actual'].max()],
             [df['actual'].min(), df['actual'].max()], 'r--')
    plt.title("Prediction Accuracy")
    plt.xlabel("Actual (ms)")
    plt.ylabel("Predicted (ms)")

    # Q-Error CDF
    plt.subplot(1, 2, 2)
    sorted_q = np.sort(df['q_error'])
    y = np.arange(len(sorted_q)) / float(len(sorted_q))
    plt.plot(sorted_q, y, color='green')
    plt.xlim(1, 5) # Focus on the "relevant" error range
    plt.axvline(2.0, color='orange', linestyle=':', label='Q=2 Threshold')
    plt.title("Cumulative Q-Error")
    plt.xlabel("Q-Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{prefix}_plots.png")