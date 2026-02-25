# data/dataset_report.py
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy

def get_node_types(node, types):
    types.append(node.get("Node Type", "Unknown"))
    if "Plans" in node:
        for sub in node["Plans"]:
            get_node_types(sub, types)
    return types

def run_master_report(file_path="query_data.json"):
    print(f"Generating Report for: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    rows, all_ops = [], []
    for item in data:
        plan = item['plan']
        rows.append({
            'runtime': item['runtime'],
            'total_cost': plan.get('Total Cost', 0),
            'plan_rows': plan.get('Plan Rows', 0),
            'plan_width': plan.get('Plan Width', 0),
            'num_nodes': str(plan).count("'Node Type'"),
            'shared_hit': plan.get('Shared Hit Blocks', 0),
            'shared_read': plan.get('Shared Read Blocks', 0)
        })
        get_node_types(plan, all_ops)

    df = pd.DataFrame(rows)

    # --- Metrics Logic ---
    corrs = df.corr()['runtime'].sort_values(ascending=False)
    counts = pd.Series(all_ops).value_counts()
    ent = entropy(counts)

    # Signal Analysis (MI)
    X = df.drop('runtime', axis=1)
    y = df['runtime']
    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    print(f"\n[Dataset Entropy]: {ent:.2f}")
    print("\n[Correlation with Runtime]:")
    print(corrs)

    # --- Visualization ---
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")

    plt.subplot(2, 2, 2)
    mi_series.plot(kind='barh', color='orchid')
    plt.title("Mutual Information (Non-linear Signal)")

    plt.subplot(2, 2, 3)
    sns.regplot(data=df, x='total_cost', y='runtime',
                scatter_kws={'alpha':0.15, 's':8}, line_kws={'color':'red'})
    plt.xscale('log')
    plt.title("Cost vs. Runtime (Log-Linear)")

    plt.subplot(2, 2, 4)
    counts.head(10).plot(kind='pie', autopct='%1.1f%%', cmap='viridis')
    plt.title("Top 10 Operators")

    plt.tight_layout()
    plt.savefig("dataset_health_visuals.png")
    print("Saved dataset report to dataset_health_visuals.png")

    return df