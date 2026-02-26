# Hybrid GNN Query Cost Estimator

### The Goal
Traditional database optimizers use static cost models that often fail to capture the non-linear complexities of modern analytical queries. The goal of this project is to build a **Learned Cost Estimator** that treats SQL execution plans as recursive data structures, providing highly accurate, real-world runtime predictions in milliseconds.

### The Model: GATv2 + Feature Masking
To achieve this, I developed a **Hybrid Graph Attention Network (GNN)** using the GATv2 architecture. 
* **Graph Representation:** SQL plans are transformed into directed graphs where nodes represent operators (e.g., `Hash Join`, `Index Scan`) and edges represent data flow.
* **Leakage Prevention:** To ensure the model actually "understands" query structure, the optimizer's internal cost estimates are masked during the graph convolution phase and only re-introduced in the final MLP head.

### 📈 Results at a Glance
After training on 20,000 PostgreSQL plans, the model achieved state-of-the-art accuracy for a local schema:
* **R² Score:** 0.9658
* **Median Q-Error:** 1.13
* **90th Percentile Q:** 1.61

<p align="center">
  <img src="gnn_report_plots.png" width="1000" title="Predicted vs Actual Runtime">
</p>

### Data Generation & Pipeline
* **Query Generation:** A recursive generator produced **20,000 unique queries**, from simple filters to 5-way joins.
* **Ground Truth:** Each query was executed via `EXPLAIN ANALYZE`. To minimize system noise, the **median of 3 runs** was used as the target label.
* **Preprocessing:** Plans were transformed into directed graphs with 27 features per node (Operators, Row Counts, Widths, and Costs).

### Model Health Report

<p align="center">
  <img src="dataset_health_visuals.png" width="1000" title="Model Health Report">
</p>
