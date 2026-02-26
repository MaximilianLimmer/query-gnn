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
  <img src="gnn_report_plot.png" width="600" title="Predicted vs Actual Runtime">
  <br>
</p>---
