# Scalable GNN–Based Preconditioners for Conjugate Gradient Methods

This project was completed as part of the course CS4350: Machine Learning for Graph Data at TU Delft.
Authors: Low Jun Yu, Nicholas Tan Yun Yu, Wu Yuhan

## Brief Introduction to Project
In this project, we focused on the scalability limitations of existing GNN-based
preconditioners for conjugate gradient methods, whose full-graph message passing
architecture incur a per-layer computational cost of O(nnz(A)) cost per layer,
and become limited for larger or denser systems on memory-limited devices. To
address this, we explored several scalable GNN strategies, including GraphSAGE,
GraphSAINT and Cluster-GCN, aiming to reduce per-layer complexity with limited
degradation in the preconditioning quality.

## Project Information
**Cluster-GCN**: Implementation can be found at [Cluster-GCN branch](https://github.com/HNU-WYH/GML-Project/tree/cluster_gnn).
**GraphSAGE and GraphSAINT**: Implementation can be found at [GraphSAGE and GraphSAINT branch](https://github.com/HNU-WYH/GML-Project/tree/scalable_graph).
**Paper**: View the [report](CS4350_Group7_report.pdf) here.
