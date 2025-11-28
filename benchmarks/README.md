# Benchmark Models Implementation

This directory contains the source code and execution scripts for the benchmark models used in the comparative analysis of the GPCRact framework. To ensure a rigorous and fair comparison, we implemented three baseline models: **DeepREAL**, **AiGPro**, and a custom **3D-GNN Baseline**.

All models were retrained on the identical GPCRact dataset using the same scaffold-based train/validation/test splits.

## 1. DeepREAL
**Reference:** [DeepREAL: a deep learning powered multi-scale modeling framework for predicting out-of-distribution ligand-induced GPCR activity (2022)](https://academic.oup.com/bioinformatics/article/38/9/2561/6547052)

DeepREAL is a multi-stage framework designed to classify ligands into agonist, antagonist, and non-binder categories. We utilized the official source code and retrained the model following the original three-stage pipeline:
1.  Self-supervised pretraining of protein sequences.
2.  Binary interaction classification (Protein + Ligand GIN).
3.  Multi-class activity prediction.

### Directory Structure
```text
benchmarks/DeepREAL/
├── data_tool_box.py    # Data utilities
├── model_Yang.py       # Core model architecture
├── resnet.py           # ResNet encoder for proteins
├── fp_models.py        # Fingerprint models
├── train.py            # Main training script (adapted)
├── run_benchmark.sh    # Execution script
└── ...
```

### How to Run
We provide a shell script to automate the environment setup and training process.
```bash
cd benchmarks/DeepREAL
bash run_benchmark.sh
```

## 2. AiGPro
**Reference:** [AiGPro: a multi-tasks model for profiling of GPCRs for agonist and antagonist (2025)](https://link.springer.com/article/10.1186/s13321-024-00945-7)

AiGPro is a multi-task framework originally designed for regression (predicting bioactivity values). To align with our objective, we adapted its architecture for three-class classification.
- Modification: The final regression output layer was replaced with a classification head (3 output nodes + Softmax).
- Loss Function: Changed from MSE to Categorical Cross-Entropy.
- Environment: Implemented using Docker to ensure perfect reproducibility of its complex dependency environment.

### Directory Structure
```text
benchmarks/AiGPro/
├── aigpro/             # Source code package
├── data/               # Data mounting point
├── docker-compose.yml  # Container configuration
├── Dockerfile          # Environment definition
└── run_benchmark.sh    # Execution script (End-to-end)
```

### How to Run
Ensure Docker and Docker Compose are installed on your system.
```bash
cd benchmarks/AiGPro
# This script handles container building, data preprocessing, and training
bash run_benchmark.sh
```

## 3. 3D-GNN Baseline
This baseline was designed to quantify the contribution of GPCRact's specific architectural modules (Hierarchical decoupling, Cross-Attention, Gated transfer) versus the benefit of simply using 3D structural data.

- Architecture: A single-stage, unified EGNN encoder with a single classification head.
- Input: Identical 3D protein and ligand graphs as GPCRact.
- Mechanism: Ligand features are pooled and injected into the protein graph, which is then processed by a standard EGNN.

### Directory Structure
```text
benchmarks/3D-GNN/
└── train.py            # Standalone training script
```

### How to Run
```bash
# Run directly with Python (uses the main environment)
python benchmarks/3D-GNN/train.py \
    --data_dir ./data/processed/ \
    --protein_graph_dir ./data/graphs/protein/ \
    --ligand_graph_dir ./data/graphs/ligand/ \
    --output_dir ./results/3D-GNN/
```

## Prerequisites & Data
Before running any benchmarks, please ensure that:
1. The main environment.yml is installed (for DeepREAL and 3D-GNN).
2. Docker is installed (for AiGPro).
3. The Preprocessing Pipeline has been completed, and the processed datasets (`train.csv`, `test.csv`, etc.) are located in the `data/` directory.

