# Model Training & Inference Scripts

This directory contains the executable scripts to train the GPCRact model, perform inference on new data, and run hyperparameter optimization (HPO).

## Table of Contents
1. [Train the Model](#1-train-the-model)
2. [Run Inference](#2-run-inference)
3. [Hyperparameter Optimization](#3-hyperparameter-optimization)

---

## 1. Train the Model

Use `train.py` to train the GPCRact model from scratch using the preprocessed data splits.

### Usage
```bash
python scripts/train.py \
    --data_dir data/splits \
    --protein_graph_dir data/processed/protein_graphs \
    --ligand_graph_dir data/processed/ligand_graphs \
    --save_dir checkpoints/ \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.0001

#### Key Arguments
`--data_dir` Path to the directory containing `scaffold_train.csv` and `scaffold_val.csv`
`--protein_graph_dir` Path to the directory containing protein graph tensors `.pt`
`--ligand_graph_dir` Path to the directory containing ligand graph tensors `.pt`
`--save_dir` Directory to save model checkpoints.
`--enc_layers` Number of layers for the EGNN encoder.
`--prop_attn_layers` Number of layers for the Global Attention module.


