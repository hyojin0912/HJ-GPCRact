# GPCRact: A Hierarchical Framework for Predicting Ligand-Induced GPCR Activity via Allosteric Communication Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository provides the official implementation and data for the paper: **"GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling"** .

[Link to Paper] ---

## Abstract

Accurate prediction of ligand-induced activity for G-protein-coupled receptors (GPCRs) is a cornerstone of drug discovery, yet it is challenged by the need to model allosteric communication—the long-range signaling linking ligand binding to distal conformational changes. To address this, we introduce GPCRact, a novel framework that models the biophysical principles of allosteric modulation in GPCR activation. It first constructs a high-resolution, three-dimensional structure-aware graph from the heavy-atom coordinates of functionally critical residues at binding and allosteric sites. A dual attention architecture then captures the activation process... (*이하 생략*)

<br>

<p align="center">
  <img src="httpse-Based Graph Modeling**
img width="2055" height="2209" alt="Figure2" src="https://github.com/user-attachments/assets/72ca9f60-2823-48f1-839b-75bf3bc4f79a
" />
  <br>
  <em>Figure: The hierarchical architecture of the GPCRact framework, which models ligand binding and subsequent allosteric signal propagation in two distinct stages.</em>
</p>

---

## ⚙️ Installation

We recommend using Conda to manage the environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/GPCRact.git](https://github.com/your-username/GPCRact.git)
    cd GPCRact
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate gpcr_act
    ```
    Alternatively, you can install packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage

### 1. Quick Start: Inference with a Pre-trained Model

To predict the activity for a new ligand-GPCR pair using our pre-trained model:

```bash
python scripts/predict.py \
    --model_checkpoint 'models/GPCRact_pretrained.pth' \
    --pdb_file 'path/to/your/gpcr_structure.pdb' \
    --ligand_smiles 'SMILES_STRING_OF_YOUR_LIGAND' \
    --output_dir 'results/'
