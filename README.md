# GPCRact: A Hierarchical Framework for Predicting Ligand-Induced GPCR Activity via Allosteric Communication Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository provides the official implementation and data for the paper: **"GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling"** .

## Abstract

Accurate prediction of ligand-induced activity for G-protein-coupled receptors (GPCRs) is a cornerstone of drug discovery, yet it is challenged by the need to model allosteric communication—the long-range signaling linking ligand binding to distal conformational changes. To address this, we introduce GPCRact, a novel framework that models the biophysical principles of allosteric modulation in GPCR activation. It first constructs a high-resolution, three-dimensional structure-aware graph from the heavy-atom coordinates of functionally critical residues at binding and allosteric sites. A dual attention architecture then captures the activation process... (*이하 생략*)

<br>
<img width="2055" height="2209" alt="Figure2" src="https://github.com/user-attachments/assets/8a06699a-bb01-4d01-923b-58bef0beb99a" />

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
## 📁 Repository Structure
GPCRact/
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml         # Conda environment file
├── requirements.txt        # Pip requirements file
│
├── configs/                # Configuration files for training/evaluation
│   └── training_config.yaml
│
├── data/
│   ├── README.md             # Description of data format and sources
│   ├── raw/                  # Raw data from ChEMBL, GPCRdb
│   └── processed/            # Processed data for training and analysis
│
├── models/                 # Pre-trained model checkpoints
│   └── GPCRact_pretrained.pth
│
└── src/                    # Source code
    ├── data_loader.py
    ├── model.py
    ├── modules.py
    └── utils.py

