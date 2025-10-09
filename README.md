# GPCRact

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository provides the official implementation and data for the paper: **"GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling"** .


<br>
<img width="2055" height="2209" alt="Figure2" src="https://github.com/user-attachments/assets/8a06699a-bb01-4d01-923b-58bef0beb99a" />

---

## ⚙️ Installation

We recommend using Conda to manage the environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hyojin0912/HJ-GPCRact.git](https://github.com/hyojin0912/HJ-GPCRact.git)
    cd HJ-GPCRact
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
```markdown
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
│   ├── raw/                  # Raw data from ChEMBL, GPCRdb ...
│   └── processed/            # Processed graph data for training
│
├── models/                 # Pre-trained model checkpoints
│   └── GPCRact_pretrained.pt
│
└── src/                    # Source code for the GPCRact model and utilities
    ├── data_loader.py
    ├── model.py
    ├── modules.py
    ├── train.py
    └── utils.py
```


---
## 📬 Contact
For questions, bug reports, or feedback, please contact Hyojin Son at hyojin0912@kaist.ac.kr.
