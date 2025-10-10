# GPCRact

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository provides the official implementation and data for the paper: **"GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling"** .


<br>
<img width="1500" height="1600" alt="Figure2" src="https://github.com/user-attachments/assets/8a06699a-bb01-4d01-923b-58bef0beb99a" />

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

## 🔬 Protocol Overview

This repository provides a complete protocol, from data construction to model training.

1.  **Data Construction (Optional):** To reconstruct the GPCRactDB from scratch, follow the detailed steps in [`preprocessing/README.md`](preprocessing/README.md).
2.  **Model Training & Inference:** To train the model using our pre-processed data or make predictions with a pre-trained model, see the `Usage` section below.

---
## 📁 Repository Structure
```plaintext
GPCRact/
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── requirements.txt
│
├── configs/                # Configuration files for experiments
│   └── training_config.yaml
│
├── data/
│   ├── raw/                  # Raw data collected from public databases
│   └── processed/            # Processed data for model training
│
├── models/                 # Pre-trained model checkpoints
│   └── GPCRact_pretrained.pt
│
├── preprocessing/          # Scripts to build the dataset from scratch
│   ├── README.md             # Guide for the preprocessing pipeline
│   ├── 01_parse_raw_data.py
│   ├── 02_generate_protein_graphs.py
│   ├── 03_generate_ligand_graphs.py
│   └── 04_create_final_dataset.py
│
├── scripts/                # Executable scripts for training and inference
│   └── train.py
│
└── src/                    # Source code for the GPCRact library
    ├── data_loader.py
    ├── model.py
    ├── modules.py
    └── utils.py
```

## 🎓 Citation
Our manuscript is currently under review. If you use GPCRact in your research, we would appreciate it if you could cite our work upon its publication. A preliminary BibTeX entry is provided below.
@article{Son2025GPCRact,
  title={{GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling}},
  author={Son, Hyojin and Yi, Gwan-Su},
  journal={Briefings in Bioinformatics},
  year={2025},
  note={Under Review}
}


---
## 📬 Contact
For questions, bug reports, or feedback, please contact Hyojin Son at hyojin0912@kaist.ac.kr.
