# GPCRact

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository provides the official implementation and data for the paper: **"GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling"** .


<br>
<p align="center">
 <img width="700" height="800" alt="Figure2" src="https://github.com/user-attachments/assets/8a06699a-bb01-4d01-923b-58bef0beb99a" />
</p>
---

## Installation ⚙️ 

We recommend using Conda to manage the environment for full reproducibility.

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

## Protocol Overview 🔬

This repository provides a complete protocol, from data construction to model training.

1.  **Data Construction:** To reconstruct the GPCRactDB from scratch, follow the detailed steps in [`preprocessing/README.md`](preprocessing/README.md).
2.  **Model Training & Inference:** To train the model using our pre-processed data or make predictions with a pre-trained model, see the `Usage` section below.


## Usage ▶️

This section provides instructions on how to use the pre-trained model for inference and how to train a new model from scratch.

### 1. Inference with a Pre-trained Model (Quick Start) 🚀

We provide a script (`scripts/predict.py`) to easily predict the activity of a novel GPCR-ligand pair. This script will generate the necessary graphs on-the-fly from a PDB file and a SMILES string.

**Command:**
```bash
python scripts/predict.py \
    --pdb "path/to/your_receptor.pdb" \
    --chain "A" \
    --smiles "OCCOCC" \
    --model_checkpoint "models/GPCRact_pretrained.pt"
```
--pdb: Path to the receptor's PDB structure file.
--chain: Chain ID of the receptor in the PDB file (default: 'A').
--smiles: The SMILES string of the ligand.
--model_checkpoint: Path to the pre-trained model file (included in this repository).

### 2. Training a New Model 🏋️‍♂️

To train or fine-tune the GPCRact model on your own data, follow these steps:

1. Get the Data
Reconstruct the entire dataset from raw files by following the guide in [`preprocessing/README.md`](preprocessing/README.md).

2. Configure Training:
Modify the parameters in `configs/training_config.yaml` to fit your experiment (e.g., learning rate, batch size, data paths).

3. Run Training:
Execute the training script from the project root directory:
```bash
python scripts/train.py
```
The script will use the configuration specified in the YAML file. Progress will be logged, and the best model will be saved in the directory defined in the config.


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
│   ├── 01_parse_pubchem_bioassay.py 
│   ├── 01_parse_other_databases.py
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
Our manuscript is currently under review. If you use GPCRact in your research, we would appreciate it if you could cite our work upon its publication. 


## 📬 Contact
For questions, bug reports, or feedback, please contact Hyojin Son at hyojin0912@kaist.ac.kr.
