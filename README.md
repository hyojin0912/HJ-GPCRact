# GPCRact

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Dataset](https://img.shields.io/badge/Dataset-GPCRactDB-green.svg)](data/)

This repository serves as the official implementation and reproducibility package for the paper **"GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling"**.

We provide the complete source code, preprocessed datasets, training scripts, and analysis notebooks required to reproduce the findings presented in the manuscript.

<br>
<p align="center">
 <img width="700" height="800" alt="Figure2" src="https://github.com/user-attachments/assets/8a06699a-bb01-4d01-923b-58bef0beb99a" />
</p>

## 📋 Table of Contents
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducibility Workflow](#reproducibility-workflow)
  - [Step 1: Data Construction](#step-1-data-construction)
  - [Step 2: Model Training](#step-2-training)
  - [Step 3: Inference](#step-3-inference)
  - [Step 4: Benchmarking](#step-4-benchmarking)
  - [Step 5: Analysis & Figure Generation](#step-5-analysis)
- [Citation](#citation)
- [Contact](#contact)

---


## <a id="repository-structure"></a>📁 Repository Structure

We have unified all resources into a single structured repository to facilitate full reproducibility.

```plaintext
GPCRact/
├── analysis/           # Jupyter Notebooks for reproducing figures and statistical analyses
├── benchmarks/         # Implementation of baseline models (DeepREAL, AiGPro, 3D-GNN)
├── configs/            # Configuration files (YAML) for training and HPO
├── data/               # Datasets
│   ├── raw/            # Raw data files (GPCRactDB v1)
│   ├── resources/      # Auxiliary bio-info files (PDB info, MSA, etc.)
│   └── splits/         # Exact Train/Val/Test scaffold splits used in the paper
├── preprocessing/      # Scripts to reconstruct the dataset from scratch
├── scripts/            # Executable scripts for Training, Inference, and HPO
├── src/                # Core library code (Model architecture, Layers, Dataloaders)
├── environment.yml     # Conda environment file
└── README.md           # Master documentation
```

## <a id="installation"></a>⚙️ Installation 

We recommend using **Conda** to manage the environment for full reproducibility.

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

## <a id="reproducibility-workflow"></a>🔬 Reproducibility Workflow

This section explicitly delineates the steps to reproduce the results reported in our study.

### <a id="step-1-data-construction"></a>Step 1: Data Construction

Users can reconstruct the GPCRactDB from raw public data or use the pre-generated splits provided in `data/splits/`. To build from scratch, follow the pipeline in the `preprocessing/` directory:
```bash
# Example: Running the final dataset creation step
jupyter notebook preprocessing/04_create_final_dataset.ipynb
```
- Note: The exact scaffold-based split files (`scaffold_train.csv`, `scaffold_val.csv`, `scaffold_test.csv`) used in our study are already provided in `data/splits/` to ensure fair benchmarking.

### <a id="step-2-training"></a>Step 2: Training the Model 🏋️‍♂️

To train the GPCRact model from scratch using the provided splits:

1. **Configure**: Modify `configs/training_config.yaml` if necessary.
2. **Run**: Execute the training script.

```bash
python scripts/train.py \
    --data_dir data/splits \
    --save_dir checkpoints/ \
    --epochs 100
```
For detailed arguments, see `scripts/README.md`.


### <a id="step-3-inference"></a>Step 3: Inference 🚀

To predict the activity (Agonist/Antagonist/Non-binder) of novel GPCR-ligand pairs using a trained model:

```bash
python scripts/inference.py \
    --data_dir data/splits \
    --model_path checkpoints/best_model.pt \
    --output_dir results/
```

### <a id="step-4-benchmarking"></a>Step 4: Benchmarking 📊

We provide the full source code and execution scripts for the baseline models compared in the manuscript (**DeepREAL**, **AiGPro**, **3D-GNN**). All baselines were retrained on the identical GPCRact dataset.
* **DeepREAL**: See `benchmarks/DeepREAL/`

* **AiGPro**: See `benchmarks/AiGPro/` (Docker support included)

* **3D-GNN Baseline**: See `benchmarks/3D-GNN/`


### <a id="step-5-analysis"></a>Step 5: Analysis & Figure Generation 📉

To reproduce the statistical analyses, mechanistic interpretations, and main figures (Fig 1, 3, 4, 7), run the notebooks in the `analysis/` directory.

* `01_receptor_dynamics_analysis.ipynb`: Structural ground truth analysis (Fig 1).

* `02_sequence_structure_correlation.ipynb`: MSA vs. 3D dynamics (Fig 3).

* `03_activity_decision_tree.ipynb`: Decision tree for activity rules (Fig 4).

* `04_mechanistic_interpretability.ipynb`: Attention weight analysis (Fig 7).

_Supplementary Validations_: PRS analysis, Sensitivity analysis, and Mutation studies are also included.


## <a id="citation"></a>🎓 Citation
Our manuscript is currently under review. If you use GPCRact in your research, we would appreciate it if you could cite our work upon its publication. 


## <a id="contact"></a>📬 Contact
For questions, bug reports, or feedback, please contact Hyojin Son at hyojin0912@kaist.ac.kr. <img width="65" height="60" alt="image" src="https://github.com/user-attachments/assets/318cf6f3-2c2a-4fad-906f-2f28293a1b62" />

