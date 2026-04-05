# GPCRact
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Dataset](https://img.shields.io/badge/Dataset-GPCRactDB-green.svg)](data/)
[![Journal](https://img.shields.io/badge/Journal-Briefings%20in%20Bioinformatics-blue)](https://academic.oup.com/bib/article/27/1/bbaf719/8426121)

This repository serves as the official implementation and reproducibility package for the paper  
**[GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling](https://academic.oup.com/bib/article/27/1/bbaf719/8426121)**  (*Briefings in Bioinformatics*, 2026).

We provide the complete source code, preprocessed datasets, training scripts, and analysis notebooks required to reproduce the findings presented in the manuscript.


## 🚀 Key Features
- **Mechanism-Driven Architecture:** Combines E(n)-equivariant GNNs with dual attention to model "binding → allosteric propagation → activity"
- **Functionally Critical Subgraphs:** Efficient 3D atomistic graph construction focusing on binding and allosteric sites.
- **Reproducible Pipeline:** Fully automated workflow from raw PDB/Bioassay data to final evaluation.
- **Bias-Aware Benchmarking:** Includes rigorous scaffold-based splits and re-implementations of SOTA baselines.


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
├── checkpoints/        # Pretrained model weights (best_model.pt) included for quickstart
├── configs/            # Configuration files (YAML) for training and HPO
├── data/               # Datasets
│   ├── raw/            # Raw data files (GPCRactDB v1)
│   ├── resources/      # Auxiliary bio-info files (PDB info, MSA, etc.)
│   ├── protein_graphs/ # Preprocessed protein graphs (.pt) & Dictionary files (.json)
│   ├── ligand_graphs/  # Processed ligand PyG graphs
│   ├── splits/         # Exact Train/Val/Test scaffold splits used in the paper
│   └── sample/         # Minimal toy dataset for quickstart and environment testing
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
    git clone https://github.com/hyojin0912/HJ-GPCRact.git
    cd HJ-GPCRact
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate gpcract
    ```
    Alternatively, you can install packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
## <a id="quickstart"></a>⚡ Quickstart (10-Second Inference)
We provide a minimal toy dataset (`data/sample/`) so you can verify the environment and test the model immediately without downloading the massive full dataset.
```bash
# Run inference on the provided sample data (Agonist, Antagonist, Non-binder)
python scripts/inference.py
```

## <a id="downloading-the-full-dataset"></a>📥 Downloading the Full Dataset
Due to the large size of the 3D atomistic graphs (>150MB for 200,000+ interactions), the complete graph dataset is hosted remotely on Hugging Face. We provide a shell script to automate the download and extraction process directly into your pipeline.
```bash
# Fetch and extract ligand and protein graphs to the data/ directory
bash scripts/download_full_data.sh
```
For detailed information about the dataset structure and curation, see `data/README.md`.

## 🤖 Pretrained Models
We provide the official pretrained weights used to generate the results in the paper. 
The default model weight (`best_model.pt`) is **already included** in the `checkpoints/` directory, allowing you to run inference immediately without manual downloads. 

*(You can also find the checkpoint file on the **[Releases Page](https://github.com/hyojin0912/HJ-GPCRact/releases/tag/v1.0.0)** if needed.)*

### How to Load Weights
```python
import torch
from src.model import GPCRact_Model

# Initialize model (ensure config matches training)
model = GPCRact_Model(...) 

# Load weights
checkpoint_path = "path/to/model.pt"
state_dict = torch.load(checkpoint_path, map_location='cuda')
model.load_state_dict(state_dict, strict=False)
model.eval()
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
- Note: Generating 3D conformers for 200,000+ ligands is computationally expensive. We highly recommend using the `download_full_data.sh` script instead of running the preprocessing pipeline from scratch.

### <a id="step-2-training"></a>Step 2: Training the Model 🏋️‍♂️

To train the GPCRact model from scratch using the provided splits:

1. **Configure**: Modify `configs/training_config.yaml` if necessary.
2. **Run**: Execute the training script.

```bash
python scripts/train.py \
    --data_dir data/splits \
    --save_dir checkpoints/ \
    --epochs 200
```
For detailed arguments, see `scripts/README.md`.


### <a id="step-3-inference"></a>Step 3: Full Inference 🚀

**⚠️ Important Note:** When running inference, **do not** execute preprocessing scripts that generate new dictionary files (e.g., `class_to_id.json`, `family_to_id.json`). Doing so will overwrite the dictionaries based on your test data and cause "size mismatch" errors with the pretrained weights. Please ensure you are using the original `.json` dictionary files provided in `data/protein_graphs/`.

To predict the activity (Agonist/Antagonist/Non-binder) of novel GPCR-ligand pairs using a trained model:

```bash
python scripts/inference.py \
    --data_dir data/splits/ \
    --query_csv scaffold_test.csv \
    --protein_graph_dir data/protein_graphs/ \
    --ligand_graph_dir data/ligand_graphs/ \
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
If you use GPCRact in your research, please cite the following paper:
```bibtex
@article{son2026gpcract,
  title={GPCRact: a hierarchical framework for predicting ligand-induced GPCR activity via allosteric communication modeling},
  author={Son, Hyojin and Yi, Gwan-Su},
  journal={Briefings in Bioinformatics},
  volume={27},
  number={1},
  pages={bbaf719},
  year={2026},
  doi={10.1093/bib/bbaf719}
}
```

## <a id="contact"></a>📬 Contact
For questions, bug reports, or feedback, please contact Hyojin Son at hyojin0912@kaist.ac.kr. 


