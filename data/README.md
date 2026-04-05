# GPCRact Data Directory

This directory contains the datasets, bio-informatic resources, and processed graph representations required to train, evaluate, and benchmark the **GPCRact** model.

To ensure strict reproducibility and transparency, we provide everything from the raw text-mined ground truth (GPCRactDB) to the exact scaffold-based splits used in our *Briefings in Bioinformatics* paper.

## 📂 Directory Structure

```text
data/
├── raw/                 # Raw experimental data and text-mined interactions
├── resources/           # Auxiliary biological data (PDBs, MSAs, structural definitions)
├── splits/              # Exact dataset splits for reproducible benchmarking
├── protein_graphs/      # Processed protein PyTorch Geometric (.pt) files
├── ligand_graphs/       # Processed ligand PyTorch Geometric (.pt) files
└── sample/              # Minimal dataset for quickstart and inference testing
```

## 📄 Detailed Description

### 1. `raw/` (Ground Truth & Activity Data)
Contains the foundational interaction data before any graph processing.
* **`GPCRactDB_v1.csv`**: The core dataset of our study. Contains over 200,000 curated GPCR-ligand interactions with definitive mode-of-action labels (Agonist/Antagonist/Non-binder), assembled via text mining from bioassay literature.

### 2. `resources/` (Biological & Structural Context)
Contains the domain-specific biological metadata required to build the mechanistically aware graphs (identifying binding sites and allosteric pathways).
* **`Binding_Sites_Heavy_Atom_based.csv`**: Explicit atom-level definitions of ligand binding pockets for each GPCR.
* **`Differential_Residues_Heavy_Atom_based.csv`**: Identification of critical residues driving conformational changes between inactive and active states.
* **`MSA_DF.csv`**: Multiple Sequence Alignment (MSA) features capturing the evolutionary conservation context of the receptors.
* **`Human_GPCR_PDB_Info.csv` & `GPCR_PDB_classification.csv`**: Comprehensive metadata and classification for all utilized human GPCR crystal/cryo-EM structures.
* **`Representative_Apo_Structures_v2.csv` & `Rep_GPCR_chain.csv`**: Mappings for the representative unbound (Apo) base structures used when modeling ligand-induced state transitions.
* **`ChEMBL_GPCR_Info.csv`**: Standardized compound information cross-referenced from the ChEMBL database.
* **`tb_aid_act_gpcr.csv`**: Supplementary bioassay mapping data used during the curation pipeline.

### 3. `splits/` (Benchmarking Splits)
To rigorously evaluate model generalization to novel chemical spaces, we use Murcko scaffold-based splitting. Using these exact files ensures fair comparison against our baseline.
* **`scaffold_train.csv`**: Training set.
* **`scaffold_val.csv`**: Validation set for hyperparameter tuning and early stopping.
* **`scaffold_test.csv`**: Hold-out test set containing functionally active/inactive compounds with unseen chemical scaffolds.

### 4. Processed Graph Inputs (`protein_graphs/` & `ligand_graphs/`)
Contains the final 3D atomistic graph objects ready to be ingested by the PyTorch Geometric data loaders.
* **`*.pt` files**: Due to large file sizes (>150MB), the full set of 200,000+ ligand graphs and 300+ protein graphs are hosted on Hugging Face. **Do not generate them from scratch.** Please use the provided download script to fetch them:
  ```bash
  bash scripts/download_full_data.sh
  ```
* **`*_to_id.json`**: Dictionary mapping files for categorical encoding (e.g., family classes, amino acid types). *Note: Do not overwrite these files during inference to maintain index consistency with the pretrained weights.*

### 5. `sample/` (Quickstart)
A minimal subset of `.pt` files and a toy `.csv` manifest designed strictly to verify that the environment and `inference.py` scripts are working correctly.
