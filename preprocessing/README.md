# GPCRactDB Preprocessing Pipeline

This directory contains the scripts to build the GPCRactDB from raw public data.
The pipeline is divided into four sequential steps. Please run the scripts in numerical order.

## Prerequisites

- Ensure all dependencies from the main `environment.yml` are installed.

## Step-by-Step Protocol

### Step 1: Parse and Integrate Raw Data

This script collects data from various public sources and integrates them into a unified format.

```bash
python preprocessing/01_parse_pubchem_bioassay.py
python preprocessing/01_parse_other_databases.py
```

### Step 2: Generate Protein Graphs

This script processes the protein PDBs from the integrated data to generate 3D structure-aware graphs.

```bash
python preprocessing/02_generate_protein_graphs.py
```

### Step 3: Generate Ligand Graphs

This script processes ligand SMILES strings to generate molecular graphs.

```bash
python preprocessing/03_generate_ligand_graphs.py
```
⚠️ Important Note: Generating 3D conformers (MMFF/UFF optimization) for over 100,000 ligands is highly computationally expensive. For training and reproduction purposes, we strongly recommend downloading the pre-computed graph dataset using our bash script (bash scripts/download_full_data.sh) rather than running this script from scratch.

### Step 4: Create Final Dataset & Scaffold Splits (Key Step)
This notebook merges all information and performs the Scaffold-based clustering and splitting. It generates the final train.csv, val.csv, and test.csv files used for model training.

* Note: This is provided as a Jupyter Notebook (.ipynb) to visualize the scaffold clustering distribution and validate the train-test separation (sanity check).

Open the notebook in Jupyter and Run all cells to generate the splits in `data/splits/.`

```bash
jupyter notebook preprocessing/04_create_final_dataset.ipynb
```

