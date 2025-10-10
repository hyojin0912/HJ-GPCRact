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

### Step 4: Create Final Dataset Splits

This final script merges all information and creates the final training, validation, and test sets using a scaffold split.

```bash
python preprocessing/04_create_final_dataset.py
```

