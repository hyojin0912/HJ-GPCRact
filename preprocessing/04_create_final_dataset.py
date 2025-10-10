import numpy as np
import pandas as pd
import random, itertools
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# --------------------------------------------------
# 0) Configuration
# --------------------------------------------------
TEST_FRAC     = 0.25      # target fraction for test set
FP_RADIUS     = 2
FP_NBITS      = 1024
CLUST_THRESH  = 0.35      # Tanimoto threshold for Butina
LABEL_TOL     = 0.1       # allowed deviation of class ratio
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------
# 1) Load data and create binary binding labels
# --------------------------------------------------
data = pd.read_csv('../Input/data.csv')
data = data[['Ikey', 'AC', 'Label', 'SMILES']]
data = data.dropna(subset=["SMILES"]).reset_index(drop=True)

labels_to_keep = ['agonist', 'antagonist', 'nonbinder']
data = data[data['Label'].isin(labels_to_keep)].copy()
data['Binding'] = np.where(data['Label'] == 'nonbinder', 0, 1)

# --- Configuration for file paths ---
PROTEIN_GRAPH_DIR = Path("../Data/Protein_Graphs_PyG/")
LIGAND_GRAPH_DIR = Path("../Data/Ligand_Graphs_PyG/")

# --- Get lists of available graph files ---
# .stem extracts the filename without the extension (e.g., "Q13547.pt" -> "Q13547")
available_proteins = {p.stem for p in PROTEIN_GRAPH_DIR.glob("*.pt")}
available_ligands = {p.stem for p in LIGAND_GRAPH_DIR.glob("*.pt")}

print(f"Found {len(available_proteins):,} unique protein graphs.")
print(f"Found {len(available_ligands):,} unique ligand graphs.")

# --- Filter the dataframe ---
original_size = len(data)
data = data[
    data['AC'].isin(available_proteins) & 
    data['Ikey'].isin(available_ligands)
].copy()

filtered_size = len(data)
print(f"\nOriginal data size: {original_size:,}")
print(f"Filtered data size (with available graphs): {filtered_size:,}")

print("Starting split process on filtered data...")
data.reset_index(drop=True, inplace=True)

# --------------------------------------------------
# NEW Step 1: Extract Unique Scaffolds
# --------------------------------------------------
print("Step 1: Extracting unique scaffolds from all molecules...")
scaffolds = {}
for i, smiles in enumerate(data['SMILES']):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smiles = Chem.MolToSmiles(scaf, isomericSmiles=False)
            if scaf_smiles: # Ensure scaffold is not empty
                scaffolds[scaf_smiles] = scaf
        except:
            continue # Handle rare cases where scaffold generation fails

unique_scaffold_smiles = list(scaffolds.keys())
unique_scaffolds = list(scaffolds.values())
print(f"Found {len(unique_scaffold_smiles):,} unique scaffolds from {len(data):,} molecules.")

# --------------------------------------------------
# NEW Step 2: Cluster the Scaffolds
# --------------------------------------------------
print("Step 2: Generating fingerprints and clustering unique scaffolds...")
scaffold_fps = [AllChem.GetMorganFingerprintAsBitVect(scaf, FP_RADIUS, nBits=FP_NBITS) for scaf in unique_scaffolds]

n_scaffolds = len(scaffold_fps)
dists = []
for i in tqdm(range(1, n_scaffolds), desc="Calculating scaffold similarities"):
    sims = DataStructs.BulkTanimotoSimilarity(scaffold_fps[i], scaffold_fps[:i])
    dists.extend([1 - x for x in sims])

scaffold_clusters = Butina.ClusterData(dists, nPts=n_scaffolds, distThresh=1 - CLUST_THRESH, isDistData=True)
print(f"Clustered scaffolds into {len(scaffold_clusters):,} groups.")

# --------------------------------------------------
# NEW Step 3: Splitting the Scaffold Clusters
# --------------------------------------------------
print("Step 3: Splitting scaffold clusters into train/test sets...")

# Convert tuple to list to allow shuffling
scaffold_clusters_list = list(scaffold_clusters)
random.shuffle(scaffold_clusters_list)

# Calculate total number of molecules per scaffold
# This step can be slow, let's optimize it by doing it once
print("Calculating molecule counts per scaffold...")
data['Scaffold'] = data['SMILES'].apply(lambda s: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)), isomericSmiles=False) if Chem.MolFromSmiles(s) else None)
scaffold_counts = data['Scaffold'].value_counts().to_dict()

target_test_size = int(len(data) * TEST_FRAC)
test_scaffold_indices = set()
current_test_size = 0

# Iterate over the SHUFFLED LIST, not the original tuple
for cluster in tqdm(scaffold_clusters_list, desc="Assigning scaffold clusters"):
    if current_test_size >= target_test_size:
        break
    
    # Calculate how many molecules this cluster represents
    cluster_size = sum(scaffold_counts.get(unique_scaffold_smiles[i], 0) for i in cluster)
    
    # Add cluster to test set if it doesn't overshoot the target too much
    if current_test_size + cluster_size < target_test_size * 1.5:
        test_scaffold_indices.update(cluster)
        current_test_size += cluster_size

test_scaffolds = {unique_scaffold_smiles[i] for i in test_scaffold_indices}
print(f"Selected {len(test_scaffolds)} scaffolds for the test set, representing {current_test_size:,} molecules.")

# --------------------------------------------------
# NEW Step 4: Assign All Molecules Based on Their Scaffold
# --------------------------------------------------
print("Step 4: Assigning all molecules to train/test splits...")
data['split'] = data['Scaffold'].apply(lambda s: 'test' if s in test_scaffolds else 'train')

# --- Save Output ---
cols = ["Ikey", "AC", "Binding", "SMILES"] # Adjust columns as needed
train_df = data[data.split == "train"]
test_df = data[data.split == "test"]

# Ensure columns exist before saving
final_cols = [col for col in cols if col in data.columns]
train_df[final_cols].to_csv("../Final/input/binding/train_set_scaf.csv", index=False)
test_df[final_cols].to_csv("../Final/input/binding/test_set_scaf.csv", index=False)

print("\n--- Final Output ---")
print(f"Train set size: {len(train_df):,} | Test set size: {len(test_df):,}")
print(f"Final test fraction: {len(test_df) / len(data):.3f}")


# + Add Activity Column
train_df = pd.read_csv("../Final/input/binding/train_set_scaf.csv")
test_df = pd.read_csv("../Final/input/binding/test_set_scaf.csv")

data = pd.read_csv("../Input/data.csv")
labels_to_keep = ['agonist', 'antagonist', 'nonbinder']
data = data[data['Label'].isin(labels_to_keep)].copy()

# Label → Activity
label_to_activity = {
    'agonist': 1,
    'antagonist': 0,
    'nonbinder': np.nan
}
data['Activity'] = data['Label'].map(label_to_activity)
mapping_dict = dict(zip(zip(data['Ikey'], data['AC']), data['Activity']))
train_df['Activity'] = train_df.apply(lambda row: mapping_dict.get((row['Ikey'], row['AC']), np.nan), axis=1)
test_df['Activity'] = test_df.apply(lambda row: mapping_dict.get((row['Ikey'], row['AC']), np.nan), axis=1)

train_df.to_csv("../Final/input/activity/train_set_scaf.csv", index = False)
test_df.to_csv("../Final/input/activity/test_set_scaf.csv", index = False)
