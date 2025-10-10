# Import necessary libraries
import os
import pandas as pd
import torch
from tqdm.notebook import tqdm
import multiprocessing as mp
from math import ceil
import time

# RDKit for cheminformatics
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter, rdchem

# PyTorch Geometric for graph data
from torch_geometric.data import Data

# --- Configuration ---
# Input file with SMILES data
LIGAND_DATABASE_FILE = "../Input/data.csv"

# Output directories
SDF_OUTPUT_DIR = "../Data/Ligand_SDF/"
GRAPH_OUTPUT_DIR = "../Data/Ligand_Graphs_PyG/"

os.makedirs(SDF_OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# Number of parallel processes to use
NUM_PROCESSES = max(1, mp.cpu_count() - 2)

# Load the main database
df = pd.read_csv(LIGAND_DATABASE_FILE)

# Create a unique list of ligands (Ikey, SMILES)
unique_ligands_df = df[['Ikey', 'SMILES']].drop_duplicates().reset_index(drop=True)
print(f"Found {len(unique_ligands_df)} unique ligands in the database.")

# Check for existing SDF files
existing_sdf_ikeys = {f.split('.')[0] for f in os.listdir(SDF_OUTPUT_DIR) if f.endswith('.sdf')}
sdf_to_process_df = unique_ligands_df[~unique_ligands_df['Ikey'].isin(existing_sdf_ikeys)]
print(f"{len(sdf_to_process_df)} ligands need 3D conformer generation.")

# Check for existing graph files
existing_graph_ikeys = {f.split('.')[0] for f in os.listdir(GRAPH_OUTPUT_DIR) if f.endswith('.pt')}
graphs_to_process_df = unique_ligands_df[~unique_ligands_df['Ikey'].isin(existing_graph_ikeys)]
print(f"{len(graphs_to_process_df)} ligands need graph conversion.")


# 3D Conformer Generation (SMILES → SDF)
def generate_3d_sdf_from_smiles(smiles: str, output_path: str, random_seed: int = 42) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv2()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol, params) == -1: return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # Fallback to UFF if MMFF fails
        AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    writer = SDWriter(output_path)
    writer.write(mol)
    writer.close()
    return True

def sdf_worker(row):
    """Worker function for parallel processing."""
    ikey, smiles = row["Ikey"], row["SMILES"]
    output_path = os.path.join(SDF_OUTPUT_DIR, f"{ikey}.sdf")
    try:
        success = generate_3d_sdf_from_smiles(smiles, output_path)
        return ikey if success else None
    except Exception:
        return None

# --- Execute SDF Generation ---
if not sdf_to_process_df.empty:
    print("\n--- Starting 3D Conformer Generation ---")
    
    # Using python's multiprocessing pool for parallel execution
    pool = mp.Pool(processes=NUM_PROCESSES)
    results = list(tqdm(pool.imap(sdf_worker, [row for _, row in sdf_to_process_df.iterrows()]), 
                        total=len(sdf_to_process_df), desc="Generating SDFs"))
    pool.close()
    pool.join()
    
    successful_count = sum(1 for r in results if r is not None)
    print(f"\nSuccessfully generated {successful_count} new SDF files.")
else:
    print("\n--- All SDF files already exist. Skipping generation. ---")


# 3D Graph Featurization (SDF → PyG Data)
def get_atom_features(atom):
    """Encodes atom features into a feature vector."""
    # Symbol (One-hot)
    possible_elements = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    element = atom.GetSymbol()
    # Use one-hot encoding for element, with a fallback for others
    features = [int(element == s) for s in possible_elements] + [int(element not in possible_elements)]
    # Degree
    features.append(atom.GetDegree())
    # Formal Charge
    features.append(atom.GetFormalCharge())
    # Hybridization (One-hot)
    hybridizations = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3]
    features += [int(atom.GetHybridization() == h) for h in hybridizations]
    # Aromaticity
    features.append(int(atom.GetIsAromatic()))
    # Number of Hydrogens
    features.append(atom.GetTotalNumHs())
    # Ring membership
    features.append(int(atom.IsInRing()))
    return features

def get_bond_features(bond):
    """Encodes bond features into a feature vector."""
    bond_types = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
    features = [int(bond.GetBondType() == t) for t in bond_types]
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    return features

def sdf_to_graph(sdf_path, ikey):
    """Converts a single SDF file to a PyTorch Geometric Data object."""
    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
        return None
    try:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = supplier[0]
        if mol is None: return None
    except:
        return None

    # Get Atom Features, 3D Coordinates, and Identifiers
    atom_features, positions, node_identifiers = [], [], []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        node_identifiers.append((atom.GetSymbol(), atom.GetIdx()))
    
    # Get Bond Features and Connectivity
    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([(i, j), (j, i)])
        bond_feat = get_bond_features(bond)
        edge_features.extend([bond_feat, bond_feat])

    # Create PyG Data object
    data = Data(
        x=torch.tensor(atom_features, dtype=torch.float32),
        pos=torch.tensor(positions, dtype=torch.float32),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        ikey=ikey, # Store Ikey for reference
        node_ids=node_identifiers 
    )
    return data

# --- Execute Graph Conversion ---
print("\n--- Starting 3D Graph Conversion (SDF -> PyG .pt files) ---")
# Process only the ligands for which we have an SDF but not a graph yet
sdf_files_available = {f.split('.')[0] for f in os.listdir(SDF_OUTPUT_DIR) if f.endswith('.sdf')}
graphs_to_create_ikeys = list(sdf_files_available - existing_graph_ikeys)
print(f"Found {len(graphs_to_create_ikeys)} new SDFs to convert to graphs.")

if graphs_to_create_ikeys:
    for ikey in tqdm(graphs_to_create_ikeys, desc="Converting SDF to PyG"):
        sdf_path = os.path.join(SDF_OUTPUT_DIR, f"{ikey}.sdf")
        graph_data = sdf_to_graph(sdf_path, ikey)
        
        if graph_data:
            torch.save(graph_data, os.path.join(GRAPH_OUTPUT_DIR, f"{ikey}.pt"))

    print(f"\nFinished graph conversion. New graphs saved in {GRAPH_OUTPUT_DIR}")
else:
    print("\n--- All graphs already exist. Skipping conversion. ---")
