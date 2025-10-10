# --- Core Libraries ---
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from collections import defaultdict
import ast

# --- RDKit for Chemical Features ---
from rdkit import Chem
from rdkit.Chem import AllChem

# --- PyTorch & PyG ---
import torch
from torch_geometric.data import Data

# --- Cheminformatics & Bioinformatics ---
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

# --- For Distance-based Edges ---
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
# --- Configuration ---
warnings.filterwarnings("ignore")

# --- Input file paths (Using Heavy-Atom based definitions) ---
BS_RESIDUES_FILE = "../Output/Region_Definitions/Binding_Sites_Heavy_Atom_based.csv"
DR_RESIDUES_FILE = "../Output/DA_Analysis/Differential_Residues_Heavy_Atom_based.csv"
REP_APO_FILE = "../Output/Final/Representative_Apo_Structures.csv"
REP_CHAIN_FILE = "../Output/Binding_Residue/Rep_GPCR_chain.csv"
SEQUENCE_INFO_FILE = "../Input/Human_GPCR_PDB_Info.csv"
GPCR_INFO_FILE = "../Input/ChEMBL_GPCR_Info.csv"

DATA_DIR = Path("../Data")
CIF_DIR = DATA_DIR / "CIF_Files"
AF_DIR = DATA_DIR / "AF_PDB"

K_NEIGHBORS = 128 
DATA_DIR = Path("../Data")
OUTPUT_DIR = Path("../Data/Protein_Graphs_PyG/") # New output dir
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 1. Load and Pre-process All Prerequisite Data ---
print("--- Loading and Pre-processing Heavy-Atom Data ---")
bs_df = pd.read_csv(BS_RESIDUES_FILE)
dr_df = pd.read_csv(DR_RESIDUES_FILE)
rep_apo_df = pd.read_csv(REP_APO_FILE)
rep_chain_df = pd.read_csv(REP_CHAIN_FILE)
seq_df = pd.read_csv(SEQUENCE_INFO_FILE)

# Representative Apo structure map
rep_apo_map = {}
exp_apo_df = rep_apo_df[rep_apo_df['Binding_Coverage'] == 100.0].sort_values('Resolution').drop_duplicates('UniProt_ID')
for _, row in exp_apo_df.iterrows():
    rep_apo_map[row['UniProt_ID']] = {'id': row['PDB_ID'], 'type': 'PDB'}
all_uniprot_ids = set(bs_df['UniProt_ID'].unique()).union(set(dr_df['uniprot_ac'].unique()))
for uid in all_uniprot_ids:
    if uid not in rep_apo_map:
        rep_apo_map[uid] = {'id': uid, 'type': 'AF2'}

# Other lookups (no changes needed here)
rep_chain_lookup = rep_chain_df.sort_values('score', ascending=False).drop_duplicates('PDB_ID').set_index('PDB_ID')['chain_id'].to_dict()
uniprot_seq_map = seq_df.set_index('Entry')['Sequence'].to_dict()

# Process BS and DR data from new files ---
# Create a union of all binding site residues for each UniProt ID
bs_df['Binding_Site_Residues'] = bs_df['Binding_Site_Residues'].apply(ast.literal_eval)
bs_residues_map = bs_df.groupby('UniProt_ID')['Binding_Site_Residues'].apply(
    lambda lists: set(item for sublist in lists for item in sublist)
).to_dict()

print("Processing DRs: Selecting Top 100 per protein based on C-alpha displacement.")

# 2. Select a unique representative DR for each residue number based on max C-alpha displacement
# A single residue might be listed multiple times from different PDBs. We pick the one with the largest C-alpha shift.
dr_unique_df = dr_df.sort_values('ca_displacement_A', ascending=False).drop_duplicates(subset=['uniprot_ac', 'uniprot_res_num'])

# 3. For each UniProt ID, get the Top 100 DRs
top_dr_df = dr_unique_df.groupby('uniprot_ac').head(100).reset_index(drop=True)

# 4. Create the new dr_residues_map from the Top 100 DRs
dr_residues_map = top_dr_df.groupby('uniprot_ac')['uniprot_res_num'].apply(set).to_dict()

# 5. Create the dr_disp_map for feature generation, using only the selected Top 100 DRs
# This map still uses 'max_displacement_A' for the feature value, as in the original code.
dr_disp_map = defaultdict(lambda: defaultdict(lambda: {'agonist': 0.0, 'antagonist': 0.0}))
for _, row in top_dr_df.iterrows():
    uniprot_id = row['uniprot_ac']
    res_num = row['uniprot_res_num']
    moa = row['moa']
    # We use max_displacement_A for the feature, even though selection was by ca_displacement_A
    displacement = row['max_displacement_A'] 
    
    # The original logic for assigning displacement values is preserved
    if displacement > dr_disp_map[uniprot_id][res_num][moa]:
        dr_disp_map[uniprot_id][res_num][moa] = displacement

# --- 2. Helper Functions Setup ---
print("\n--- Initializing Helper Functions ---")
cif_parser = MMCIFParser(QUIET=True)
pdb_parser = PDBParser(QUIET=True)

# MODIFIED: To return structure_path for RDKit
def load_structure(source_id, structure_type):
    path = CIF_DIR / f"{source_id.lower()}.cif" if structure_type == 'PDB' else AF_DIR / f"AF-{source_id}-F1-model_v3.pdb"
    parser = cif_parser if structure_type == 'PDB' else pdb_parser
    if not path.exists(): return None, None
    try:
        structure = parser.get_structure(source_id, str(path))
        return structure, path
    except Exception:
        return None, None

# NEW: Function to extract RDKit features (imported from heavy-atom script)
def get_rdkit_features(pdb_path, chain_id):
    if not pdb_path or not os.path.exists(pdb_path):
        return {}

    VALID_ELEMENTS = {
        'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 
        'MG', 'ZN', 'MN', 'CA', 'FE', 'NA', 'K' 
    }
    
    cleaned_pdb_lines = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    element_symbol = line[76:78].strip().upper()
                    if element_symbol in VALID_ELEMENTS:
                        cleaned_pdb_lines.append(line)
    except Exception as e:
        print(f"Warning: Could not read or process file {pdb_path}. Error: {e}")
        return {}

    if not cleaned_pdb_lines: return {}
    pdb_block = "".join(cleaned_pdb_lines)
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)

    if not mol:
        print(f"Warning: RDKit failed to create molecule from cleaned PDB: {pdb_path}")
        return {}

    rdkit_feature_map = {}
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]

    for atom in mol.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info and pdb_info.GetChainId().strip() == chain_id:
            res_num = pdb_info.GetResidueNumber()
            atom_name = pdb_info.GetName().strip()

            formal_charge = atom.GetFormalCharge()
            is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
            hybrid = atom.GetHybridization()
            h_hybrid = [1.0 if hybrid == h_type else 0.0 for h_type in hybridization_types]

            rdkit_feature_map[(res_num, atom_name)] = [formal_charge, is_aromatic] + h_hybrid
    return rdkit_feature_map

def create_uniprot_to_pdb_res_map(chain_obj, uniprot_seq):
    if not uniprot_seq or not chain_obj: return None
    pdb_res_list = [res for res in chain_obj if is_aa(res, standard=True)]
    if not pdb_res_list: return None
    pdb_seq_1letter = "".join([three_to_one(res.get_resname()) for res in pdb_res_list])
    alignments = pairwise2.align.localds(uniprot_seq, pdb_seq_1letter, matlist.blosum62, -10, -0.5)
    if not alignments: return None
    aligned_uni_seq, aligned_pdb_seq = alignments[0][:2]
    uni_to_pdb_map, pdb_idx, uni_pos = {}, 0, 0
    for uni_char, pdb_char in zip(aligned_uni_seq, aligned_pdb_seq):
        if uni_char != '-': uni_pos += 1
        if pdb_char != '-':
            if pdb_idx < len(pdb_res_list):
                if uni_char != '-':
                    uni_to_pdb_map[uni_pos] = pdb_res_list[pdb_idx]
                pdb_idx += 1
    return uni_to_pdb_map

print("Helper functions are ready.")


# Function based atom selection
KEY_ATOMS_MAP = {
    # --- Aliphatic ---
    'ALA': ['CB'],      # Simple methyl group.
    'VAL': ['CG1', 'CG2'],# Both methyls define the 'V' shape and steric properties.
    'LEU': ['CD1', 'CD2'],# Both terminal methyls are key for hydrophobic interactions.
    'ILE': ['CG2', 'CD1'],# Captures both the beta-branch (CG2) and the chain terminus (CD1).
    'PRO': ['CD'],      # The atom completing the rigid pyrrolidine ring, critical for backbone constraints.

    # --- Aromatic ---
    'PHE': ['CZ'],      # Represents the planar phenyl ring's extremity.
    'TYR': ['CZ', 'OH'],  # CZ for the ring, OH for the key hydrogen-bonding hydroxyl group.
    'TRP': ['NE1', 'CH2'],# NE1 is the H-bond donor in the indole ring; CH2 is the ring's geometric end.

    # --- Polar / Uncharged ---
    'SER': ['OG'],      # Key hydroxyl group.
    'THR': ['OG1', 'CG2'],# OG1 for polarity (hydroxyl) and CG2 for steric bulk (methyl).
    'CYS': ['SG'],      # The highly reactive and functionally critical thiol sulfur.
    'ASN': ['OD1', 'ND2'],# Both the carbonyl oxygen and amide nitrogen are crucial for H-bonding.
    'GLN': ['OE1', 'NE2'],# Same logic as ASN for the longer sidechain.

    # --- Sulfur-containing ---
    'MET': ['SD', 'CE'],  # SD is the flexible, polarizable sulfur; CE is the terminal hydrophobic methyl.

    # --- Charged (Acidic) ---
    'ASP': ['OD1', 'OD2'],# The entire carboxylate group defines its negative charge and interaction potential.
    'GLU': ['OE1', 'OE2'],# Same logic as ASP for the longer sidechain.

    # --- Charged (Basic) ---
    'LYS': ['NZ'],      # The terminal primary amine that holds the positive charge.
    'ARG': ['CZ', 'NH2'],# CZ is the central carbon of the planar guanidinium group; NH2 is a terminal nitrogen. This captures the resonance and H-bond capacity.
    'HIS': ['ND1', 'NE2'] # Both imidazole nitrogens are vital for its role as a proton donor/acceptor and metal ligand.
}

def select_representative_atoms(residue_obj):
    # Selects a comprehensive set of representative atoms for a given residue.
    if 'CA' not in residue_obj:
        return None, []

    ca_atom = residue_obj['CA']
    ca_coord = ca_atom.get_coord()
    
    # Use a set to store unique representative atoms
    selected_atoms = set()

    # --- Rule 1: Find the geometrically furthest heavy atom ---
    max_dist = 0.0
    furthest_atom = None
    for atom in residue_obj.get_atoms():
        if atom.element != 'H':  # Only consider heavy atoms
            dist = np.linalg.norm(atom.get_coord() - ca_coord)
            if dist > max_dist:
                max_dist = dist
                furthest_atom = atom
    
    if furthest_atom and furthest_atom.get_name() != 'CA':
        selected_atoms.add(furthest_atom)

    # --- Rule 2: Find all predefined key chemical/structural atoms ---
    res_name = residue_obj.get_resname()
    key_atom_names = KEY_ATOMS_MAP.get(res_name, [])
    for atom_name in key_atom_names:
        if atom_name in residue_obj:
            selected_atoms.add(residue_obj[atom_name])

    return ca_atom, list(selected_atoms)


print("\n--- Loading and Processing GPCR Class/Family Information ---")
# Load the GPCR info file
gpcr_info_df = pd.read_csv(GPCR_INFO_FILE)

# Handle potential multiple UniProt IDs in a single cell (though not in the example)
# This makes the code more robust.
gpcr_info_df['UniProt Accessions'] = gpcr_info_df['UniProt Accessions'].str.split('; ')
gpcr_info_df = gpcr_info_df.explode('UniProt Accessions').dropna(subset=['UniProt Accessions'])

# Create unique integer IDs for 'Class'
all_classes = sorted(gpcr_info_df['Class'].unique())
class_to_id = {name: i for i, name in enumerate(all_classes)}
gpcr_info_df['class_id'] = gpcr_info_df['Class'].map(class_to_id)

# Create unique integer IDs for 'Receptor Family'
all_families = sorted(gpcr_info_df['Receptor Family'].unique())
family_to_id = {name: i for i, name in enumerate(all_families)}
gpcr_info_df['family_id'] = gpcr_info_df['Receptor Family'].map(family_to_id)

# Create the final lookup dictionaries from UniProt ID to integer ID
uniprot_to_class_id = pd.Series(
    gpcr_info_df.class_id.values, 
    index=gpcr_info_df['UniProt Accessions']
).to_dict()

uniprot_to_family_id = pd.Series(
    gpcr_info_df.family_id.values, 
    index=gpcr_info_df['UniProt Accessions']
).to_dict()

# Save the mappings for later use in model training
import json
with open(OUTPUT_DIR / 'class_to_id.json', 'w') as f:
    json.dump(class_to_id, f, indent=4)
with open(OUTPUT_DIR / 'family_to_id.json', 'w') as f:
    json.dump(family_to_id, f, indent=4)


# --- 3. Main Graph Construction Loop ---
for uniprot_id, apo_info in tqdm(rep_apo_map.items(), desc="Generating Enhanced Protein Graphs"):
    
    output_path = OUTPUT_DIR / f"{uniprot_id}.pt"
    if output_path.exists(): continue

    try:
        bs_set = bs_residues_map.get(uniprot_id, set())
        dr_set = dr_residues_map.get(uniprot_id, set())
        
        target_residue_numbers = sorted(list(bs_set.union(dr_set)))
        if not target_residue_numbers: continue

        # MODIFIED: Handle two return values
        structure, structure_path = load_structure(apo_info['id'], apo_info['type'])
        sequence = uniprot_seq_map.get(uniprot_id)
        if not structure or not sequence: continue

        chain_id = rep_chain_lookup.get(apo_info['id']) if apo_info['type'] == 'PDB' else 'A'
        if not chain_id or chain_id not in structure[0]: continue
        
        chain_obj = structure[0][chain_id]
        uniprot_to_pdb_res_map = create_uniprot_to_pdb_res_map(chain_obj, sequence)
        if not uniprot_to_pdb_res_map: continue
        
        # NEW: Pre-fetch RDKit features for the entire structure
        rdkit_features = get_rdkit_features(structure_path, chain_id)

        # --- NODE SELECTION (Ensemble: C-alpha + Furthest + Key Atoms) --- # UPDATED
        node_atoms = []
        node_metadata = []

        for res_num in target_residue_numbers:
            res_obj = uniprot_to_pdb_res_map.get(res_num)
            if res_obj:
                # --- Call the NEW, COMPREHENSIVE function ---
                ca_atom, representative_atoms_list = select_representative_atoms(res_obj)

                if ca_atom:
                    # 1. Add the C-alpha atom as a node
                    node_atoms.append(ca_atom)
                    node_metadata.append({
                        "role": "ca", 
                        "res_num": res_num, 
                        "res_obj": res_obj,
                        "dist_from_ca": 0.0, 
                        "atom_obj": ca_atom
                    })

                    # 2. Iterate through the list of other representative atoms and add each as a node
                    ca_coord = ca_atom.get_coord()
                    for rep_atom in representative_atoms_list:
                        dist = np.linalg.norm(rep_atom.get_coord() - ca_coord)
                        node_atoms.append(rep_atom)
                        node_metadata.append({
                            # Role is now more general - can be 'furthest', 'functional', or both
                            "role": "functional", 
                            "res_num": res_num, 
                            "res_obj": res_obj,
                            "dist_from_ca": dist, 
                            "atom_obj": rep_atom
                        })
        
        if len(node_atoms) < 10: continue
           
        # --- EDGE CONSTRUCTION (k-Nearest Neighbors) ---
        pos = torch.tensor([atom.get_coord() for atom in node_atoms], dtype=torch.float)
        num_nodes = len(node_atoms)

        # Ensure k is not greater than the number of available neighbors
        actual_k = min(K_NEIGHBORS, num_nodes - 1)

        if actual_k < 1: # If not enough nodes to form any edge
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            # Build KDTree for efficient neighbor search
            tree = KDTree(pos.numpy())

            # Query the tree to find k+1 nearest neighbors for each node
            # (the first neighbor is always the node itself)
            distances, indices = tree.query(pos.numpy(), k=actual_k + 1)

            # Create edge list
            # Source nodes: [0, 0, ..., 1, 1, ..., n, n, ...]
            edge_src = np.repeat(np.arange(num_nodes), actual_k)

            # Destination nodes: flatten the neighbor indices, skipping the first column (self-loops)
            edge_dst = indices[:, 1:].flatten()

            edge_index = torch.from_numpy(np.vstack([edge_src, edge_dst])).long()

        # --- ENHANCED NODE FEATURIZATION ---
        AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
        # NEW: Integer mapping for elements
        ELEMENT_MAP = {elem: i for i, elem in enumerate(['C', 'O', 'N', 'S', 'P', 'OTHER'])}
        
        node_features = []
        default_rdkit_feats = [0.0] * 7 # Default if RDKit lookup fails

        # ✅ NEW: Create a numeric list for node roles
        node_roles_numeric = []

        for meta in node_metadata:
            res_obj = meta['res_obj']
            res_num = meta['res_num']
            res_name_3 = res_obj.get_resname()
            res_name_1 = three_to_one(res_name_3) if is_aa(res_obj, standard=True) else 'X'

            # ✅ MODIFIED: Append integer-encoded role to the list
            node_roles_numeric.append(0 if meta['role'] == 'ca' else 1)
            
            # 1. Common Features (23 dims)
            h_res_type = [1.0 if res_name_1 == code else 0.0 for code in AA_CODES]
            is_bs = [1.0] if res_num in bs_set else [0.0]
            dr_info = dr_disp_map.get(uniprot_id, {}).get(res_num, {'agonist': 0.0, 'antagonist': 0.0})
            ago_disp = [dr_info.get('agonist', 0.0)]
            ant_disp = [dr_info.get('antagonist', 0.0)]
            
            # 2. Individual Features
            atom = meta['atom_obj']
            
            # MODIFIED: Element Type (Integer Index)
            element_idx = [ELEMENT_MAP.get(atom.element, ELEMENT_MAP['OTHER'])] # 1 dim
            
            # NEW: Relative Position to C-alpha
            if 'CA' in res_obj:
                ca_coord = res_obj['CA'].get_coord()
                atom_coord = atom.get_coord()
                rel_pos_vec = atom_coord - ca_coord
            else: # Fallback if CA is somehow missing
                rel_pos_vec = np.array([0.0, 0.0, 0.0])
            h_rel_pos_to_ca = list(rel_pos_vec) # 3 dims
            
            # Original distance from C-alpha feature
            h_dist_from_ca = [meta['dist_from_ca']] # 1 dim

            # NEW: RDKit Chemical Features
            res_pdb_num = res_obj.get_id()[1]
            atom_name = atom.get_name().strip()
            atom_key = (res_pdb_num, atom_name)
            h_rdkit = rdkit_features.get(atom_key, default_rdkit_feats) # 7 dims
            
            # Assemble the final feature vector (Total: 20+3 + 2+1+3+1+7 = 37 dims)
            # MODIFIED: Assemble the feature vector without h_role
            final_feature = (h_res_type + is_bs + ago_disp + ant_disp +
                             element_idx + h_rel_pos_to_ca + 
                             h_dist_from_ca + h_rdkit)
            node_features.append(final_feature)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # --- Create final PyG Data object ---
        node_res_nums_tensor = torch.tensor([meta['res_num'] for meta in node_metadata], dtype=torch.long)
        bs_mask = torch.tensor([res_num in bs_set for res_num in node_res_nums_tensor.tolist()], dtype=torch.bool)
        
        protein_graph = Data(x=x, pos=pos, edge_index=edge_index,
                             uniprot_id=uniprot_id,
                             node_res_num=node_res_nums_tensor,
                             node_role=torch.tensor(node_roles_numeric, dtype=torch.long),
                             bs_mask=bs_mask
                            )
        
        torch.save(protein_graph, output_path)
    
    except Exception as e:
        print(f"Failed to process {uniprot_id}. Error: {e}")
        pass

print(f"\n--- Graph generation complete! Graphs saved to {OUTPUT_DIR} ---")
