# scripts/predict.py
"""
This script provides a command-line interface to predict GPCR-ligand activity
using a pre-trained GPCRact model. It takes a protein PDB file and a ligand
SMILES string as input, generates the necessary molecular graphs on-the-fly,
and outputs the predicted binding and activity class.
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
import yaml

# Third-party libraries
import numpy as np
import torch
from torch_geometric.data import Data, Batch

# RDKit for ligand processing
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem

# BioPython for protein processing
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa

# Add project root to Python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DAGN_HybridModel

# --- Suppress unnecessary warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# ON-THE-FLY GRAPH GENERATION
# These functions are adapted from the preprocessing scripts for single-input inference.
# ==============================================================================

# --- Ligand Graph Generation (from 03_generate_ligand_graphs.py) ---
def get_ligand_atom_features(atom):
    """Encodes ligand atom features into a feature vector."""
    possible_elements = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    features = [int(atom.GetSymbol() == s) for s in possible_elements] + [int(atom.GetSymbol() not in possible_elements)]
    features += [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization() == rdchem.HybridizationType.SP),
        int(atom.GetHybridization() == rdchem.HybridizationType.SP2),
        int(atom.GetHybridization() == rdchem.HybridizationType.SP3),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        int(atom.IsInRing())
    ]
    return features

def smiles_to_pyg_graph(smiles: str, random_seed: int = 42) -> Data:
    """Generates a PyTorch Geometric Data object from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError("RDKit could not parse the SMILES string.")
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv2()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol, params) == -1:
        warnings.warn("Could not generate 3D conformer. Using 2D coordinates.")
        AllChem.Compute2DCoords(mol)
    else:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol)
    
    atom_features, positions = [], []
    for atom in mol.GetAtoms():
        atom_features.append(get_ligand_atom_features(atom))
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([(i, j), (j, i)])

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float32),
        pos=torch.tensor(positions, dtype=torch.float32),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    )

# --- Protein Graph Generation (from 02_generate_protein_graphs.py) ---
KEY_ATOMS_MAP = {
    'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'LEU': ['CD1', 'CD2'], 'ILE': ['CG2', 'CD1'], 
    'PRO': ['CD'], 'PHE': ['CZ'], 'TYR': ['CZ', 'OH'], 'TRP': ['NE1', 'CH2'], 
    'SER': ['OG'], 'THR': ['OG1', 'CG2'], 'CYS': ['SG'], 'ASN': ['OD1', 'ND2'], 
    'GLN': ['OE1', 'NE2'], 'MET': ['SD', 'CE'], 'ASP': ['OD1', 'OD2'], 
    'GLU': ['OE1', 'OE2'], 'LYS': ['NZ'], 'ARG': ['CZ', 'NH2'], 'HIS': ['ND1', 'NE2']
}

def select_representative_atoms(residue):
    """Selects C-alpha and key functional atoms from a residue object."""
    if 'CA' not in residue: return None, []
    ca_atom = residue['CA']
    selected_atoms = set()
    key_atom_names = KEY_ATOMS_MAP.get(residue.get_resname(), [])
    for atom_name in key_atom_names:
        if atom_name in residue:
            selected_atoms.add(residue[atom_name])
    return ca_atom, list(selected_atoms)

def pdb_to_pyg_graph(pdb_path: Path, chain_id: str) -> Data:
    """Generates a PyTorch Geometric Data object from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    chain = structure[0][chain_id]
    
    residues = [res for res in chain if is_aa(res, standard=True)]
    if not residues: raise ValueError(f"No standard amino acid residues found in chain {chain_id}.")

    node_atoms, node_metadata = [], []
    for res in residues:
        ca_atom, rep_atoms = select_representative_atoms(res)
        if ca_atom:
            node_atoms.append(ca_atom)
            node_metadata.append({"role": 0, "res_obj": res, "atom_obj": ca_atom}) # role 0 for C-alpha
            for atom in rep_atoms:
                node_atoms.append(atom)
                node_metadata.append({"role": 1, "res_obj": res, "atom_obj": atom}) # role 1 for side-chain

    pos = torch.tensor([atom.get_coord() for atom in node_atoms], dtype=torch.float)
    
    # Heuristic Binding Site Definition: 25 residues closest to the geometric center
    ca_coords = np.array([res['CA'].get_coord() for res in residues if 'CA' in res])
    geometric_center = ca_coords.mean(axis=0)
    distances = np.linalg.norm(ca_coords - geometric_center, axis=1)
    bs_res_indices = {residues[i].get_id()[1] for i in np.argsort(distances)[:25]}

    # Feature Generation
    AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
    ELEMENT_MAP = {elem: i for i, elem in enumerate(['C', 'O', 'N', 'S', 'P', 'OTHER'])}
    
    node_features, node_roles_numeric, bs_mask_list = [], [], []
    for meta in node_metadata:
        res_obj, atom_obj = meta['res_obj'], meta['atom_obj']
        res_name_1 = three_to_one(res_obj.get_resname())
        
        # NOTE: For inference on new proteins, BS/DR features are unknown and set to 0.
        h_res_type = [1.0 if res_name_1 == code else 0.0 for code in AA_CODES]
        is_bs = [1.0] if res_obj.get_id()[1] in bs_res_indices else [0.0]
        ago_disp, ant_disp = [0.0], [0.0] 

        element_idx = [ELEMENT_MAP.get(atom_obj.element, ELEMENT_MAP['OTHER'])]
        
        ca_coord = res_obj['CA'].get_coord() if 'CA' in res_obj else np.zeros(3)
        rel_pos_vec = atom_obj.get_coord() - ca_coord
        
        # RDKit features are complex to generate without a full PDB context via RDKit Mol object
        # For simplicity in this script, we default them to zero.
        # For a more advanced implementation, the get_rdkit_features logic could be integrated.
        h_rdkit = [0.0] * 7

        feature_vector = h_res_type + is_bs + ago_disp + ant_disp + element_idx + list(rel_pos_vec) + [np.linalg.norm(rel_pos_vec)] + h_rdkit
        node_features.append(feature_vector)
        node_roles_numeric.append(meta['role'])
        bs_mask_list.append(is_bs[0] == 1.0 and meta['role'] == 0) # BS mask for C-alphas only

    x = torch.tensor(node_features, dtype=torch.float)
    x_float_clean = torch.cat([x[:, :20], x[:, 24:28], x[:, 28:]], dim=1)
    x_float_full = torch.cat([x[:, :20], x[:, 20:23], x[:, 24:28], x[:, 28:]], dim=1)
    
    return Data(
        x_float_clean=x_float_clean,
        x_float_full=x_float_full,
        x_elem=x[:, 23].long(),
        pos=pos,
        edge_index=torch.empty((2, 0), dtype=torch.long), # The model's EGNN layers build edges implicitly
        node_roles=torch.tensor(node_roles_numeric, dtype=torch.long),
        bs_mask=torch.tensor(bs_mask_list, dtype=torch.bool),
        # Dummy attributes to match training data structure
        gpcr_class_id=torch.tensor([[-1]]),
        gpcr_family_id=torch.tensor([[-1]])
    )

def main(args):
    """Main function to run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # 1. Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize model architecture from config
    print("Initializing model architecture...")
    model_params = config['model']
    # You may need to manually get num_classes and num_families if not in config
    # For this example, we hardcode them, but ideally they'd be in the config or a separate file.
    num_classes = 7 
    num_families = 32
    model = DAGN_HybridModel(
        protein_in_dim_clean=model_params['protein_in_dim_clean'],
        protein_in_dim_full=model_params['protein_in_dim_full'],
        ligand_in_dim=model_params['ligand_in_dim'],
        hidden_dim=model_params['hidden_dim'],
        protein_config={"type": model_params['protein_type'], "n_layers": model_params['protein_layers']},
        ligand_config={"type": model_params['ligand_type'], "n_layers": model_params['ligand_layers']},
        element_embedding_dim=model_params['element_embedding_dim'],
        dropout=model_params['dropout'],
        n_attn_heads=model_params['attention_heads'],
        propagation_attention_layers=model_params['propagation_attention_layers'],
        num_gpcr_classes=num_classes,
        num_gpcr_families=num_families
    ).to(device)

    # 3. Load pre-trained weights
    print(f"Loading pre-trained weights from {args.model_checkpoint}")
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.eval()

    # 4. Generate graphs from inputs
    try:
        ligand_graph = smiles_to_pyg_graph(args.smiles)
        protein_graph = pdb_to_pyg_graph(args.pdb, args.chain)
    except Exception as e:
        print(f"Error during graph generation: {e}")
        return

    # 5. Create a batch and run inference
    protein_batch = Batch.from_data_list([protein_graph]).to(device)
    ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        binding_logit, activity_type_logit, _ = model(protein_batch, ligand_batch)

    # 6. Interpret and print results
    binding_prob = torch.sigmoid(binding_logit).item()
    print("\n--- Prediction Results ---")
    print(f"Binding Probability: {binding_prob:.4f}")

    if binding_prob < 0.5:
        print("Predicted Class: Non-binder")
    else:
        print("Predicted Class: Binder")
        activity_probs = torch.softmax(activity_type_logit, dim=1).squeeze()
        activity_idx = torch.argmax(activity_probs).item()
        activity_class = "Antagonist" if activity_idx == 0 else "Agonist"
        
        print(f"  └ Predicted Activity: {activity_class}")
        print(f"    (Probabilities: Antagonist={activity_probs[0]:.4f}, Agonist={activity_probs[1]:.4f})")
    print("------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict GPCR-ligand activity using a pre-trained GPCRact model.")
    parser.add_argument("--pdb", type=Path, required=True, help="Path to the input GPCR PDB file.")
    parser.add_argument("--chain", type=str, default='A', help="Chain ID of the GPCR in the PDB file (default: A).")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the input ligand.")
    parser.add_argument("--model_checkpoint", type=Path, default="models/GPCRact_pretrained.pt", help="Path to the pre-trained model checkpoint.")
    parser.add_argument("--config", type=Path, default="configs/training_config.yaml", help="Path to the model configuration YAML file.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on.")
    
    args = parser.parse_args()
    main(args)