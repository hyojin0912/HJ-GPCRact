# src/data_loader.py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Dataset as PyGDataset, Batch

class GraphDataset(PyGDataset):
    """
    Loads the graph files with the new, cleaned data structure.
    - No more one-hot 'h_role' in the x matrix.
    - 'node_role' is loaded as a separate integer tensor.
    """
    def __init__(self, root, df, protein_graph_dir, ligand_graph_dir):
        self.df = df.reset_index(drop=True)
        self.protein_graph_dir = Path(protein_graph_dir)
        self.ligand_graph_dir = Path(ligand_graph_dir)
        super().__init__(root)

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        ikey, uniprot_id = row['Ikey'], row['AC']
        binding_label = row['Binding']
        # Use np.nan for missing values, then convert to a tensor-friendly format
        activity_label = row['Activity']
        activity_label_tensor_val = activity_label if not np.isnan(activity_label) else -1.0

        try:
            protein_graph = torch.load(self.protein_graph_dir / f"{uniprot_id}.pt", map_location='cpu', weights_only=False)
            ligand_graph = torch.load(self.ligand_graph_dir / f"{ikey}.pt", map_location='cpu', weights_only=False)
            
            original_x = protein_graph.x
            
            # --- START OF MODIFICATION ---
            # The new 'x' format from graph generation is:
            # AA(20)|is_bs(1)|ago(1)|ant(1)|elem_idx(1)|rel_pos(3)|dist(1)|rdkit(7)
            # Total dimensions = 20+1+2+1+3+1+7 = 35

            # 1. Slice features based on the NEW, CLEANED format.
            h_res_type = original_x[:, :20]
            h_is_bs    = original_x[:, 20:21]
            h_disp     = original_x[:, 21:23]
            # h_role is no longer in 'x'
            
            # The element index is now at column 23
            protein_graph.x_elem = original_x[:, 23].long()
            
            h_rel_pos  = original_x[:, 24:27]
            h_dist_ca  = original_x[:, 27:28]
            h_rdkit    = original_x[:, 28:]

            # 2. Reconstruct feature sets. 'h_role' is removed as it was redundant.
            # The model uses the separate 'node_roles' attribute for conditional logic.
            protein_graph.x_float_full = torch.cat([
                h_res_type, h_is_bs, h_disp, h_rel_pos, h_dist_ca, h_rdkit
            ], dim=1)

            protein_graph.x_float_clean = torch.cat([
                h_res_type, h_rel_pos, h_dist_ca, h_rdkit
            ], dim=1)
            
            # 3. Clean up the original 'x' tensor
            del protein_graph.x

            # 4. Rename 'node_role' to 'node_roles' to match what the model expects.
            # The data is already a tensor, so no conversion is needed.
            protein_graph.node_roles = protein_graph.node_role
            del protein_graph.node_role

            # --- END OF MODIFICATION ---

            # Add labels and identifiers (this part is unchanged)
            protein_graph.binding_label = torch.tensor([binding_label], dtype=torch.float)
            protein_graph.activity_label = torch.tensor([activity_label_tensor_val], dtype=torch.float)
            protein_graph.ikey = ikey
            protein_graph.uniprot_id = uniprot_id

            return protein_graph, ligand_graph

        except FileNotFoundError:
            print(f"Warning: File not found for idx {idx} (ikey: {ikey}, uniprot: {uniprot_id}). Skipping.")
            return None, None

tqdm.pandas(desc="Validating graph files") 
def get_valid_indices(df, protein_dir, ligand_dir):
    """
    Systematically validates data pairs. Checks for:
    1. Existence of both protein and ligand graph files.
    2. The ligand graph having more than one node.
    3. The protein graph being loadable and containing essential attributes.
    """
    print("Systematically validating graph files (this may take a while on first run)...")
    
    # Helper function to check a single row
    def is_valid_pair(row):
        p_path = protein_dir / f"{row['AC']}.pt"
        l_path = ligand_dir / f"{row['Ikey']}.pt"
        
        # 1. Check file existence first (fast check)
        if not p_path.exists() or not l_path.exists():
            return False
        
        try:
            # 2. Load ligand graph and check node count
            ligand_graph = torch.load(l_path, map_location='cpu', weights_only=False)
            if ligand_graph.num_nodes <= 1 or ligand_graph.num_edges == 0:
                return False

            # --- ✅ NEW VALIDATION STEP ---
            # 3. Load protein graph and check for essential attributes for the model
            protein_graph = torch.load(p_path, map_location='cpu', weights_only=False)
            # Check if attributes required by the model's forward pass exist
            if not hasattr(protein_graph, 'gpcr_class_id') or \
               not hasattr(protein_graph, 'gpcr_family_id') or \
               not hasattr(protein_graph, 'node_role'): # or any other essential attribute
                # This graph is missing critical metadata, mark as invalid
                return False
        except Exception as e:
            return False

        # If all checks pass, it's a valid pair
        return True

    # Use .progress_apply to show a progress bar
    valid_mask = df.progress_apply(is_valid_pair, axis=1)
    return valid_mask.to_numpy()

def collate_fn(data_list):
    valid_data = [item for item in data_list if item[0] is not None]
    if not valid_data: return None, None
    p, l = zip(*valid_data)
    
    protein_batch = Batch.from_data_list(p)
    ligand_batch = Batch.from_data_list(l)
    
    # Identifiers are already in the protein_graph object from the Dataset
    return protein_batch, ligand_batch