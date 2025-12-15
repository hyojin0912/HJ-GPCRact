import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Dataset as PyGDataset, Batch

class GraphDataset(PyGDataset):
    """
    Loads graph files with cleaned data structure.
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
        activity_label = row['Activity']
        activity_label_tensor_val = activity_label if not np.isnan(activity_label) else -1.0

        try:
            protein_graph = torch.load(self.protein_graph_dir / f"{uniprot_id}.pt", map_location='cpu', weights_only=False)
            ligand_graph = torch.load(self.ligand_graph_dir / f"{ikey}.pt", map_location='cpu', weights_only=False)
            
            original_x = protein_graph.x
            
            # [AA(20) | BS(1) | ElemIdx(1) | RelPos(3) | DistFromCA(1) | RDKit(7)]
            # Indices: 0-19     20          21           22-24          25            26-32
            h_res_type = original_x[:, :20]
            protein_graph.x_elem = original_x[:, 21].long()
            
            h_rel_pos  = original_x[:, 22:25]
            h_dist_ca  = original_x[:, 25:26]
            h_rdkit    = original_x[:, 26:]

            protein_graph.x_float_full = original_x.clone() # Keep full features if needed for other embeddings
            
            h_is_bs = original_x[:, 20:21]

            protein_graph.x_float_full = torch.cat([
                h_res_type, h_is_bs, h_rel_pos, h_dist_ca, h_rdkit
            ], dim=1)

            protein_graph.x_float_clean = torch.cat([
                h_res_type, h_rel_pos, h_dist_ca, h_rdkit
            ], dim=1)
            
            del protein_graph.x

            protein_graph.node_roles = protein_graph.node_role
            del protein_graph.node_role

            protein_graph.binding_label = torch.tensor([binding_label], dtype=torch.float)
            protein_graph.activity_label = torch.tensor([activity_label_tensor_val], dtype=torch.float)
            protein_graph.ikey = ikey
            protein_graph.uniprot_id = uniprot_id

            return protein_graph, ligand_graph

        except FileNotFoundError:
            return None, None

def get_valid_indices(df, protein_dir, ligand_dir):
    """
    Validates data pairs.
    """
    protein_dir = Path(protein_dir)
    ligand_dir = Path(ligand_dir)
    
    def is_valid_pair(row):
        p_path = protein_dir / f"{row['AC']}.pt"
        l_path = ligand_dir / f"{row['Ikey']}.pt"
        
        if not p_path.exists() or not l_path.exists():
            return False
        
        try:
            ligand_graph = torch.load(l_path, map_location='cpu', weights_only=False)
            if ligand_graph.num_nodes <= 1 or ligand_graph.num_edges == 0:
                return False

            protein_graph = torch.load(p_path, map_location='cpu', weights_only=False)
            if not hasattr(protein_graph, 'node_role'):
                return False
            
        except Exception:
            return False

        return True

    tqdm.pandas(desc="Validating graph files")
    valid_mask = df.progress_apply(is_valid_pair, axis=1)
    return valid_mask.to_numpy()

def collate_fn(data_list):
    valid_data = [item for item in data_list if item[0] is not None]
    if not valid_data: return None, None
    p, l = zip(*valid_data)
    return Batch.from_data_list(p), Batch.from_data_list(l)