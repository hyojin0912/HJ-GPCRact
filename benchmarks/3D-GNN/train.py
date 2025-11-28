import sys
import os
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import Dataset as PyGDataset, Batch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add, scatter_mean

# AMP (Automatic Mixed Precision) utilities
from torch.cuda import amp
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# ==============================================================================
# 1. SETUP & UTILITIES
# ==============================================================================
def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_best = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta
        self.path = path
        self.mode = mode

    def __call__(self, score, model):
        if self.mode == 'min':
            current_score = -score
            best_score_comp = -self.val_score_best
        else: # mode == 'max'
            current_score = score
            best_score_comp = self.val_score_best

        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(score, model)
        elif current_score < best_score_comp + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(f'Validation score improved ({self.val_score_best:.6f} --> {score:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_score_best = score

# ==============================================================================
# 2. MODEL DEFINITIONS (EGNN & UnifiedEGNN)
# ==============================================================================
# Note: Copied from original EGNN implementation to ensure standalone reproducibility
def unsorted_segment_sum(data, segment_ids, num_segments):
    out = data.new_zeros((num_segments, data.size(1)))
    scatter_add(data, segment_ids, out=out, dim=0)
    return out

class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        self.node_norm = nn.LayerNorm(output_nf)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp_list = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp_list.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp_list)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
        if self.normalize:
            norm = torch.sqrt(radial + self.epsilon)
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def edge_model(self, h_row, h_col, radial, edge_attr):
        if edge_attr is not None:
            out = torch.cat([h_row, h_col, radial, edge_attr], dim=1)
        else:
            out = torch.cat([h_row, h_col, radial], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_feat, node_attr=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        coord_diff_normalized = coord_diff / (torch.norm(coord_diff, dim=-1, keepdim=True) + self.epsilon)
        trans = coord_diff_normalized * self.coord_mlp(edge_feat)
        if self.coords_agg=='sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg=='mean':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0)) / (unsorted_segment_sum(torch.ones_like(trans), row, num_segments=coord.size(0)) + 1e-8)
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        h_row, h_col = h[row], h[col]
        e_ij = self.edge_model(h_row, h_col, radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, e_ij)
        h = self.node_model(h, edge_index, e_ij, node_attr)
        h = self.node_norm(h)
        return h, coord, e_ij

class E_GCL_Gated(E_GCL):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL_Gated, self).__init__(input_nf, output_nf, hidden_nf, edges_in_d, act_fn, residual, attention, normalize, coords_agg, tanh)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
            nn.Sigmoid()
        )
        self.node_norm = nn.LayerNorm(output_nf)

    def node_model(self, x, edge_index, edge_feat, node_attr=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        if node_attr is not None:
            agg_cat = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg_cat = torch.cat([x, agg], dim=1)
        update_val = self.node_mlp(agg_cat)
        gate_val = self.gate_mlp(agg_cat)
        if self.residual:
            out = x * gate_val + (x + update_val) * (1 - gate_val)
        else:
            out = x * gate_val + update_val * (1 - gate_val)
        out = self.node_norm(out)
        return out

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, n_layers=4, residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf, residual=residual, attention=attention, normalize=normalize, coords_agg=coords_agg, tanh=tanh))
    
    def forward(self, h, coord, edge_index, edge_attr=None):
        device = self.embedding_in.weight.device
        h, coord, edge_index = h.to(device), coord.to(device), edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, coord, _ = self._modules["gcl_%d" % i](h, edge_index, coord, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, coord

class EGNN_Gated_GlobalResidual(EGNN):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, n_layers=4, residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGNN, self).__init__(in_node_nf, hidden_nf, out_node_nf, in_edge_nf, n_layers, residual, attention, normalize, coords_agg, tanh)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_Gated(input_nf=hidden_nf, output_nf=hidden_nf, hidden_nf=hidden_nf, edges_in_d=in_edge_nf, residual=residual, attention=attention, normalize=normalize, coords_agg=coords_agg, tanh=tanh))

    def forward(self, h, coord, edge_index, edge_attr=None):
        device = self.embedding_in.weight.device
        h, coord, edge_index = h.to(device), coord.to(device), edge_index.to(device)
        if edge_attr is not None: edge_attr = edge_attr.to(device)
        h_initial = self.embedding_in(h)
        h = h_initial
        for i in range(0, self.n_layers):
            h, coord, _ = self._modules[f"gcl_{i}"](h, edge_index, coord, edge_attr=edge_attr)
        h = h + h_initial
        h = self.embedding_out(h)
        return h, coord

def create_encoder(config, in_dim, hidden_dim):
    encoder_type = config['type']
    kwargs = {'in_node_nf': in_dim, 'hidden_nf': hidden_dim, 'out_node_nf': hidden_dim, 'n_layers': config['n_layers'], 'attention': True, 'tanh': True}
    if encoder_type == 'gated_residual':
        return EGNN_Gated_GlobalResidual(**kwargs)
    else:
        # Simplified for benchmark script to only support used type or extend as needed
        raise ValueError(f"Encoder type {encoder_type} not implemented in benchmark script.")

class UnifiedEGNN(nn.Module):
    """
    [UNIFIED BASELINE]
    A monolithic, generic 3D-GNN baseline (Unified-EGNN).
    """
    def __init__(self, protein_in_dim_full, ligand_in_dim, hidden_dim, protein_config, ligand_config, element_embedding_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.element_embedding = nn.Embedding(num_embeddings=6, embedding_dim=element_embedding_dim)
        self.ligand_encoder = create_encoder(ligand_config, ligand_in_dim, hidden_dim)
        self.protein_embedding_ca = nn.Linear(protein_in_dim_full, hidden_dim)
        self.protein_embedding_sc = nn.Linear(protein_in_dim_full, hidden_dim)
        self.unified_encoder = create_encoder(protein_config, hidden_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3) 
        )

    def forward(self, protein_batch, ligand_batch):
        h_l, _ = self.ligand_encoder(ligand_batch.x, ligand_batch.pos, ligand_batch.edge_index)
        v_l = scatter_mean(h_l, ligand_batch.batch, dim=0)

        p_features_full = torch.cat([protein_batch.x_float_full, self.element_embedding(protein_batch.x_elem)], dim=1)
        h_p = torch.zeros(p_features_full.size(0), self.hidden_dim, device=p_features_full.device)
        ca_mask = (protein_batch.node_roles == 0)
        sc_mask = (protein_batch.node_roles == 1)
        h_p[ca_mask] = self.protein_embedding_ca(p_features_full[ca_mask]).to(h_p.dtype)
        h_p[sc_mask] = self.protein_embedding_sc(p_features_full[sc_mask]).to(h_p.dtype)

        v_l_expanded = v_l[protein_batch.batch]
        h_p_injected = h_p + v_l_expanded

        h_p_final, _ = self.unified_encoder(h_p_injected, protein_batch.pos, protein_batch.edge_index)
        h_p_final = self.final_norm(h_p_final)
        v_p_final = scatter_mean(h_p_final, protein_batch.batch, dim=0)
        
        logits_3class = self.prediction_head(v_p_final)
        return logits_3class

# ==============================================================================
# 3. DATASET DEFINITION
# ==============================================================================
class GraphDataset(PyGDataset):
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
            protein_graph = torch.load(self.protein_graph_dir / f"{uniprot_id}.pt", map_location='cpu')
            ligand_graph = torch.load(self.ligand_graph_dir / f"{ikey}.pt", map_location='cpu')
            
            original_x = protein_graph.x
            # Feature slicing tailored for UnifiedEGNN (match your notebook logic)
            h_res_type = original_x[:, :20]
            h_is_bs    = original_x[:, 20:21]
            h_disp     = original_x[:, 21:23]
            protein_graph.x_elem = original_x[:, 23].long()
            h_rel_pos  = original_x[:, 24:27]
            h_dist_ca  = original_x[:, 27:28]
            h_rdkit    = original_x[:, 28:]

            protein_graph.x_float_full = torch.cat([h_res_type, h_is_bs, h_disp, h_rel_pos, h_dist_ca, h_rdkit], dim=1)
            
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

# ==============================================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0
    scaler = amp.GradScaler()
    optimizer.zero_grad()

    for i, data in enumerate(tqdm(loader, desc="Training", leave=False)):
        if data[0] is None: continue
        protein_batch, ligand_batch = data
        protein_batch, ligand_batch = protein_batch.to(device), ligand_batch.to(device)
        
        binding_labels = protein_batch.binding_label.squeeze(-1)
        activity_labels = protein_batch.activity_label.squeeze(-1)
        
        final_targets = torch.zeros_like(binding_labels, dtype=torch.long)
        final_targets[activity_labels == 0] = 1 
        final_targets[activity_labels == 1] = 2 

        with amp.autocast():
            logits = model(protein_batch, ligand_batch)
            loss = criterion(logits, final_targets.to(device))
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
            
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, return_df=False):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    all_ikeys, all_uniprots = [], []

    for data in tqdm(loader, desc="Evaluating", leave=False):
        if data is None or data[0] is None: continue
        protein_batch, ligand_batch = data
        protein_batch, ligand_batch = protein_batch.to(device), ligand_batch.to(device)

        all_ikeys.extend(protein_batch.ikey)
        all_uniprots.extend(protein_batch.uniprot_id)

        activity_labels = protein_batch.activity_label.squeeze(-1).cpu()
        binding_labels = protein_batch.binding_label.squeeze(-1).cpu()

        logits = model(protein_batch, ligand_batch)
        predicted_class = torch.argmax(logits, dim=1).cpu()
        
        all_preds.append(predicted_class)
        all_logits.append(logits.cpu())

        final_targets = torch.zeros_like(binding_labels, dtype=torch.long)
        final_targets[activity_labels == 0] = 1 
        final_targets[activity_labels == 1] = 2 
        all_labels.append(final_targets)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    
    bacc = balanced_accuracy_score(all_labels, all_preds)

    if return_df:
        results_df = pd.DataFrame({
            'Ikey': all_ikeys, 'UniProt': all_uniprots,
            'Final_Label': all_labels,
            'Final_Pred': all_preds,
            'Logit_NB': all_logits[:, 0],
            'Logit_Ant': all_logits[:, 1],
            'Logit_Ago': all_logits[:, 2],
        })
        return bacc, results_df
    else:
        return bacc

def collate_fn(data_list):
    valid_data = [item for item in data_list if item[0] is not None]
    if not valid_data: return None, None
    p, l = zip(*valid_data)
    return Batch.from_data_list(p), Batch.from_data_list(l)

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Unified-EGNN Benchmark")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CSV files")
    parser.add_argument("--protein_graph_dir", type=str, required=True, help="Path to protein graphs")
    parser.add_argument("--ligand_graph_dir", type=str, required=True, help="Path to ligand graphs")
    parser.add_argument("--output_dir", type=str, default="./results/benchmark/unified_egnn", help="Output directory")
    args = parser.parse_args()

    # Config hardcoded for benchmark reproducibility
    CONFIG = {
        "SEED": 0,
        "HIDDEN_DIM": 128,
        "PROTEIN_LAYERS": 4, "LIGAND_LAYERS": 4,
        "PROTEIN_TYPE": "gated_residual", "LIGAND_TYPE": "gated_residual",
        "ELEMENT_EMBEDDING_DIM": 8, "DROPOUT": 0.4,
        "LR": 1e-4, "WEIGHT_DECAY": 1e-5, "BATCH_SIZE": 16,
        "EPOCHS": 100, "PATIENCE": 10
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG["SEED"])
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    train_df = pd.read_csv(Path(args.data_dir) / "scaffold_train.csv")
    valid_df = pd.read_csv(Path(args.data_dir) / "scaffold_val.csv")
    test_df = pd.read_csv(Path(args.data_dir) / "scaffold_test.csv")
    
    train_df['Final_Label'] = 0
    train_df.loc[train_df['Activity'] == 0, 'Final_Label'] = 1
    train_df.loc[train_df['Activity'] == 1, 'Final_Label'] = 2

    valid_df['Final_Label'] = 0
    valid_df.loc[valid_df['Activity'] == 0, 'Final_Label'] = 1
    valid_df.loc[valid_df['Activity'] == 1, 'Final_Label'] = 2
    
    # Data Loaders
    dataset_kwargs = {'protein_graph_dir': args.protein_graph_dir, 'ligand_graph_dir': args.ligand_graph_dir}
    train_dataset = GraphDataset(root=str(output_path / 'cache_train'), df=train_df, **dataset_kwargs)
    valid_dataset = GraphDataset(root=str(output_path / 'cache_valid'), df=valid_df, **dataset_kwargs)
    test_dataset = GraphDataset(root=str(output_path / 'cache_test'), df=test_df, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Initialize Model
    # Note: Using dummy input size calculation or hardcoding based on known dimensions
    # Ideally get from first batch
    sample_p, sample_l = train_dataset[0]
    if sample_p is None: raise ValueError("Invalid first sample")
    
    protein_dim = sample_p.x_float_full.shape[1] + CONFIG['ELEMENT_EMBEDDING_DIM']
    ligand_dim = sample_l.x.shape[1]

    model = UnifiedEGNN(
        protein_in_dim_full=protein_dim,
        ligand_in_dim=ligand_dim,
        hidden_dim=CONFIG['HIDDEN_DIM'],
        protein_config={"type": CONFIG['PROTEIN_TYPE'], "n_layers": CONFIG['PROTEIN_LAYERS']},
        ligand_config={"type": CONFIG['LIGAND_TYPE'], "n_layers": CONFIG['LIGAND_LAYERS']},
        element_embedding_dim=CONFIG['ELEMENT_EMBEDDING_DIM'],
        dropout=CONFIG['DROPOUT']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=CONFIG['PATIENCE'], verbose=True, path=output_path / "model.pt", mode='max')

    print("Starting training...")
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=32)
        val_bacc = evaluate(model, valid_loader, device)
        
        print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val BACC: {val_bacc:.4f}")
        
        early_stopper(val_bacc, model)
        if early_stopper.early_stop:
            print("Early stopping.")
            break

    # Final Test
    model.load_state_dict(torch.load(output_path / "model.pt"))
    test_bacc, test_df = evaluate(model, test_loader, device, return_df=True)
    test_df.to_csv(output_path / "predictions_test.csv", index=False)
    print(f"Test Set BACC: {test_bacc:.4f}")

if __name__ == "__main__":
    main()