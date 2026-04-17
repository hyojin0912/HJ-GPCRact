import sys
import os
import wandb
import torch
import torch.nn as nn
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPCRact_Model
from src.dataset import GraphDataset, collate_fn, get_valid_indices
from src.utils import set_seed, EarlyStopping
from scripts.train import train_epoch, evaluate

def run_trial():
    run = wandb.init()
    cfg = wandb.config
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NOTE: You must adjust these paths to match your local environment for reproduction
    # Since we can't upload graph data, this script serves as a reference for the method.
    DATA_DIR = Path("../data/splits") 
    # HPO often uses different k-NN graphs. Here we assume paths are set via environment or fixed structure.
    # For GitHub release, we can use a placeholder or assume user generated graphs.
    P_GRAPH_DIR = Path(f"../data/processed/protein_graphs_k{cfg.k_nn}") 
    L_GRAPH_DIR = Path("../data/processed/ligand_graphs")
    
    # Check if data exists (Graceful exit if running as demo)
    if not P_GRAPH_DIR.exists():
        print(f"Graph directory {P_GRAPH_DIR} not found. Skipping trial.")
        return

    # Load Data (Using a subset for speed in HPO)
    full_df = pd.read_csv(DATA_DIR / "scaffold_train.csv")
    # Stratify using the raw Label column (binder vs non-binder).
    full_df['_strat'] = (full_df['Label'].astype(str).str.strip().str.lower() != 'nonbinder').astype(int)
    _, subset_df = train_test_split(full_df, test_size=0.2, random_state=42,
                                    stratify=full_df['_strat'])
    train_df, val_df = train_test_split(subset_df, test_size=0.2, random_state=42,
                                        stratify=subset_df['_strat'])
    # Drop helper column before passing to dataset.
    train_df = train_df.drop(columns=['_strat']).reset_index(drop=True)
    val_df   = val_df.drop(columns=['_strat']).reset_index(drop=True)
    
    # Filter Valid
    train_idx = get_valid_indices(train_df, P_GRAPH_DIR, L_GRAPH_DIR)
    train_df = train_df[train_idx].reset_index(drop=True)
    val_idx = get_valid_indices(val_df, P_GRAPH_DIR, L_GRAPH_DIR)
    val_df = val_df[val_idx].reset_index(drop=True)
    
    train_ds = GraphDataset(root=None, df=train_df, protein_graph_dir=P_GRAPH_DIR, ligand_graph_dir=L_GRAPH_DIR)
    val_ds = GraphDataset(root=None, df=val_df, protein_graph_dir=P_GRAPH_DIR, ligand_graph_dir=L_GRAPH_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Model Setup
    p_s, l_s = train_ds[0]
    p_dim_clean = p_s.x_float_clean.shape[1] + 8
    p_dim_full = p_s.x_float_full.shape[1] + 8
    l_dim = l_s.x.shape[1]

    use_cf = True
    num_classes, num_families = 0, 0

    class_json = P_GRAPH_DIR / "class_to_id.json"
    family_json = P_GRAPH_DIR / "family_to_id.json"
    if (not class_json.exists()) or (not family_json.exists()):
        use_cf = False
    else:
        with open(class_json, "r") as f:
            num_classes = len(json.load(f))
        with open(family_json, "r") as f:
            num_families = len(json.load(f))
    
    model = GPCRact_Model(
        protein_in_dim_clean=p_dim_clean,
        protein_in_dim_full=p_dim_full,
        ligand_in_dim=l_dim,
        hidden_dim=cfg.hidden_dim,
        protein_config={"type": "gated_residual", "n_layers": 4},
        ligand_config={"type": "gated_residual", "n_layers": 4},
        element_embedding_dim=8,
        n_attn_heads=cfg.attention_heads,
        dropout=cfg.dropout,
        propagation_attention_layers=cfg.propagation_attn_layers,
        use_gpcr_cf_embed=use_cf,
        num_gpcr_classes=num_classes,
        num_gpcr_families=num_families,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    crit_bind = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], device=device))
    crit_act  = nn.CrossEntropyLoss(reduction='none')
    
    # Train Loop
    for epoch in range(1, cfg.epochs + 1):
        # No activity-class weighting for HPO sweeps (keeps the search space smaller).
        loss, _, _ = train_epoch(model, train_loader, crit_act, crit_bind, optimizer, device,
                                 accum_steps=32, lambda_act=1.0, w_fn=None)
        
        val_bacc, _ = evaluate(model, val_loader, device)
        
        wandb.log({"val_bacc": val_bacc, "train_loss": loss})
        print(f"Epoch {epoch} | Val BAcc: {val_bacc:.4f}")

if __name__ == "__main__":
    # Typically run via 'wandb agent <sweep_id>'
    # This block allows running a single run for debugging
    if os.environ.get("WANDB_SWEEP_ID"):
        run_trial()
    else:

        print("Please run this script via wandb agent or initialize a sweep.")
