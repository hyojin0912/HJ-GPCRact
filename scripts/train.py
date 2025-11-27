import sys
import os
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import amp

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DAGN_HybridModel
from src.dataset import GraphDataset, collate_fn, get_valid_indices
from src.utils import set_seed, EarlyStopping

# Tqdm needs to be imported here to work in script
from tqdm import tqdm

def train_epoch(model, loader, criterion_act, criterion_bind, optimizer, device, accum_steps, lambda_act, w_fn):
    model.train()
    total_loss, total_act_loss, total_bind_loss = 0, 0, 0
    scaler = amp.GradScaler("cuda")
    optimizer.zero_grad()
    
    valid_batches = 0
    
    for i, data in enumerate(tqdm(loader, desc="Training", leave=False)):
        if data[0] is None: continue
        protein_batch, ligand_batch = data
        protein_batch, ligand_batch = protein_batch.to(device), ligand_batch.to(device)
        
        binding_labels = protein_batch.binding_label.squeeze(-1)
        activity_labels = protein_batch.activity_label.squeeze(-1)

        try:
            with amp.autocast("cuda"):
                bind_logit, act_logit = model(protein_batch, ligand_batch)
                
                # 1. Binding Loss
                bind_logit = bind_logit.squeeze(-1)
                loss_bind = criterion_bind(bind_logit.float(), binding_labels.float())
                
                # 2. Activity Loss (Conditional)
                loss_act = torch.tensor(0.0, device=device)
                true_binder_mask = (binding_labels == 1) & (activity_labels != -1.0)
                
                if true_binder_mask.sum() > 0:
                    act_logits_filt = act_logit[true_binder_mask]
                    act_labels_filt = activity_labels[true_binder_mask].long()
                    
                    per_sample_loss = criterion_act(act_logits_filt, act_labels_filt)
                    
                    with torch.no_grad():
                        bind_prob = torch.sigmoid(bind_logit[true_binder_mask])
                        is_fn = ~(bind_prob >= 0.5) & (binding_labels[true_binder_mask] == 1)
                    
                    weights = torch.ones_like(per_sample_loss)
                    weights[is_fn] = w_fn
                    loss_act = (per_sample_loss * weights).mean()
                
                # 3. Total Loss
                loss = loss_bind + lambda_act * loss_act
                loss_accum = loss / accum_steps
            
            scaler.scale(loss_accum).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item()
            total_act_loss += loss_act.item()
            total_bind_loss += loss_bind.item()
            valid_batches += 1
            
        except RuntimeError as e:
            if "nan" in str(e).lower():
                print(f"NaN detected at batch {i}. Skipping.")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    if valid_batches == 0: return 0, 0, 0
    return total_loss / valid_batches, total_act_loss / valid_batches, total_bind_loss / valid_batches

@torch.no_grad()
def evaluate(model, loader, device, return_df=False):
    model.eval()
    all_preds, all_labels = [], []
    all_bind_probs, all_bind_labels = [], []
    ikeys, uniprots = [], []
    
    for data in tqdm(loader, desc="Evaluating", leave=False):
        if data[0] is None: continue
        p_batch, l_batch = data
        p_batch, l_batch = p_batch.to(device), l_batch.to(device)
        
        ikeys.extend(p_batch.ikey)
        uniprots.extend(p_batch.uniprot_id)
        
        bind_labels = p_batch.binding_label.squeeze(-1).cpu()
        act_labels = p_batch.activity_label.squeeze(-1).cpu()
        
        bind_logit, act_logit = model(p_batch, l_batch)
        
        bind_prob = torch.sigmoid(bind_logit).squeeze(-1).cpu()
        pred_binder = (bind_prob >= 0.5).long()
        
        pred_act_type = torch.argmax(act_logit, dim=1).cpu() # 0=Ant, 1=Ago
        
        # Final Pred: 0=NB, 1=Ant, 2=Ago
        final_pred = torch.zeros_like(pred_binder)
        final_pred[pred_binder == 1] = pred_act_type[pred_binder == 1] + 1
        
        # Final Label
        final_target = torch.zeros_like(bind_labels, dtype=torch.long)
        final_target[act_labels == 0] = 1
        final_target[act_labels == 1] = 2
        
        all_preds.append(final_pred)
        all_labels.append(final_target)
        all_bind_probs.append(bind_prob)
        all_bind_labels.append(bind_labels)
        
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_bind_probs = torch.cat(all_bind_probs).numpy()
    all_bind_labels = torch.cat(all_bind_labels).numpy()
    
    from sklearn.metrics import balanced_accuracy_score
    bacc_3class = balanced_accuracy_score(all_labels, all_preds)
    bacc_bind = balanced_accuracy_score(all_bind_labels, all_bind_probs > 0.5)
    
    if return_df:
        return bacc_3class, bacc_bind, pd.DataFrame({
            'Ikey': ikeys, 'UniProt': uniprots,
            'Label': all_labels, 'Prediction': all_preds,
            'Binding_Prob': all_bind_probs
        })
    return bacc_3class, bacc_bind

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path Setup
    DATA_DIR = Path(args.data_dir)
    P_GRAPH_DIR = Path(args.protein_graph_dir)
    L_GRAPH_DIR = Path(args.ligand_graph_dir)
    SAVE_DIR = Path(args.save_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data Loading
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "scaffold_train.csv") # Use splits generated in Step 4
    val_df = pd.read_csv(DATA_DIR / "scaffold_val.csv")
    
    # Filter valid
    train_idx = get_valid_indices(train_df, P_GRAPH_DIR, L_GRAPH_DIR)
    val_idx = get_valid_indices(val_df, P_GRAPH_DIR, L_GRAPH_DIR)
    train_df = train_df[train_idx].reset_index(drop=True)
    val_df = val_df[val_idx].reset_index(drop=True)
    
    # Datasets
    train_ds = GraphDataset(root=None, df=train_df, protein_graph_dir=P_GRAPH_DIR, ligand_graph_dir=L_GRAPH_DIR)
    val_ds = GraphDataset(root=None, df=val_df, protein_graph_dir=P_GRAPH_DIR, ligand_graph_dir=L_GRAPH_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Model Init
    # Infer input dims from a sample
    p_s, l_s = train_ds[0]
    p_dim_clean = p_s.x_float_clean.shape[1] + args.elem_emb_dim
    p_dim_full = p_s.x_float_full.shape[1] + args.elem_emb_dim
    l_dim = l_s.x.shape[1]
    
    model = DAGN_HybridModel(
        protein_in_dim_clean=p_dim_clean,
        protein_in_dim_full=p_dim_full,
        ligand_in_dim=l_dim,
        hidden_dim=args.hidden_dim,
        protein_config={"type": "gated_residual", "n_layers": args.enc_layers},
        ligand_config={"type": "gated_residual", "n_layers": args.enc_layers},
        element_embedding_dim=args.elem_emb_dim,
        n_attn_heads=args.attn_heads,
        dropout=args.dropout,
        propagation_attention_layers=args.prop_attn_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit_bind = nn.BCEWithLogitsLoss()
    crit_act = nn.CrossEntropyLoss(reduction='none')
    
    stopper = EarlyStopping(patience=args.patience, verbose=True, path=SAVE_DIR / "best_model.pt", mode='max')
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, t_act, t_bind = train_epoch(
            model, train_loader, crit_act, crit_bind, optimizer, device,
            args.accum_steps, args.lambda_act, args.w_fn
        )
        
        val_bacc, val_bind_bacc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Val BAcc: {val_bacc:.4f} (Bind: {val_bind_bacc:.4f})")
        
        stopper(val_bacc, model)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break
            
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPCRact")
    parser.add_argument("--data_dir", type=str, default="../data/splits", help="Path to split CSVs")
    parser.add_argument("--protein_graph_dir", type=str, required=True, help="Path to protein .pt files")
    parser.add_argument("--ligand_graph_dir", type=str, required=True, help="Path to ligand .pt files")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save model")
    
    # Hparams
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--prop_attn_layers", type=int, default=3)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--elem_emb_dim", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=32)
    parser.add_argument("--lambda_act", type=float, default=1.0)
    parser.add_argument("--w_fn", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    main(args)