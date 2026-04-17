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
from sklearn.metrics import balanced_accuracy_score

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPCRact_Model
from src.dataset import GraphDataset, collate_fn, get_valid_indices
from src.utils import set_seed, EarlyStopping

# Tqdm needs to be imported here to work in script
from tqdm import tqdm

def train_epoch(model, loader, criterion_act, criterion_bind, optimizer, device, accum_steps, lambda_act, w_fn):
    model.train()
    total_loss, total_act_loss, total_bind_loss = 0, 0, 0
    
    # Device-agnostic GradScaler: only enable AMP scaling on CUDA.
    is_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=is_cuda)
    optimizer.zero_grad()
    
    valid_batches = 0
    
    for i, data in enumerate(tqdm(loader, desc="Training", leave=False)):
        # collate_fn may return None itself when the whole mini-batch is invalid.
        if data is None or data[0] is None: continue
        protein_batch, ligand_batch = data
        protein_batch, ligand_batch = protein_batch.to(device), ligand_batch.to(device)
        
        # Flatten label tensors from [B, 1] -> [B] so they align with logits.
        binding_labels = protein_batch.binding_label.squeeze(-1)
        activity_labels = protein_batch.activity_label.long().squeeze(-1)

        # Device-agnostic autocast: mixed precision only on CUDA.
        with torch.autocast(device_type=device.type, enabled=is_cuda):
            # Model returns (binding_logit, activity_type_logit). See src/model.py.
            binding_logit, logits_act = model(protein_batch, ligand_batch)
            logits_bind = binding_logit.squeeze(-1)  # [B, 1] -> [B]
                
            loss_act, loss_bind = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            
            valid_bind_mask = binding_labels != -1.0
            if valid_bind_mask.any():
                loss_bind = criterion_bind(logits_bind[valid_bind_mask], binding_labels[valid_bind_mask])
                
            valid_act_mask = activity_labels != -1
            if valid_act_mask.any():
                loss_act = criterion_act(logits_act[valid_act_mask], activity_labels[valid_act_mask])
                if w_fn is not None:
                    weights = w_fn(activity_labels[valid_act_mask]).to(device)
                    loss_act = (loss_act * weights).mean()
                else:
                    loss_act = loss_act.mean()
                    
            # Independent multi-task weighting (see Supplementary Table S5):
            # lambda_act scales the Stage-2 activity loss.
            # Stage-1 binding loss keeps unit weight; its class imbalance is handled via pos_weight in the BCE criterion (FN penalty = 1.5).
            loss = lambda_act * loss_act + loss_bind
            loss = loss / accum_steps
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        total_act_loss += loss_act.item() if isinstance(loss_act, torch.Tensor) else loss_act
        total_bind_loss += loss_bind.item() if isinstance(loss_bind, torch.Tensor) else loss_bind
        valid_batches += 1
        
    return total_loss / valid_batches, total_act_loss / valid_batches, total_bind_loss / valid_batches

def evaluate(model, loader, device, return_df=False):
    model.eval()
    all_act_preds, all_act_labels = [], []
    all_bind_preds, all_bind_labels = [], []
    results = []
    
    # Check if device is CUDA for autocast
    is_cuda = device.type == 'cuda'
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            if data is None or data[0] is None: continue
            protein_batch, ligand_batch = data
            protein_batch, ligand_batch = protein_batch.to(device), ligand_batch.to(device)
            
            with torch.autocast(device_type=device.type, enabled=is_cuda):
                binding_logit, logits_act = model(protein_batch, ligand_batch)
                logits_bind = binding_logit.squeeze(-1)  # [B, 1] -> [B]
            
            probs_act = torch.softmax(logits_act, dim=-1)
            preds_act = torch.argmax(probs_act, dim=-1)
            probs_bind = torch.sigmoid(logits_bind)
            
            act_labels = protein_batch.activity_label.squeeze(-1).cpu().numpy()
            bind_labels = protein_batch.binding_label.squeeze(-1).cpu().numpy()
            preds_act_np = preds_act.cpu().numpy()
            probs_bind_np = probs_bind.cpu().numpy()
            
            ikeys = protein_batch.ikey
            uniprots = protein_batch.uniprot_id
            
            for i in range(len(act_labels)):
                if act_labels[i] != -1:
                    all_act_preds.append(preds_act_np[i])
                    all_act_labels.append(act_labels[i])
                if bind_labels[i] != -1.0:
                    all_bind_preds.append(1 if float(probs_bind_np[i]) > 0.5 else 0)
                    all_bind_labels.append(bind_labels[i])

                if return_df:
                    # Stage 1: probability of being a binder.
                    # Stage 2: class 0 = antagonist, class 1 = agonist (binder-only).
                    results.append({
                        'Ikey': ikeys[i],
                        'UniProt': uniprots[i],
                        'Binding_Prob': float(probs_bind_np[i]),
                        'Activity_Pred': int(preds_act_np[i]),
                        'Logit_Antagonist': float(logits_act[i, 0].item()),
                        'Logit_Agonist':    float(logits_act[i, 1].item()),
                    })

    bacc = balanced_accuracy_score(all_act_labels, all_act_preds) if all_act_labels else 0.0
    bind_bacc = balanced_accuracy_score(all_bind_labels, all_bind_preds) if all_bind_labels else 0.0
    
    if return_df:
        return bacc, bind_bacc, pd.DataFrame(results)
    return bacc, bind_bacc

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data. Custom CSV paths override the data_dir defaults.
    print("Loading data splits...")
    train_csv_path = Path(args.train_csv) if args.train_csv else Path(args.data_dir) / "scaffold_train.csv"
    val_csv_path   = Path(args.val_csv)   if args.val_csv   else Path(args.data_dir) / "scaffold_val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df   = pd.read_csv(val_csv_path)
    print(f"  Train CSV: {train_csv_path} ({len(train_df)} rows)")
    print(f"  Val CSV:   {val_csv_path} ({len(val_df)} rows)")
    
    train_valid_idx = get_valid_indices(train_df, args.protein_graph_dir, args.ligand_graph_dir)
    val_valid_idx = get_valid_indices(val_df, args.protein_graph_dir, args.ligand_graph_dir)
    
    train_df = train_df[train_valid_idx].reset_index(drop=True)
    val_df = val_df[val_valid_idx].reset_index(drop=True)
    
    train_ds = GraphDataset(root=None, df=train_df, protein_graph_dir=args.protein_graph_dir, ligand_graph_dir=args.ligand_graph_dir)
    val_ds = GraphDataset(root=None, df=val_df, protein_graph_dir=args.protein_graph_dir, ligand_graph_dir=args.ligand_graph_dir)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"No valid samples found after filtering. "
            f"Each CSV row requires its protein graph "
            f"({args.protein_graph_dir}/<AC>.pt) and ligand graph "
            f"({args.ligand_graph_dir}/<Ikey>.pt) to exist on disk."
        )

    # 2. Extract Dimensions
    
    # 2. Extract Dimensions
    p_s, l_s = train_ds[0]
    p_dim_clean = p_s.x_float_clean.shape[1] + args.elem_emb_dim
    p_dim_full = p_s.x_float_full.shape[1] + args.elem_emb_dim
    l_dim = l_s.x.shape[1]
           
    class_dict_path = Path(args.protein_graph_dir) / "class_to_id.json"
    family_dict_path = Path(args.protein_graph_dir) / "family_to_id.json"
    
    with open(class_dict_path, "r") as f: num_classes = len(json.load(f))
    with open(family_dict_path, "r") as f: num_families = len(json.load(f))
    
    # 3. Initialize Model
    model = GPCRact_Model(
        protein_in_dim_clean=p_dim_clean,
        protein_in_dim_full=p_dim_full,
        ligand_in_dim=l_dim,
        hidden_dim=args.hidden_dim,
        protein_config={"type": "gated_residual", "n_layers": args.enc_layers},
        ligand_config={"type": "gated_residual", "n_layers": args.enc_layers},
        element_embedding_dim=args.elem_emb_dim,
        n_attn_heads=args.attn_heads,
        dropout=args.dropout,
        propagation_attention_layers=args.prop_attn_layers,
        use_gpcr_cf_embed=args.use_gpcr_cf_embed,
        num_gpcr_classes=num_classes,
        num_gpcr_families=num_families
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Compute class weights for Stage-2 (activity) loss from the raw Label column.
    # Only binder-labeled rows contribute; LABEL_MAP: antagonist=0, agonist=1.
    label_to_act = {k: v[1] for k, v in GraphDataset.LABEL_MAP.items() if v[1] >= 0}
    act_series = train_df['Label'].astype(str).str.strip().str.lower().map(label_to_act)
    act_labels_np = act_series.dropna().astype(int).values
    class_counts = np.bincount(act_labels_np, minlength=2)
    total_samples = len(act_labels_np)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    def get_class_weights(labels):
        return class_weights_tensor[labels]
        
    criterion_act  = nn.CrossEntropyLoss(reduction='none')
    # Binding false-negative penalty = 1.5 (Supplementary Table S5):
    # pos_weight up-weights the positive class (binder=1) in BCE-with-logits.
    criterion_bind = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.binding_fn_penalty], device=device)
    )
    
    early_stopping = EarlyStopping(patience=15, verbose=True, path=os.path.join(args.save_dir, 'best_model.pt'), mode='max')
    
    # 4. Training Loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        t_loss, a_loss, b_loss = train_epoch(model, train_loader, criterion_act, criterion_bind, optimizer, device, args.accum_steps, args.lambda_act, get_class_weights)
        val_bacc, val_bind_bacc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:03d} | Train Loss: {t_loss:.4f} | Val Act BACC: {val_bacc:.4f} | Val Bind BACC: {val_bind_bacc:.4f}")
        
        early_stopping(val_bacc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    print("Training complete. Best model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/splits",
                        help="Default location for scaffold_train.csv and scaffold_val.csv (used when --train_csv / --val_csv are not provided).")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to a custom training CSV. Overrides --data_dir.")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to a custom validation CSV. Overrides --data_dir.")
    parser.add_argument("--protein_graph_dir", type=str, required=True)
    parser.add_argument("--ligand_graph_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save model")
    parser.add_argument("--use_gpcr_cf_embed", action=argparse.BooleanOptionalAction, default=True)
    
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
    parser.add_argument("--lambda_act", type=float, default=1.0,
                        help="Weight for Stage-2 activity loss (Table S5: 1.0).")
    parser.add_argument("--binding_fn_penalty", type=float, default=1.5,
                        help="BCE pos_weight for Stage-1 binding loss (Table S5: 1.5).")
    
    args = parser.parse_args()
    main(args)
