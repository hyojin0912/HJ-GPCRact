# scripts/train.py
import sys
import os
from pathlib import Path
import yaml
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch import amp

from src.utils import set_seed, EarlyStopping
from src.data_loader import GraphDataset, get_valid_indices, collate_fn
from src.model import DAGN_HybridModel

def train_epoch(model, loader, criterion_activity_type, criterion_binding, optimizer, device, accumulation_steps=4, lambda_activity=1.0, w_fn=2.0):
    model.train()
    total_loss, total_act_loss, total_bind_loss = 0, 0, 0
    scaler = amp.GradScaler("cuda")
    optimizer.zero_grad()

    for i, data in enumerate(tqdm(loader, desc="Training", leave=False)):
        if data[0] is None: continue
        protein_batch, ligand_batch = data
        protein_batch, ligand_batch = protein_batch.to(device), ligand_batch.to(device)
        
        binding_labels = protein_batch.binding_label.squeeze(-1)
        activity_labels = protein_batch.activity_label.squeeze(-1)

        with amp.autocast("cuda"):
            binding_logit, activity_type_logit, _ = model(protein_batch, ligand_batch)
            
            # --- 1. Binding Loss (Binary: Binder vs. Non-binder) ---
            binding_logit = binding_logit.squeeze(-1)
            loss_bind = criterion_binding(binding_logit.float(), binding_labels.float())

            # --- 2. Activity Type Loss (Binary: Antagonist vs. Agonist) ---
            # This loss is ONLY calculated for true binders.
            loss_activity = torch.tensor(0.0, device=device)
            
            # Use the robust mask we developed before
            true_binder_mask = (binding_labels == 1) & (activity_labels != -1.0)

            if true_binder_mask.sum() > 0:
                activity_type_logits_filtered = activity_type_logit[true_binder_mask]
                activity_type_labels_filtered = activity_labels[true_binder_mask].long()
                
                per_sample_loss_activity = criterion_activity_type(activity_type_logits_filtered, activity_type_labels_filtered)
                
                with torch.no_grad():
                    binding_logit_filtered = binding_logit[true_binder_mask]
                    binding_labels_filtered = binding_labels[true_binder_mask]
                    binding_prob_filtered = torch.sigmoid(binding_logit_filtered)
                    predicted_binders_filtered = (binding_prob_filtered >= 0.5)
                    is_fn = ~predicted_binders_filtered & (binding_labels_filtered == 1)

                # <<< 최종 수정된 가중치 적용 로직 >>>
                weights = torch.ones_like(per_sample_loss_activity)
                # is_fn 마스크가 True인 위치에만 w_fn 값을 할당.
                # per_sample_loss가 스칼라(샘플 1개)이고 is_fn이 True일 경우에도 올바르게 작동함.
                weights[is_fn] = w_fn
                loss_activity = (per_sample_loss_activity * weights).mean()

            # --- 3. Total Loss ---
            total_loss_step = loss_bind + lambda_activity * loss_activity
            loss_to_accumulate = total_loss_step / accumulation_steps
        
        scaler.scale(loss_to_accumulate).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += total_loss_step.item()
        total_act_loss += loss_activity.item()
        total_bind_loss += loss_bind.item()
            
    return total_loss / len(loader), total_act_loss / len(loader), total_bind_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, return_df=False):
    model.eval()
    all_final_preds, all_final_labels = [], []
    all_binding_preds, all_binding_labels_raw = [], []
    all_activity_type_logits = []
    all_ikeys, all_uniprots = [], []

    for data in tqdm(loader, desc="Evaluating", leave=False):
        if data is None or data[0] is None: continue
        protein_batch, ligand_batch = data
        protein_batch, ligand_batch = protein_batch.to(device), ligand_batch.to(device)

        all_ikeys.extend(protein_batch.ikey)
        all_uniprots.extend(protein_batch.uniprot_id)

        binding_labels = protein_batch.binding_label.squeeze(-1).cpu()
        activity_labels = protein_batch.activity_label.squeeze(-1).cpu()

        binding_logit, activity_type_logit, _ = model(protein_batch, ligand_batch)
        
        # --- 1. Store Binding Prediction Info ---
        binding_prob = torch.sigmoid(binding_logit).squeeze(-1).cpu()
        predicted_as_binder = (binding_prob >= 0.5).long()
        all_binding_preds.append(binding_prob)
        all_binding_labels_raw.append(binding_labels)
        
        # --- 2. Store Activity Type Prediction Info ---
        predicted_activity_type = torch.argmax(activity_type_logit, dim=1).cpu() # 0=Ant, 1=Ago
        all_activity_type_logits.append(activity_type_logit.cpu())

        # --- 3. Combine for Final 3-Class Prediction ---
        final_preds = torch.zeros_like(predicted_as_binder)
        final_preds[predicted_as_binder == 1] = predicted_activity_type[predicted_as_binder == 1] + 1
        all_final_preds.append(final_preds)

        # --- 4. Create Final 3-Class Ground Truth ---
        final_targets = torch.zeros_like(binding_labels, dtype=torch.long)
        # Non-binders are class 0 (already initialized)
        final_targets[activity_labels == 0] = 1 # Antagonists (activity_label=0) are class 1
        final_targets[activity_labels == 1] = 2 # Agonists (activity_label=1) are class 2
        all_final_labels.append(final_targets)

    all_binding_preds = torch.cat(all_binding_preds).numpy()
    all_binding_labels_raw = torch.cat(all_binding_labels_raw).numpy()
    all_final_preds = torch.cat(all_final_preds).numpy()
    all_final_labels = torch.cat(all_final_labels).numpy()
    all_activity_type_logits = torch.cat(all_activity_type_logits).numpy()
    
    # 1. Primary Metric: Final 3-Class Balanced Accuracy
    final_3class_bacc = balanced_accuracy_score(all_final_labels, all_final_preds)
    # 2. Auxiliary Metric: Binding Prediction BAcc
    bind_bacc = balanced_accuracy_score(all_binding_labels_raw, all_binding_preds > 0.5)

    if return_df:
        results_df = pd.DataFrame({
            'Ikey': all_ikeys, 'UniProt': all_uniprots,
            'Binding_Label': all_binding_labels_raw, 'Binding_Prob': all_binding_preds,
            'Final_Label': all_final_labels, # 0=NB, 1=Ant, 2=Ago
            'Final_Pred': all_final_preds,   # 0=NB, 1=Ant, 2=Ago
            'Logit_Antagonist': all_activity_type_logits[:, 0],
            'Logit_Agonist': all_activity_type_logits[:, 1],
        })
        return final_3class_bacc, bind_bacc, results_df
    else:
        return final_3class_bacc, bind_bacc

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    print("--- Starting Hierarchical Head Model Training ---")
    
    set_seed(CONFIG["SEED"])
    total_epochs = CONFIG["EPOCHS"]

    CONFIG["CACHE_DIR"].mkdir(parents=True, exist_ok=True)
    CONFIG["RESULTS_DIR"].mkdir(parents=True, exist_ok=True)
    CONFIG["MODEL_SAVE_DIR"].mkdir(parents=True, exist_ok=True)
    
    # --- Load and Prepare Data ---
    print("--- Loading and preparing datasets ---")
    train_full_df = pd.read_csv(CONFIG["DATA_DIR"] / "train_set_scaf.csv")
    test_df = pd.read_csv(CONFIG["DATA_DIR"] / "test_set_scaf.csv")

    # Filter the training and test sets
    train_valid_mask = get_valid_indices(train_full_df, CONFIG['PROTEIN_GRAPH_DIR'], CONFIG['LIGAND_GRAPH_DIR'])
    test_valid_mask = get_valid_indices(test_df, CONFIG['PROTEIN_GRAPH_DIR'], CONFIG['LIGAND_GRAPH_DIR'])

    print(f"Original train set size: {len(train_full_df)}")
    print(f"Original test set size: {len(test_df)}")
    
    train_full_df = train_full_df[train_valid_mask].reset_index(drop=True)
    test_df = test_df[test_valid_mask].reset_index(drop=True)

    print(f"Filtered train set size: {len(train_full_df)}")
    print(f"Filtered test set size: {len(test_df)}")

    #### Train - Test Split
    train_df, valid_df = train_test_split(train_full_df, test_size=0.2, random_state=CONFIG["SEED"], stratify=train_full_df['Binding'])
    
    # Perform undersampling on the training set for the binding task
    nonbinder_df = train_df[train_df['Binding'] == 0]
    binder_df = train_df[train_df['Binding'] == 1]
    
    # Ensure we don't sample more binders than available if nonbinders are more numerous
    num_binders_to_sample = min(len(nonbinder_df), len(binder_df))
    
    train_df_undersampled = pd.concat([
        nonbinder_df,
        binder_df.sample(n=num_binders_to_sample, random_state=CONFIG["SEED"])
    ]).sample(frac=1, random_state=CONFIG["SEED"])

    print(f"Full Train Set: {len(train_df)}")
    print(f"Undersampled Train Set for training: {len(train_df_undersampled)}")
    print(f"Validation Set: {len(valid_df)}")
    print(f"Test Set: {len(test_df)}")

    # --- Create Datasets and DataLoaders ---
    dataset_kwargs = {
        'protein_graph_dir': CONFIG['PROTEIN_GRAPH_DIR'],
        'ligand_graph_dir': CONFIG['LIGAND_GRAPH_DIR'],
    }
    
    train_dataset = GraphDataset(root=str(CONFIG['CACHE_DIR'] / 'train'), df=train_df_undersampled, **dataset_kwargs)
    valid_dataset = GraphDataset(root=str(CONFIG['CACHE_DIR'] / 'valid'), df=valid_df, **dataset_kwargs)
    test_dataset = GraphDataset(root=str(CONFIG['CACHE_DIR'] / 'test'), df=test_df, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['GPU_BATCH_SIZE'], shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['GPU_BATCH_SIZE'], shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['GPU_BATCH_SIZE'], shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    # --- Initialize Model and Training Components ---
    p_sample, l_sample = next(item for item in train_dataset if item[0] is not None)
    protein_input_dim_clean = p_sample.x_float_clean.shape[1] + CONFIG['ELEMENT_EMBEDDING_DIM']
    protein_input_dim_full = p_sample.x_float_full.shape[1] + CONFIG['ELEMENT_EMBEDDING_DIM']

    model = DAGN_HybridModel(
        protein_in_dim_clean=protein_input_dim_clean,
        protein_in_dim_full=protein_input_dim_full,
        ligand_in_dim=l_sample.x.shape[1],
        hidden_dim=CONFIG['HIDDEN_DIM'],
        protein_config={"type": CONFIG['PROTEIN_TYPE'], "n_layers": CONFIG['PROTEIN_LAYERS']},
        ligand_config={"type": CONFIG['LIGAND_TYPE'], "n_layers": CONFIG['LIGAND_LAYERS']},
        element_embedding_dim=CONFIG['ELEMENT_EMBEDDING_DIM'],
        dropout=CONFIG['DROPOUT'],
        n_attn_heads=CONFIG['ATTENTION_HEADS'],
        propagation_attention_layers=CONFIG['PROPAGATION_ATTENTION_LAYERS']
    ).to(CONFIG['DEVICE'])

    # Binary classification for binding (Binder vs Non-binder)
    criterion_binding = nn.BCEWithLogitsLoss()
    # 2-Class classification for activity type (Antagonist vs Agonist)
    criterion_activity_type = nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    
    model_save_path = CONFIG["MODEL_SAVE_DIR"] / "dagn_hierarchical_head.pt"
    early_stopper = EarlyStopping(patience=CONFIG['EARLY_STOPPING_PATIENCE'], verbose=True, path=model_save_path, mode='max')
    
    print("DAGN Hierarchical Head Model initialized. Starting training...")
    
    # --- Training Loop ---
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        print(f"\nEpoch {epoch:03d}/{total_epochs}")

        train_loss, train_act_loss, train_bind_loss = train_epoch(
            model, train_loader, 
            criterion_activity_type,
            criterion_binding, 
            optimizer, CONFIG['DEVICE'],
            accumulation_steps=CONFIG['ACCUMULATION_STEPS'],
            lambda_activity=1.0,
            w_fn=CONFIG['W_FN_BIND']
        )
                
        # Get modified evaluation results
        val_3class_bacc, val_bind_bacc = evaluate(model, valid_loader, CONFIG['DEVICE'])
        
        print(f"gating_lossweight Epoch {epoch:03d} | Train Loss: {train_loss:.4f} (Act: {train_act_loss:.4f}, Bind: {train_bind_loss:.4f}) | "
              f"Val 3-Class BACC: {val_3class_bacc:.4f} | Val Bind BACC: {val_bind_bacc:.4f}")
        
        early_stopper(val_3class_bacc, model) # Monitor 3-class BACC
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
            
    # --- Final Evaluation and Prediction Saving ---
    print("\n--- Training finished. Loading best model for final evaluation. ---")
    model.load_state_dict(torch.load(model_save_path))

    print("Evaluating on Training Set...")
    train_3class_bacc, train_bind_bacc, train_preds_df = evaluate(model, train_loader, CONFIG['DEVICE'], return_df=True)
    train_preds_df.to_csv(CONFIG["RESULTS_DIR"] / "predictions_train.csv", index=False)

    print("Evaluating on Validation Set...")
    val_3class_bacc, val_bind_bacc, valid_preds_df = evaluate(model, valid_loader, CONFIG['DEVICE'], return_df=True)
    valid_preds_df.to_csv(CONFIG["RESULTS_DIR"] / "predictions_valid.csv", index=False)

    print("Evaluating on Test Set...")
    test_3class_bacc, test_bind_bacc, test_preds_df = evaluate(model, test_loader, CONFIG['DEVICE'], return_df=True)
    test_preds_df.to_csv(CONFIG["RESULTS_DIR"] / "predictions_test.csv", index=False)
        
    print("\n--- Final Performance Summary ---")
    print(f"Train Set: 3-Class BACC={train_3class_bacc:.4f}, Bind BACC={train_bind_bacc:.4f}")
    print(f"Valid Set: 3-Class BACC={val_3class_bacc:.4f}, Bind BACC={val_bind_bacc:.4f}")
    print(f"Test Set:  3-Class BACC={test_3class_bacc:.4f}, Bind BACC={test_bind_bacc:.4f}")
    print("-----------------------------------")
    print(f"Best model saved to: {model_save_path}")
    print(f"Prediction CSVs saved in: {CONFIG['RESULTS_DIR']}")

if __name__ == '__main__':

    main()
