import sys
import os
import argparse
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPCRact_Model
from src.dataset import GraphDataset, collate_fn, get_valid_indices
from scripts.train import evaluate # Reuse evaluate function

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Query Data (Modified to use custom query_csv argument)
    csv_path = Path(args.data_dir) / args.query_csv
    test_df = pd.read_csv(csv_path)
    
    valid_idx = get_valid_indices(test_df, args.protein_graph_dir, args.ligand_graph_dir)
    test_df = test_df[valid_idx].reset_index(drop=True)
    
    test_ds = GraphDataset(root=None, df=test_df, 
                           protein_graph_dir=args.protein_graph_dir, 
                           ligand_graph_dir=args.ligand_graph_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Initialize Model (Must match training config)
    # Note: Ideally, load config from a file saved during training. 
    # Here we assume arguments provided match training.
    p_s, l_s = test_ds[0]
    p_dim_clean = p_s.x_float_clean.shape[1] + args.elem_emb_dim
    p_dim_full = p_s.x_float_full.shape[1] + args.elem_emb_dim
    l_dim = l_s.x.shape[1]

    class_dict_path = Path(args.protein_graph_dir) / "class_to_id.json"
    family_dict_path = Path(args.protein_graph_dir) / "family_to_id.json"
    
    with open(class_dict_path, "r") as f:
        num_classes = len(json.load(f))
    with open(family_dict_path, "r") as f:
        num_families = len(json.load(f))
    
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
        num_classes=num_classes,
        num_families=num_families
    ).to(device)
    
    # Load Weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate
    bacc, bind_bacc, results_df = evaluate(model, test_loader, device, return_df=True)
    
    print(f"Test Balanced Accuracy: {bacc:.4f}")
    print(f"Test Binding Balanced Accuracy: {bind_bacc:.4f}")
    
    # Dynamically set output filename based on input query_csv
    input_stem = Path(args.query_csv).stem
    save_path = Path(args.output_dir) / f"predictions_{input_stem}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/splits")
    # Added argument to handle custom user-provided CSV files
    parser.add_argument("--query_csv", type=str, default="scaffold_test.csv", help="Filename of the query CSV located inside data_dir")
    
    parser.add_argument("--protein_graph_dir", type=str, required=True)
    parser.add_argument("--ligand_graph_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt file")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--use_gpcr_cf_embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpcr_meta_dir", type=str, default=None)
    
    # Model Hparams (Must match training)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--prop_attn_layers", type=int, default=3)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--elem_emb_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
