import sys
import os
import argparse
import pandas as pd
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPCRact_Model
from src.dataset import GraphDataset, collate_fn, get_valid_indices
from scripts.train import evaluate # Reuse evaluate function

# Configure logging for industry-grade outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting inference on device: {device.type.upper()}")
    
    # 1. Load Query Data
    csv_path = Path(args.data_dir) / args.query_csv
    logger.info(f"Loading dataset from: {csv_path}")
    
    try:
        test_df = pd.read_csv(csv_path)
        valid_idx = get_valid_indices(test_df, args.protein_graph_dir, args.ligand_graph_dir)
        test_df = test_df.iloc[valid_idx].reset_index(drop=True)
        
        # Guard against empty datasets
        if test_df.empty:
            logger.error("No valid samples found. Please check your data paths and graph directories.")
            return

        test_ds = GraphDataset(root=None, df=test_df, 
                               protein_graph_dir=args.protein_graph_dir, 
                               ligand_graph_dir=args.ligand_graph_dir)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        logger.info(f"Successfully loaded {len(test_ds)} valid samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # 2. Initialize Model
    logger.info("Initializing GPCRact model architecture...")
    try:
        # Dynamically infer dimensions from the first sample
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
            use_gpcr_cf_embed=args.use_gpcr_cf_embed,
            num_gpcr_classes=num_classes,
            num_gpcr_families=num_families
        ).to(device)
    except Exception as e:
        logger.error(f"Failed to initialize model architecture: {e}")
        return
    
    # 3. Load Weights
    logger.info(f"Loading pretrained weights from: {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return
    
    # 4. Evaluate and Save Results
    logger.info("Running evaluation...")
    try:
        bacc, bind_bacc, results_df = evaluate(model, test_loader, device, return_df=True)
        
        logger.info(f"Test Balanced Accuracy: {bacc:.4f}")
        logger.info(f"Test Binding Balanced Accuracy: {bind_bacc:.4f}")
        
        # Dynamically set output filename based on input query_csv
        input_stem = Path(args.query_csv).stem
        save_path = Path(args.output_dir) / f"predictions_{input_stem}.csv"
        results_df.to_csv(save_path, index=False)
        logger.info(f"Predictions successfully saved to {save_path}")
    except Exception as e:
        logger.error(f"Inference failed during evaluation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPCRact Inference Script")
    
    # Set defaults to target the Quickstart (Sample) setup. Removed required=True.
    parser.add_argument("--data_dir", type=str, default="data/sample", help="Directory containing the query CSV")
    parser.add_argument("--query_csv", type=str, default="toy_dataset.csv", help="Filename of the query CSV")
    parser.add_argument("--protein_graph_dir", type=str, default="data/sample/protein_graphs", help="Directory for protein graphs")
    parser.add_argument("--ligand_graph_dir", type=str, default="data/sample/ligand_graphs", help="Directory for ligand graphs")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to .pt weight file")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--gpcr_meta_dir", type=str, default=None)
    parser.add_argument("--use_gpcr_cf_embed", action=argparse.BooleanOptionalAction, default=True)
    
    # Model Hparams (Must match training configuration)
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