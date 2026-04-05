import sys
import os
import argparse
import pandas as pd
import torch
import logging
import json
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPCRact_Model
from src.dataset import GraphDataset, collate_fn, get_valid_indices
from scripts.train import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting inference on device: {device.type.upper()}")
    
    csv_path = Path(args.data_dir) / args.query_csv
    
    try:
        test_df = pd.read_csv(csv_path)
        valid_idx = get_valid_indices(test_df, args.protein_graph_dir, args.ligand_graph_dir)
        test_df = test_df[valid_idx].reset_index(drop=True)
        
        if test_df.empty:
            logger.error("No valid samples found.")
            return

        test_ds = GraphDataset(root=None, df=test_df, 
                               protein_graph_dir=args.protein_graph_dir, 
                               ligand_graph_dir=args.ligand_graph_dir)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        logger.info(f"Successfully loaded {len(test_ds)} valid samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    logger.info(f"Loading pretrained weights from: {args.model_path}")
    try:
        # Load weights and resolve FutureWarning
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        
        # Infer dimensions dynamically from checkpoint
        p_dim_clean = state_dict['bs_encoder.embedding_in.weight'].shape[1]
        p_dim_full = state_dict.get('protein_embedding_ca.weight', state_dict['bs_encoder.embedding_in.weight']).shape[1]
        l_dim = state_dict['ligand_encoder.embedding_in.weight'].shape[1]
        use_cf = 'gpcr_class_embedding.weight' in state_dict
    except Exception as e:
        logger.error(f"Failed to read checkpoint: {e}")
        return

    logger.info("Initializing GPCRact model architecture...")
    try:
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
            use_gpcr_cf_embed=use_cf,
            num_gpcr_classes=num_classes,
            num_gpcr_families=num_families
        ).to(device)

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return
    
    logger.info("Running evaluation...")
    try:
        bacc, bind_bacc, results_df = evaluate(model, test_loader, device, return_df=True)
        logger.info(f"Test Balanced Accuracy: {bacc:.4f}")
        logger.info(f"Test Binding Balanced Accuracy: {bind_bacc:.4f}")
        
        input_stem = Path(args.query_csv).stem
        save_path = Path(args.output_dir) / f"predictions_{input_stem}.csv"
        results_df.to_csv(save_path, index=False)
        logger.info(f"✅ Predictions successfully saved to {save_path}")
    except Exception as e:
        logger.error(f"Inference failed during evaluation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/sample")
    parser.add_argument("--query_csv", type=str, default="toy_dataset.csv")
    parser.add_argument("--protein_graph_dir", type=str, default="data/sample/protein_graphs")
    parser.add_argument("--ligand_graph_dir", type=str, default="data/sample/ligand_graphs")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--gpcr_meta_dir", type=str, default=None)
    parser.add_argument("--use_gpcr_cf_embed", action=argparse.BooleanOptionalAction, default=True)
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
