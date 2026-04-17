import sys
import os
import argparse
import numpy as np
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

def apply_confidence_rescue(df, lower_bind_thresh=0.4, upper_bind_thresh=0.5,
                            activity_conf_thresh=0.95):
    """
    Confidence-based rescue logic (Supplementary Table S5 / Figure S9).

    Recombines the two per-sample stage outputs into the original three-class prediction {0: non-binder, 1: antagonist, 2: agonist},
    with an optional rescue step that promotes low-confidence binders to the class predicted by a highly confident activity head.

    Rule:
      1. Hard-threshold binding at `upper_bind_thresh` (paper: 0.5).
      2. For samples with `binding_prob` in the uncertainty window
         (`lower_bind_thresh`, `upper_bind_thresh`) (paper: (0.4, 0.5)),
         if the activity head's max softmax probability exceeds
         `activity_conf_thresh` (paper: 0.95), promote the sample to
         a binder and assign the activity-head-predicted class.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `evaluate(..., return_df=True)` with columns
        `Binding_Prob`, `Logit_Antagonist`, `Logit_Agonist`.
    lower_bind_thresh, upper_bind_thresh : float
        Bounds of the binding-probability uncertainty window.
    activity_conf_thresh : float
        Minimum activity-head softmax probability required to rescue a
        sample from the uncertainty window.

    Returns
    -------
    np.ndarray
        Integer array of shape (len(df),) with final class labels
        (0: non-binder, 1: antagonist, 2: agonist).
    """
    binding_prob = df['Binding_Prob'].values

    # Numerically stable softmax over activity logits.
    activity_logits = df[['Logit_Antagonist', 'Logit_Agonist']].values
    exp_logits = np.exp(activity_logits - np.max(activity_logits, axis=1, keepdims=True))
    activity_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    predicted_as_binder = (binding_prob >= upper_bind_thresh).astype(int)
    predicted_activity_type = np.argmax(activity_probs, axis=1)  # 0: ant, 1: ago

    # Primary decision: 0 = non-binder, shift activity {0,1} -> {1,2} for binders.
    final_preds = np.zeros(len(df), dtype=int)
    binder_mask = (predicted_as_binder == 1)
    final_preds[binder_mask] = predicted_activity_type[binder_mask] + 1

    # Rescue: uncertain binding + high-confidence activity -> promote to binder.
    grey_area_mask = (binding_prob > lower_bind_thresh) & (binding_prob < upper_bind_thresh)
    activity_max_prob = np.max(activity_probs, axis=1)
    high_confidence_mask = (activity_max_prob > activity_conf_thresh)
    rescue_mask = grey_area_mask & high_confidence_mask

    if rescue_mask.any():
        final_preds[rescue_mask] = predicted_activity_type[rescue_mask] + 1

    return final_preds

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

        # Optional post-hoc rescue logic (Supplementary Table S5).
        if args.apply_rescue:
            final_pred = apply_confidence_rescue(
                results_df,
                lower_bind_thresh=args.rescue_lower,
                upper_bind_thresh=args.rescue_upper,
                activity_conf_thresh=args.rescue_conf,
            )
            results_df['Final_Pred'] = final_pred  # 0: NB, 1: Antagonist, 2: Agonist
            logger.info(
                f"Applied rescue logic with (lower={args.rescue_lower}, "
                f"upper={args.rescue_upper}, conf={args.rescue_conf}); "
                f"added 'Final_Pred' column."
            )

        input_stem = Path(args.query_csv).stem
        save_path = Path(args.output_dir) / f"predictions_{input_stem}.csv"
        results_df.to_csv(save_path, index=False)
        logger.info(f"Predictions successfully saved to {save_path}")
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
    parser.add_argument("--use_gpcr_cf_embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--prop_attn_layers", type=int, default=3)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--elem_emb_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=32)
    # Confidence-based rescue logic (Supplementary Table S5).
    parser.add_argument("--apply_rescue", action="store_true",
                        help="Apply post-hoc confidence-based rescue logic to produce a final 3-class prediction column (Final_Pred).")
    parser.add_argument("--rescue_lower", type=float, default=0.4,
                        help="Lower bound of binding-probability uncertainty window.")
    parser.add_argument("--rescue_upper", type=float, default=0.5,
                        help="Upper bound of binding-probability uncertainty window (also serves as the primary binder threshold).")
    parser.add_argument("--rescue_conf", type=float, default=0.95,
                        help="Activity-head confidence threshold for rescue.")
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
