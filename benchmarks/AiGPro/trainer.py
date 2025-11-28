import os
import torch
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
# The 'aigpro' package is recognized via PYTHONPATH in docker-compose
from aigpro.models.model_module import PDBModelModuleGPCR
from aigpro.data.gpcr_dataset import GPCRDataModule

torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Docker Internal Paths
BASE_DIR = "/app"
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

def move_to_cuda(obj, device='cuda:0'):
    """Recursively moves tensors to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cuda(o, device) for o in obj)
    elif isinstance(obj, dict):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    return obj

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
    )

    # Initialize Model
    model = PDBModelModuleGPCR(learning_rate=1e-4, batch_size=64)
    
    # Initialize DataModule
    datamodule = GPCRDataModule(
        batch_size=64,
        num_workers=0,  # Disable multiprocessing to avoid Docker shared memory issues
        train_file=os.path.join(DATA_DIR, "train_set_updated.csv"),
        test_file=os.path.join(DATA_DIR, "test_set_updated.csv"),
        val_file=os.path.join(DATA_DIR, "validation_set_updated.csv")
    )

    datamodule.setup()

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    # Start Training
    trainer.fit(model, datamodule=datamodule)

    # --- Prediction Phase ---
    print("Starting prediction on Train and Test sets...")
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 1. Prediction on Train Set
    train_dataloader = datamodule.predict_train_dataloader()
    train_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            batch = move_to_cuda(batch, device=device)

            # Debug: Check device of tensors if needed
            # for i, item in enumerate(batch):
            #     if isinstance(item, torch.Tensor):
            #         print(f"Batch {batch_idx}, Item {i} device: {item.device}")
            
            probs = model.predict_step(batch, batch_idx=0)
            train_predictions.extend(probs.cpu().numpy())
    
    # Save Train Predictions
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_set_updated.csv"))
    # Use 2-class columns
    train_pred_df = pd.DataFrame(train_predictions, columns=["prob_antagonist", "prob_agonist"])  
    train_df_merged = pd.concat([train_df, train_pred_df], axis=1)
    train_df_merged.to_csv(os.path.join(DATA_DIR, "train_set_cold_prn_pred.csv"), index=False)

    # 2. Prediction on Test Set
    test_dataloader = datamodule.test_dataloader()
    test_predictions = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = move_to_cuda(batch, device=device)
            probs = model.predict_step(batch, batch_idx=0)
            test_predictions.extend(probs.cpu().numpy())
    
    # Save Test Predictions
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_set_updated.csv"))
    test_pred_df = pd.DataFrame(test_predictions, columns=["prob_antagonist", "prob_agonist"])
    test_df_merged = pd.concat([test_df, test_pred_df], axis=1)
    test_df_merged.to_csv(os.path.join(DATA_DIR, "test_set_cold_prn_pred.csv"), index=False)
    
    print("AiGPro Benchmark Pipeline Finished successfully.")

if __name__ == "__main__":
    main()