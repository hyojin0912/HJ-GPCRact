import pandas as pd
import numpy as np
import os
import warnings
from aigpro.chem.descriptors import add_desc, add_charge_fp, morgan_fingerprint

warnings.filterwarnings('ignore')

# Docker Internal Paths
BASE_DIR = "/app/data"
TRAIN_INPUT = os.path.join(BASE_DIR, "train_set.csv")
TEST_INPUT = os.path.join(BASE_DIR, "test_set.csv")

TRAIN_UPDATED = os.path.join(BASE_DIR, "train_set_updated.csv")
TEST_UPDATED = os.path.join(BASE_DIR, "test_set_updated.csv")
VAL_UPDATED = os.path.join(BASE_DIR, "validation_set_updated.csv")

def preprocess_dataset(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Map labels
    if 'Label' in df.columns:
        df['Label'] = df['Label'].map({'agonist': 1, 'antagonist': 0})

    # Rename columns to match AiGPro's expectations
    df = df.rename(columns={
        "SMILES": "canonical_smiles",
        "Sequence": "Align_Sequence",
        "Label": "pEndPoint"
    })

    print("Computing descriptors (this may take a while)...")
    # Add required columns
    df["desc"] = df.apply(lambda row: add_desc(row, col="canonical_smiles", verbose=False), axis=1)
    df["charge_fp"] = df.apply(lambda row: add_charge_fp(row, col="canonical_smiles", verbose=False), axis=1)
    df["morgan_fp"] = df["canonical_smiles"].apply(lambda x: morgan_fingerprint(x).tolist())

    # Add label and uniprot_id
    df["label"] = df["pEndPoint"] 
    df["uniprot_id"] = df["Align_Sequence"].apply(lambda x: x[:6] if len(x) >= 6 else "UNK") 

    # Remove rows where any required column failed to compute
    df = df.dropna(subset=["canonical_smiles", "Align_Sequence", "pEndPoint", "desc", "charge_fp", "morgan_fp"])

    # Save intermediate updated CSV
    cols = ["canonical_smiles", "Align_Sequence", "pEndPoint", "desc", "charge_fp", "morgan_fp", "label", "uniprot_id"]
    df[cols].to_csv(output_path, index=False)
    print(f"Intermediate processing saved to {output_path}")

# 1. Run Initial Preprocessing
print("--- Step 1: Initial Preprocessing ---")
preprocess_dataset(TRAIN_INPUT, TRAIN_UPDATED)
preprocess_dataset(TEST_INPUT, TEST_UPDATED)

# 2. Split Validation Set & Truncate Vectors (Logic from original shell script)
print("--- Step 2: Splitting and Truncating ---")

# Process Train/Val
print(f"Splitting {TRAIN_UPDATED} into Train/Val...")
df_train = pd.read_csv(TRAIN_UPDATED)

# Truncate vectors as per original requirements
df_train['desc'] = df_train['desc'].apply(lambda x: eval(x)[:170] if isinstance(x, str) else x)
df_train['charge_fp'] = df_train['charge_fp'].apply(lambda x: eval(x)[:512] if isinstance(x, str) else x)

# Random Split
train_df = df_train.sample(frac=0.8, random_state=143)
val_df = df_train.drop(train_df.index)

train_df.to_csv(TRAIN_UPDATED, index=False)
val_df.to_csv(VAL_UPDATED, index=False)
print(f"Final Train Set: {len(train_df)}, Validation Set: {len(val_df)}")

# Process Test
print(f"Truncating Test Set {TEST_UPDATED}...")
df_test = pd.read_csv(TEST_UPDATED)
df_test['desc'] = df_test['desc'].apply(lambda x: eval(x)[:170] if isinstance(x, str) else x)
df_test['charge_fp'] = df_test['charge_fp'].apply(lambda x: eval(x)[:512] if isinstance(x, str) else x)
df_test.to_csv(TEST_UPDATED, index=False)

print("Preprocessing and Splitting Complete.")