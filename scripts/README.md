# 🚀 Model Training & Inference Scripts

This directory contains the executable scripts to train the GPCRact model, perform inference on new data, and run hyperparameter optimization (HPO).

## 1. Run Inference (Quickstart & Evaluation)

Use `inference.py` to evaluate a trained model checkpoint on a test set. It generates a CSV file with predicted probabilities for Agonism, Antagonism, and Non-binding.

### ⚡ Quickstart (Toy Sample)
By default, the script is configured to run on a minimal toy dataset. You can run it directly without any arguments:
```bash
python scripts/inference.py
```
(This will read from `data/sample/` and output predictions to `results/predictions.csv`)

### 🧪 Full Evaluation
To evaluate the model on the full test split after downloading the dataset:
```bash
python scripts/inference.py \
    --data_dir data/splits \
    --query_csv scaffold_test.csv \
    --model_path checkpoints/best_model.pt \
    --protein_graph_dir data/protein_graphs \
    --ligand_graph_dir data/ligand_graphs \
    --output_dir results/ \
    --batch_size 32
```

### Output
The script saves a CSV file (e.g., `predictions_scaffold_test.csv`) to `--output_dir`. Columns:

* `Ikey`, `UniProt` — identifiers.
* `Binding_Prob` — Stage-1 sigmoid probability of being a binder.
* `Activity_Pred` — Stage-2 argmax over {0: antagonist, 1: agonist}. Only meaningful for predicted binders.
* `Logit_Antagonist`, `Logit_Agonist` — raw Stage-2 logits (before softmax).
* `Final_Pred` *(added only when `--apply_rescue` is passed)* — post-hoc 3-class label {0: non-binder, 1: antagonist, 2: agonist} produced by the confidence-based rescue rule described in Supplementary Table S5.

### With Rescue Logic (Paper-matched 3-class output)
To reproduce the paper's final 3-class prediction, pass `--apply_rescue`. Defaults match Supplementary Table S5 (lower=0.4, upper=0.5, conf=0.95):
```bash
python scripts/inference.py \
    --data_dir data/splits --query_csv scaffold_test.csv \
    --model_path checkpoints/best_model.pt \
    --protein_graph_dir data/protein_graphs \
    --ligand_graph_dir data/ligand_graphs \
    --output_dir results/ \
    --apply_rescue
```


## 2. Train the Model

Use `train.py` to train the GPCRact model from scratch using the scaffold-based data splits.

### Usage
```bash
python scripts/train.py \
    --data_dir data/splits \
    --protein_graph_dir data/protein_graphs \
    --ligand_graph_dir data/ligand_graphs \
    --save_dir checkpoints/ \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.0001
```

#### Key Arguments
`--data_dir` Path to the directory containing `scaffold_train.csv` and `scaffold_val.csv`

`--protein_graph_dir` Path to the directory containing protein graph tensors `.pt`

`--ligand_graph_dir` Path to the directory containing ligand graph tensors `.pt`

`--save_dir` Directory to save model checkpoints.

`--enc_layers` Number of layers for the EGNN encoder.

`--prop_attn_layers` Number of layers for the Global Attention module.


## 3. Hyperparameter Optimization

We use **Weights & Biases (W&B)** for Bayesian hyperparameter optimization. The optimization configuration is defined in `configs/sweep_config.yaml.`

### Prerequisites

Ensure you have a W&B account and have logged in:

```bash
wandb login
```

### Running a Sweep

**Step 1: Initialize the sweep** This command registers the sweep configuration with the W&B server.

```bash
wandb sweep configs/sweep_config.yaml
```

**Step 2: Start the Agent** Run the agent using the command provided in the previous step. The agent will pull parameters from the server and run `scripts/hyperparam_opt.py.`

```bash
wandb agent <username/project/sweep_id>
```

* Note: The HPO script assumes that graph datasets with different k-NN settings (e.g., k=32, 64, 128) have been pre-generated. Please check `scripts/hyperparam_opt.py` to adjust paths if necessary.

