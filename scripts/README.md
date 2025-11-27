# Model Training & Inference Scripts

This directory contains the executable scripts to train the GPCRact model, perform inference on new data, and run hyperparameter optimization (HPO).


## 1. Train the Model

Use `train.py` to train the GPCRact model from scratch using the preprocessed data splits.

### Usage
```bash
python scripts/train.py \
    --data_dir data/splits \
    --protein_graph_dir data/processed/protein_graphs \
    --ligand_graph_dir data/processed/ligand_graphs \
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


## 2. Run Inference

Use `inference.py` to evaluate a trained model checkpoint on a test set (or any dataset provided in a CSV). It generates a CSV file with predicted probabilities for Binding, Antagonism, and Agonism.

### Usage
```bash
python scripts/inference.py \
    --data_dir data/splits \
    --model_path checkpoints/best_model.pt \
    --protein_graph_dir data/processed/protein_graphs \
    --ligand_graph_dir data/processed/ligand_graphs \
    --output_dir results/ \
    --batch_size 32
```

#### Output
The script will save a CSV file (e.g., `predictions_test.csv`) to the output_dir. The CSV includes:
* `Ikey`, `UniProt`: Identifiers

* `Binding_Prob`: Predicted probability of binding.

* `Prediction`: Final class prediction (0: Non-binder, 1: Antagonist, 2: Agonist).

* `Logit_Antagonist`, `Logit_Agonist`: Raw model outputs.


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

