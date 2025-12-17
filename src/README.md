# GPCRact Source Code

This directory contains the core implementation of the GPCRact framework. The code is modularized to ensure readability and maintainability.

## File Descriptions

### `model.py`
Contains the definition of the **GPCRact Hierarchical Architecture** (`GPCRact_Model`). It implements:
- **Stage 1:** Protein-Ligand Interaction Module (Cross-Attention).
- **Stage 2:** Allosteric Propagation Module (EGNN + Gating Mechanism).
- **Stage 3:** Global Integration (Transformer Encoder) and Activity Prediction Heads.

### `layers.py`
Implements the building blocks of the neural network:
- **EGNN / E_GCL:** Equivariant Graph Convolutional Layers with coordinate updates.
- **E_GCL_Gated:** Custom gated layer to control signal flow.
- **Global Residuals:** Skip connections for better gradient flow in deep networks.

### `dataset.py`
Handles data loading and graph processing:
- **GraphDataset:** PyTorch Geometric dataset class for loading protein and ligand graphs.
- **Data Validation:** Automatic validation of graph integrity before training.
- **Collate Functions:** Batch processing utilities.

### `loss.py`
Contains custom loss functions, including implementations for handling class imbalance (e.g., Focal Loss) used during development.

### `utils.py`
Utility functions for reproducibility and training control:
- **Seeding:** Ensures deterministic results.
- **EarlyStopping:** Prevents overfitting by monitoring validation metrics.
