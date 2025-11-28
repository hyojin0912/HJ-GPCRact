# Analysis and Reproduction Notebooks

This directory contains the Jupyter Notebooks required to reproduce the statistical analyses, main figures, and mechanistic validations presented in the manuscript.

The notebooks are organized sequentially, corresponding to the flow of the paper—from initial biophysical investigations to the post-hoc interpretability of the trained GPCRact model.

## ⚠️ Prerequisites

Some notebooks require trained model artifacts and processed graph datasets. Since these files are too large to host directly on GitHub, you must first run the training pipeline or ensure the data is prepared.

1.  **Environment:** Ensure the `gpcr_act` conda environment is activated.
2.  **Data:** The preprocessing pipeline (in `preprocessing/`) must be completed.
3.  **Model Checkpoints:** Notebooks `04`, `06`, and `07` require a trained GPCRact model (`models/saved/gpcract_final.pt`).

---

## Notebook Descriptions

### 1. Biophysical Principles & Data Analysis (Figures 1-4)

* **[`01_receptor_dynamics_analysis.ipynb`](01_receptor_dynamics_analysis.ipynb)** (Figure 1)
    * **Objective:** Defines the structural ground truth of activation.
    * **Content:** Performs geometric analysis on PDB structures to quantify the outward displacement of TM6 and inward movement of TM7 across agonist- and antagonist-bound states.

* **[`02_sequence_structure_correlation.ipynb`](02_sequence_structure_correlation.ipynb)** (Figure 3)
    * **Objective:** Demonstrates the limitations of sequence-based features.
    * **Content:** Investigates correlations between MSA-derived features (residue conservation, sequence distance) and actual 3D structural dynamics, highlighting why 1D representations fail to capture allosteric movements.

* **[`03_activity_decision_tree.ipynb`](03_activity_decision_tree.ipynb)** (Figure 4)
    * **Objective:** Proves that 3D geometry deterministically encodes activity.
    * **Content:** Trains and visualizes an interpretable Decision Tree classifier to identify key inter-residue distances (e.g., TM3-TM6) that distinguish agonists from antagonists using structural features alone.

### 2. Mechanistic Interpretability (Figures 7)

* **[`04_mechanistic_interpretability.ipynb`](04_mechanistic_interpretability.ipynb)** (Figures 7)
    * **Objective:** Validates that GPCRact learns biologically relevant allosteric pathways.
    * **Content:**
        * **Attention Weight Analysis:** Extracts self-attention weights from the trained model to quantify the importance of canonical motifs (DRY, NPxxY, PIF) across different receptor families.


### 3. Supplementary Validations

* **[`05_supplementary_PRS_analysis.ipynb`](05_supplementary_PRS_analysis.ipynb)**
    * **Objective:** Comparison with classical biophysical methods.
    * **Content:** Implements Perturbation Response Scanning (PRS) using ProDy to compare intrinsic protein dynamics against the ligand-conditioned functional predictions of GPCRact.

* **[`06_sensitivity_analysis.ipynb`](06_sensitivity_analysis.ipynb)**
    * **Objective:** Validates robustness to conformer selection.
    * **Content:** Generates an ensemble of 20 distinct 3D conformers for test ligands and measures the variance in model predictions, demonstrating stability across input geometries.

* **[`07_mutation_study_causality.ipynb`](07_mutation_study_causality.ipynb)**
    * **Objective:** Validates physical causality via *in silico* mutation.
    * **Content:** Generates the graph for the **R3.50A** mutant of the ADRB2 receptor and compares the predicted agonist probability against the Wild-Type, confirming that the model captures the causal role of this key switch residue.

---

## How to Run

Launch Jupyter Lab or Notebook from this directory:

```bash
cd analysis
jupyter lab
```

- Note: Ensure that DATA_DIR and MODEL_SAVE_PATH in the configuration cells of each notebook point to the correct locations in your local setup.
