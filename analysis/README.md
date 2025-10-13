# Analysis and Figure Generation Notebooks

This directory contains the Jupyter Notebooks used to perform the analyses and generate the main figures presented in the manuscript.

## Notebook Descriptions

-   **[`01_receptor_dynamics_analysis.ipynb`](01_receptor_dynamics_analysis.ipynb):**
    This notebook performs the analysis of structural dynamics upon ligand binding. It generates the boxplots comparing the outward displacement of Transmembrane Helix 6 (TM6) and the inward movement of Transmembrane Helix 7 (TM7) between antagonist-bound and agonist-bound states.

-   **[`02_MSA_3D_correlation_analysis.ipynb`](02_MSA_3D_correlation_analysis.ipynb):**
-   **[`02_MSA_Ca_correlation_analysis.ipynb`](02_MSA_Ca_correlation_analysis.ipynb):**
    This notebook investigates the correlations between different data modalities. It includes the code to generate the raincloud plots for:
    1.  MSA Distance vs. 3D Cα Distance.
    2.  MSA Conservation vs. Structural Dynamics (Cα Displacement).

-   **[`03_activity_decision_tree.ipynb`](03_activity_decision_tree.ipynb):**
    This notebook trains and visualizes a simple decision tree classifier to identify the key structural features (inter-residue distances) that distinguish between agonist and antagonist activity.


To run these notebooks, please ensure you have activated the `gpcr_act` conda environment.
