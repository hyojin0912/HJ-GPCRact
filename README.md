# HJ-GPCRact

This repository contains the datasets and scripts associated with the study:

> **GPCRact: A Hierarchical Framework for Predicting Ligand-Induced GPCR Activity via Structure-Based Graph Modeling**

## 📁 Repository Contents
- `data/` – curated datasets for training and evaluation  
- `model/` – pretrained model checkpoints  
- `scripts/` – preprocessing, training, and evaluation scripts  

## 🧬 Data Description
The dataset includes ligand–GPCR pairs curated from ChEMBL and GPCRdb, with structure-based features extracted using AlphaFold2 and GNN-derived embeddings.

## 🧠 Model Information
The GPCRact framework consists of:
1. A structure-based graph encoder for ligand and receptor
2. An equivariant GNN for residue-level activation mapping
3. A hierarchical classifier predicting agonist vs. antagonist activity

## 📬 Contact
For questions, please contact:
**Hyojin Son** (hyojin0912@kaist.ac.kr)
