#!/bin/bash

set -e

echo "==========================================="
echo "   Downloading GPCRact Full Dataset...     "
echo "==========================================="

mkdir -p data/ligand_graphs
mkdir -p data/protein_graphs

echo "[1/2] Downloading Ligand Graphs (146 MB)..."
wget -O data/ligand_graphs.tar.gz https://huggingface.co/datasets/hyojin0912/GPCRact/resolve/main/ligand_graphs.tar.gz
echo "Extracting Ligand Graphs..."
tar -xzf data/ligand_graphs.tar.gz -C data/
rm data/ligand_graphs.tar.gz

echo "[2/2] Downloading Protein Graphs (10.8 MB)..."
wget -O data/protein_graphs.tar.gz https://huggingface.co/datasets/hyojin0912/GPCRact/resolve/main/protein_graphs.tar.gz
echo "Extracting Protein Graphs..."
tar -xzf data/protein_graphs.tar.gz -C data/
rm data/protein_graphs.tar.gz

echo "==========================================="
echo "   Dataset successfully downloaded!        "
echo "==========================================="