#!/bin/bash

# ===================================================================
# AiGPro Benchmark Reproduction Script
# ===================================================================

# 1. Move to the script's directory (Safety for execution context)
cd "$(dirname "$0")"
echo "INFO: Working directory is $(pwd)"

# 2. Verify Data Existence
if [ ! -f "./data/train_set.csv" ] || [ ! -f "./data/test_set.csv" ]; then
    echo "ERROR: Raw data files not found in ./data/"
    echo "       Please copy 'train_set_cold_prn.csv' and 'test_set_cold_prn.csv' to benchmarks/AiGPro/data/"
    exit 1
fi

# 3. Restart Docker Container
echo "INFO: Building and starting Docker container..."
docker-compose down
docker-compose up -d --build

# 4. Check GPU Availability
echo "INFO: Checking GPU availability..."
docker exec -t aigpro_benchmark nvidia-smi

# 5. Run Preprocessing & Splitting
# Note: The logic for splitting and truncation is integrated into preprocess_data.py
echo "INFO: Starting Preprocessing and Data Splitting..."
docker exec -t aigpro_benchmark python ./data/preprocess_data.py

# 6. Run Training
echo "INFO: Starting Model Training..."
# Execute the script directly instead of using module flag to avoid path confusion
docker exec -t aigpro_benchmark python ./scripts/trainer.py

echo "INFO: Benchmark Completed. Results are saved in ./data/"