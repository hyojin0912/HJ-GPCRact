#!/bin/bash

# ===================================================================
# DeepREAL Benchmark Reproduction Script
# ===================================================================

# 1. Move to script directory
cd "$(dirname "$0")"
echo "INFO: Working directory is $(pwd)"

# 2. Check for missing dependencies (Crucial Step)
# Since we only updated train.py and models.py, other files must be copied by the user
MISSING_FILES=0
for file in "data_tool_box.py" "model_Yang.py" "resnet.py" "fp_models.py" "utils.py"; do
    if [ ! -f "./$file" ]; then
        echo "ERROR: Missing dependency '$file'. Please copy it from the original DeepREAL repository."
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    exit 1
fi

# 3. Create Output Directory
mkdir -p results

# 4. Run Training Script
echo "INFO: Starting DeepREAL Training..."
python train.py \
    --cwd ./ \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-5 \
    --prot_descriptor DISAE \
    --use_cuda True

echo "INFO: Benchmark Finished."