#!/bin/bash

# ============================================================================
# Narrator Pipeline for Squad v2 Dataset
# ============================================================================
# This script runs the complete pipeline:
# 1. Generate infographic data using Qwen
# 2. Merge bounding boxes
# 3. Generate images with BizGen
#
# Usage:
#   bash scripts/run_narrator_pipeline.sh <GPU_ID> <START_IDX> <END_IDX>
#
# Example:
#   bash scripts/run_narrator_pipeline.sh 0 0 10000      # GPU 0: images 0-10000
#   bash scripts/run_narrator_pipeline.sh 1 10000 20000  # GPU 1: images 10000-20000
#   bash scripts/run_narrator_pipeline.sh 2 20000 30000  # GPU 2: images 20000-30000
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================
if [ $# -ne 3 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    echo "Usage: $0 <GPU_ID> <START_IDX> <END_IDX>"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID    : GPU device ID (0, 1, 2, ...)"
    echo "  START_IDX : Start index (inclusive)"
    echo "  END_IDX   : End index (exclusive)"
    echo ""
    echo "Example:"
    echo "  $0 0 0 10000      # GPU 0: process 10000 images (indices 0-9999)"
    echo "  $0 1 10000 20000  # GPU 1: process 10000 images (indices 10000-19999)"
    echo ""
    exit 1
fi

GPU_ID=$1
START_IDX=$2
END_IDX=$3

# ============================================================================
# Configuration
# ============================================================================
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="./:$PYTHONPATH"

# Conda environments
CONDA_WIKI="wiki"
CONDA_BIZGEN="bizgen"

# Paths
SQUAD_TRAIN="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl"
TEMPLATE_PATH="./src/prompts/bizgen_context_qa_full.jinja"
INFOGRAPHIC_DIR="src/data/create_data/output/infographic"
NARRATOR_FORMAT_DIR="src/data/create_data/output/narrator_format"
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"

# Model settings
MODEL_NAME="unsloth/Qwen3-8B"
BATCH_SIZE=8
DATASET_TYPE="squad_v2"
SEED=42

# Calculate expected outputs
NUM_IMAGES=$((END_IDX - START_IDX))
NUM_FILES=$((NUM_IMAGES / 50))
FIRST_FILE_IDX=$(((START_IDX / 50) + 1))
LAST_FILE_IDX=$(((END_IDX - 1) / 50 + 1))

# Calculate subset for bizgen (1-based indexing)
BIZGEN_START=$((START_IDX + 1))
BIZGEN_END=$((END_IDX))
SUBSET="${BIZGEN_START}:${BIZGEN_END}"

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "Narrator Pipeline - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Start Index         : $START_IDX"
echo "  End Index           : $END_IDX"
echo "  Number of Images    : $NUM_IMAGES"
echo "  Expected Files      : $NUM_FILES (infographic$(printf '%06d' $FIRST_FILE_IDX).json - infographic$(printf '%06d' $LAST_FILE_IDX).json)"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Paths:"
echo "  Input Dataset       : $SQUAD_TRAIN"
echo "  Template            : $TEMPLATE_PATH"
echo "  Infographic Output  : $INFOGRAPHIC_DIR"
echo "  Narrator Format     : $NARRATOR_FORMAT_DIR"
echo "  BizGen Directory    : $BIZGEN_DIR"
echo ""
echo "======================================================================"
echo ""

# ============================================================================
# Step 1: Generate Infographic Data using Qwen
# ============================================================================
echo "======================================================================"
echo "Step 1/3: Generating Infographic Data with Qwen"
echo "======================================================================"
echo "Activating conda environment: $CONDA_WIKI"
echo ""

conda activate $CONDA_WIKI

python src/data/narrator/generate_infographic_data.py \
    --model_name "$MODEL_NAME" \
    --input_data "$SQUAD_TRAIN" \
    --template_path "$TEMPLATE_PATH" \
    --dataset_type "$DATASET_TYPE" \
    --start $START_IDX \
    --end $END_IDX \
    --batch_size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 1 completed successfully!"
    echo "  Generated files: infographic$(printf '%06d' $FIRST_FILE_IDX).json - infographic$(printf '%06d' $LAST_FILE_IDX).json"
else
    echo ""
    echo "✗ Step 1 failed!"
    exit 1
fi

# ============================================================================
# Step 2: Merge Bounding Boxes
# ============================================================================
echo ""
echo "======================================================================"
echo "Step 2/3: Merging Bounding Boxes"
echo "======================================================================"
echo ""

python src/data/narrator/merge_narrator_bboxes.py \
    --infographic-dir "$INFOGRAPHIC_DIR" \
    --start $START_IDX \
    --end $END_IDX \
    --seed $SEED

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 2 completed successfully!"
    echo "  Generated wiki files in: $NARRATOR_FORMAT_DIR"
else
    echo ""
    echo "✗ Step 2 failed!"
    exit 1
fi

# ============================================================================
# Step 3: Generate Images with BizGen
# ============================================================================
echo ""
echo "======================================================================"
echo "Step 3/3: Generating Images with BizGen"
echo "======================================================================"
echo "Activating conda environment: $CONDA_BIZGEN"
echo "Subset: $SUBSET"
echo ""

conda activate $CONDA_BIZGEN

cd "$BIZGEN_DIR"

python inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "../create_data/output/narrator_format/" \
    --subset "$SUBSET"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 3 completed successfully!"
    
    # Calculate output directory
    OUTPUT_SUBSET_DIR="output/subset_${BIZGEN_START}_${BIZGEN_END}"
    echo "  Generated images in: $BIZGEN_DIR/$OUTPUT_SUBSET_DIR"
    
    # Count generated images
    if [ -d "$OUTPUT_SUBSET_DIR" ]; then
        NUM_GENERATED=$(find "$OUTPUT_SUBSET_DIR" -name "*.png" -o -name "*.jpg" | wc -l)
        echo "  Total images generated: $NUM_GENERATED"
    fi
else
    echo ""
    echo "✗ Step 3 failed!"
    cd - > /dev/null
    exit 1
fi

cd - > /dev/null

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "Pipeline Completed Successfully! - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  GPU                 : $GPU_ID"
echo "  Processed Range     : [$START_IDX, $END_IDX)"
echo "  Number of Images    : $NUM_IMAGES"
echo "  Infographic Files   : $NUM_FILES files"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Output Locations:"
echo "  1. Infographic JSON : $INFOGRAPHIC_DIR/infographic$(printf '%06d' $FIRST_FILE_IDX).json - infographic$(printf '%06d' $LAST_FILE_IDX).json"
echo "  2. Narrator Format  : $NARRATOR_FORMAT_DIR/wiki$(printf '%06d' $FIRST_FILE_IDX).json - wiki$(printf '%06d' $LAST_FILE_IDX).json"
echo "  3. Generated Images : $BIZGEN_DIR/output/subset_${BIZGEN_START}_${BIZGEN_END}/"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"
