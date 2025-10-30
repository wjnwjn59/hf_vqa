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
#   bash scripts/run_narrator_pipeline.sh <GPU_ID> <START_FILE_IDX> <END_FILE_IDX>
#
# Example:
#   bash scripts/run_narrator_pipeline.sh 0 1 201      # GPU 0: files 1-200 (10,000 images)
#   bash scripts/run_narrator_pipeline.sh 1 201 401   # GPU 1: files 201-400 (10,000 images)
#   bash scripts/run_narrator_pipeline.sh 2 401 601   # GPU 2: files 401-600 (10,000 images)
#
# Note: Each file contains 50 images. File indices are 1-based.
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================
if [ $# -ne 3 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    echo "Usage: $0 <GPU_ID> <START_FILE_IDX> <END_FILE_IDX>"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID        : GPU device ID (0, 1, 2, ...)"
    echo "  START_FILE_IDX: Start file index (inclusive, 1-based)"
    echo "  END_FILE_IDX  : End file index (exclusive, 1-based)"
    echo ""
    echo "Note: Each file contains 50 images. Use multiples of 50 for image counts."
    echo ""
    echo "Example:"
    echo "  $0 0 1 201     # GPU 0: process files 1-200 (10,000 images)"
    echo "  $0 1 201 401   # GPU 1: process files 201-400 (10,000 images)"
    echo ""
    exit 1
fi

GPU_ID=$1
START_FILE_IDX=$2
END_FILE_IDX=$3

# ============================================================================
# Validation
# ============================================================================
# Validate that file indices are positive integers
if ! [[ "$START_FILE_IDX" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: START_FILE_IDX must be a positive integer >= 1"
    exit 1
fi

if ! [[ "$END_FILE_IDX" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: END_FILE_IDX must be a positive integer >= 1"
    exit 1
fi

# Validate that end > start
if [ "$END_FILE_IDX" -le "$START_FILE_IDX" ]; then
    echo "Error: END_FILE_IDX must be greater than START_FILE_IDX"
    exit 1
fi

# ============================================================================
# Configuration
# ============================================================================
# Do NOT set CUDA_VISIBLE_DEVICES - let inference.py handle GPU selection via --device
export PYTHONPATH="./:$PYTHONPATH"

# Conda environment paths
CONDA_WIKI="/opt/miniconda3/envs/thinh_wiki/bin/python"
CONDA_BIZGEN="/opt/miniconda3/envs/khoina_bizgen/bin/python"

# Paths
SQUAD_TRAIN="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl"
TEMPLATE_PATH="./src/prompts/bizgen_context_qa_full.jinja"
INFOGRAPHIC_DIR="`src/data/create_data/output/infographic`"
NARRATOR_FORMAT_DIR="src/data/create_data/output/narrator_format"
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"

# Model settings
MODEL_NAME="unsloth/Qwen3-8B"
BATCH_SIZE=8
DATASET_TYPE="squad_v2"
DATASET_NAME="squad_v2"
SEED=42

# Calculate expected outputs
NUM_FILES=$((END_FILE_IDX - START_FILE_IDX))
NUM_IMAGES=$((NUM_FILES * 50))

# Calculate subset for bizgen (convert file indices to image indices for bizgen)
# File index is 1-based, bizgen expects 1-based image indices
BIZGEN_START=$(((START_FILE_IDX - 1) * 50 + 1))
BIZGEN_END=$(((END_FILE_IDX - 1) * 50))
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
echo "  Start File Index    : $START_FILE_IDX"
echo "  End File Index      : $END_FILE_IDX"
echo "  Number of Files     : $NUM_FILES"
echo "  Number of Images    : $NUM_IMAGES (50 images per file)"
echo "  Expected Files      : wiki$(printf '%06d' $START_FILE_IDX).json - wiki$(printf '%06d' $((END_FILE_IDX-1))).json"
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
echo "Using conda environment: thinh_wiki"
echo "Python: $CONDA_WIKI"
echo "Using GPU: $GPU_ID"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_WIKI src/data/narrator/generate_infographic_data.py \
    --model_name "$MODEL_NAME" \
    --input_data "$SQUAD_TRAIN" \
    --template_path "$TEMPLATE_PATH" \
    --dataset_type "$DATASET_TYPE" \
    --start $START_FILE_IDX \
    --end $END_FILE_IDX \
    --batch_size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 1 completed successfully!"
    echo "  Generated files: infographic$(printf '%06d' $START_FILE_IDX).json - infographic$(printf '%06d' $((END_FILE_IDX-1))).json"
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

$CONDA_WIKI src/data/narrator/merge_narrator_bboxes.py \
    --infographic-dir "$INFOGRAPHIC_DIR" \
    --start $START_FILE_IDX \
    --end $END_FILE_IDX \
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
echo "Using conda environment: khoina_bizgen"
echo "Python: $CONDA_BIZGEN"
echo "Using GPU: cuda:$GPU_ID (via --device parameter)"
echo "Subset: $SUBSET"
echo ""

cd "$BIZGEN_DIR"

$CONDA_BIZGEN inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "../create_data/output/narrator_format/" \
    --subset "$SUBSET" \
    --dataset_name "$DATASET_NAME" \
    --device "cuda:$GPU_ID"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 3 completed successfully!"
    
    # Output is now in dataset_name/narratorXXXXXX folders
    echo "  Generated images in: $BIZGEN_DIR/output/$DATASET_NAME/"
    
    # Count generated images across all narrator folders
    OUTPUT_DATASET_DIR="output/$DATASET_NAME"
    if [ -d "$OUTPUT_DATASET_DIR" ]; then
        NUM_GENERATED=$(find "$OUTPUT_DATASET_DIR" -name "*.png" | grep -v "bbox" | grep -v "lcfg" | wc -l)
        NUM_FOLDERS=$(find "$OUTPUT_DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  Total images generated: $NUM_GENERATED"
        echo "  Total folders created: $NUM_FOLDERS"
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
echo "  Processed File Range: [$START_FILE_IDX, $END_FILE_IDX)"
echo "  Number of Files     : $NUM_FILES"
echo "  Number of Images    : $NUM_IMAGES"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Output Locations:"
echo "  1. Infographic JSON : $INFOGRAPHIC_DIR/infographic$(printf '%06d' $START_FILE_IDX).json - infographic$(printf '%06d' $((END_FILE_IDX-1))).json"
echo "  2. Narrator Format  : $NARRATOR_FORMAT_DIR/wiki$(printf '%06d' $START_FILE_IDX).json - wiki$(printf '%06d' $((END_FILE_IDX-1))).json"
echo "  3. Generated Images : $BIZGEN_DIR/output/$DATASET_NAME/narrator*/"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"
