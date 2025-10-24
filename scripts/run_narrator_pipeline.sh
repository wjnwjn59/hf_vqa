#!/bin/bash

# ============================================================================
# Narrator Pipeline for Squad v2 Dataset
# ============================================================================
# This script runs the complete NARRATOR pipeline:
# 1. Generate 3-stage infographic data using Qwen (generate_stage_infographic.py)
# 2. Merge bounding boxes (merge_stage_narrator.py)
# 3. Generate images with BizGen
#
# Usage:
#   bash scripts/run_narrator_pipeline.sh <GPU_ID> <START_SUBSET> <END_SUBSET>
#
# Example:
#   bash scripts/run_narrator_pipeline.sh 0 1 201      # GPU 0: subsets 1-200 (10,000 images)
#   bash scripts/run_narrator_pipeline.sh 1 201 401   # GPU 1: subsets 201-400 (10,000 images)
#   bash scripts/run_narrator_pipeline.sh 2 401 601   # GPU 2: subsets 401-600 (10,000 images)
#
# Note: Each subset contains 50 images. Subset indices are 1-based.
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================
if [ $# -ne 3 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    echo "Usage: $0 <GPU_ID> <START_SUBSET> <END_SUBSET>"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID       : GPU device ID (0, 1, 2, ...)"
    echo "  START_SUBSET : Start subset index (inclusive, 1-based)"
    echo "  END_SUBSET   : End subset index (exclusive, 1-based)"
    echo ""
    echo "Note: Each subset contains 50 images."
    echo ""
    echo "Example:"
    echo "  $0 0 1 201     # GPU 0: process subsets 1-200 (10,000 images)"
    echo "  $0 1 201 401   # GPU 1: process subsets 201-400 (10,000 images)"
    echo ""
    exit 1
fi

GPU_ID=$1
START_SUBSET=$2
END_SUBSET=$3

# ============================================================================
# Validation
# ============================================================================
# Validate that subset indices are positive integers
if ! [[ "$START_SUBSET" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: START_SUBSET must be a positive integer >= 1"
    exit 1
fi

if ! [[ "$END_SUBSET" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: END_SUBSET must be a positive integer >= 1"
    exit 1
fi

# Validate that end > start
if [ "$END_SUBSET" -le "$START_SUBSET" ]; then
    echo "Error: END_SUBSET must be greater than START_SUBSET"
    exit 1
fi

# ============================================================================
# Configuration
# ============================================================================
export PYTHONPATH="./:$PYTHONPATH"

# Conda environment paths
CONDA_WIKI="/opt/miniconda3/envs/thinh_wiki/bin/python"
CONDA_BIZGEN="/opt/miniconda3/envs/bizgen/bin/python"

# Paths
SQUAD_TRAIN="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl"
STAGE_A_TEMPLATE="./src/prompts/content_des_stage_1.jinja"
STAGE_B_TEMPLATE="./src/prompts/content_des_stage_2.jinja"
STAGE_C_TEMPLATE="./src/prompts/content_des_stage_3.jinja"
INFOGRAPHIC_V2_DIR="src/data/create_data/output/infographic_v2"
NARRATOR_FORMAT_V2_DIR="src/data/create_data/output/narrator_format_v2"
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"

# Model settings
MODEL_NAME="unsloth/Qwen3-8B"
BATCH_SIZE=8
DATASET_NAME="squad_v2_new"
SEED=42

# Calculate expected outputs
NUM_SUBSETS=$((END_SUBSET - START_SUBSET))
NUM_IMAGES=$((NUM_SUBSETS * 50))

# Calculate subset for bizgen (convert subset indices to image indices for bizgen)
# Subset index is 1-based, bizgen expects 1-based image indices
BIZGEN_START=$(((START_SUBSET - 1) * 50 + 1))
BIZGEN_END=$(((END_SUBSET - 1) * 50))
BIZGEN_SUBSET="${BIZGEN_START}:${BIZGEN_END}"

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "Narrator Pipeline - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Start Subset        : $START_SUBSET"
echo "  End Subset          : $END_SUBSET"
echo "  Number of Subsets   : $NUM_SUBSETS"
echo "  Number of Images    : $NUM_IMAGES (50 images per subset)"
echo "  Expected Files      : infographic$(printf '%06d' $START_SUBSET).json - infographic$(printf '%06d' $((END_SUBSET-1))).json"
echo "  BizGen Subset       : $BIZGEN_SUBSET"
echo ""
echo "Paths:"
echo "  Input Dataset       : $SQUAD_TRAIN"
echo "  Stage 1 Template    : $STAGE_A_TEMPLATE"
echo "  Stage 2 Template    : $STAGE_B_TEMPLATE"
echo "  Stage 3 Template    : $STAGE_C_TEMPLATE"
echo "  Infographic v2 Out  : $INFOGRAPHIC_V2_DIR"
echo "  Narrator Format v2  : $NARRATOR_FORMAT_V2_DIR"
echo "  BizGen Directory    : $BIZGEN_DIR"
echo ""
echo "======================================================================"
echo ""

# ============================================================================
# Step 1: Generate 3-Stage Infographic Data using Qwen
# ============================================================================
echo "======================================================================"
echo "Step 1/3: Generating 3-Stage Infographic Data with Qwen"
echo "======================================================================"
echo "Using conda environment: thinh_wiki"
echo "Python: $CONDA_WIKI"
echo "Using GPU: $GPU_ID"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_WIKI src/data/narrator/generate_stage_infographic.py \
    --model_name "$MODEL_NAME" \
    --input_data "$SQUAD_TRAIN" \
    --stage_a "$STAGE_A_TEMPLATE" \
    --stage_b "$STAGE_B_TEMPLATE" \
    --stage_c "$STAGE_C_TEMPLATE" \
    --output_dir "$INFOGRAPHIC_V2_DIR" \
    --start $START_SUBSET \
    --end $END_SUBSET \
    --batch_size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 1 completed successfully!"
    echo "  Generated files: infographic$(printf '%06d' $START_SUBSET).json - infographic$(printf '%06d' $((END_SUBSET-1))).json"
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

$CONDA_WIKI src/data/narrator/merge_stage_narrator.py \
    --infographic-dir "$INFOGRAPHIC_V2_DIR" \
    --output-dir "$NARRATOR_FORMAT_V2_DIR" \
    --start $START_SUBSET \
    --end $END_SUBSET \
    --seed $SEED

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 2 completed successfully!"
    echo "  Generated wiki files in: $NARRATOR_FORMAT_V2_DIR"
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
echo "Subset: $BIZGEN_SUBSET"
echo ""

cd "$BIZGEN_DIR"

$CONDA_BIZGEN inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "../create_data/output/narrator_format_v2/" \
    --subset "$BIZGEN_SUBSET" \
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
echo "  Processed Subsets   : [$START_SUBSET, $END_SUBSET)"
echo "  Number of Subsets   : $NUM_SUBSETS"
echo "  Number of Images    : $NUM_IMAGES"
echo "  BizGen Subset       : $BIZGEN_SUBSET"
echo ""
echo "Output Locations:"
echo "  1. Infographic v2   : $INFOGRAPHIC_V2_DIR/infographic$(printf '%06d' $START_SUBSET).json - infographic$(printf '%06d' $((END_SUBSET-1))).json"
echo "  2. Narrator Format  : $NARRATOR_FORMAT_V2_DIR/wiki$(printf '%06d' $START_SUBSET).json - wiki$(printf '%06d' $((END_SUBSET-1))).json"
echo "  3. Generated Images : $BIZGEN_DIR/output/$DATASET_NAME/narrator*/"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"