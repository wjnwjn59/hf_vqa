#!/bin/bash

# ============================================================================
# BizGen Image Generation Pipeline
# ============================================================================
# This script only runs BizGen image generation step
# (assumes narrator format JSON files already exist)
#
# Usage:
#   bash scripts/run_bizgen_only.sh <GPU_ID> <START_IDX> <END_IDX> [DATASET_NAME]
#
# Example:
#   bash scripts/run_bizgen_only.sh 0 0 10000                # GPU 0: images 0-10000, dataset "squad_v2"
#   bash scripts/run_bizgen_only.sh 1 10000 20000            # GPU 1: images 10000-20000, dataset "squad_v2"
#   bash scripts/run_bizgen_only.sh 2 20000 30000 "my_data"  # GPU 2: images 20000-30000, dataset "my_data"
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    echo "Usage: $0 <GPU_ID> <START_IDX> <END_IDX> [DATASET_NAME]"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID       : GPU device ID (0, 1, 2, ...)"
    echo "  START_IDX    : Start index (inclusive)"
    echo "  END_IDX      : End index (exclusive)"
    echo "  DATASET_NAME : Optional dataset name (default: squad_v2)"
    echo ""
    echo "Example:"
    echo "  $0 0 0 10000             # GPU 0: process 10000 images with default dataset name"
    echo "  $0 1 10000 20000         # GPU 1: process 10000 images with default dataset name"
    echo "  $0 2 20000 30000 my_data # GPU 2: process 10000 images with custom dataset name"
    echo ""
    exit 1
fi

GPU_ID=$1
START_IDX=$2
END_IDX=$3
DATASET_NAME=${4:-"squad_v2"}  # Default to squad_v2 if not provided

# ============================================================================
# Configuration
# ============================================================================
export PYTHONPATH="./:$PYTHONPATH"

# Conda environment path
CONDA_BIZGEN="/opt/miniconda3/envs/khoina_bizgen/bin/python"

# Paths
NARRATOR_FORMAT_DIR="src/data/create_data/output/narrator_format"
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"

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
echo "BizGen Image Generation - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Start Index         : $START_IDX"
echo "  End Index           : $END_IDX"
echo "  Number of Images    : $NUM_IMAGES"
echo "  Expected Files      : $NUM_FILES (wiki$(printf '%06d' $FIRST_FILE_IDX).json - wiki$(printf '%06d' $LAST_FILE_IDX).json)"
echo "  BizGen Subset       : $SUBSET"
echo "  Dataset Name        : $DATASET_NAME"
echo ""
echo "Paths:"
echo "  Narrator Format     : $NARRATOR_FORMAT_DIR"
echo "  BizGen Directory    : $BIZGEN_DIR"
echo "  Checkpoint Dir      : $CKPT_DIR"
echo ""
echo "======================================================================"
echo ""

# ============================================================================
# Check Prerequisites
# ============================================================================
echo "======================================================================"
echo "Checking Prerequisites"
echo "======================================================================"

# Check if narrator format directory exists
if [ ! -d "$NARRATOR_FORMAT_DIR" ]; then
    echo "✗ Error: Narrator format directory not found: $NARRATOR_FORMAT_DIR"
    echo "  Please run the full pipeline first to generate narrator format files."
    exit 1
fi

# Check if expected wiki files exist
EXPECTED_FILES=0
MISSING_FILES=0

for ((i=FIRST_FILE_IDX; i<=LAST_FILE_IDX; i++)); do
    WIKI_FILE="$NARRATOR_FORMAT_DIR/wiki$(printf '%06d' $i).json"
    EXPECTED_FILES=$((EXPECTED_FILES + 1))
    
    if [ ! -f "$WIKI_FILE" ]; then
        if [ $MISSING_FILES -eq 0 ]; then
            echo "✗ Missing narrator format files:"
        fi
        echo "  - $(basename $WIKI_FILE)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "✗ Error: $MISSING_FILES out of $EXPECTED_FILES expected files are missing."
    echo "  Please run the full pipeline first to generate these files:"
    echo "  bash scripts/run_narrator_pipeline.sh $GPU_ID $START_IDX $END_IDX"
    exit 1
fi

echo "✓ All $EXPECTED_FILES narrator format files found"
echo ""

# ============================================================================
# Generate Images with BizGen
# ============================================================================
echo "======================================================================"
echo "Generating Images with BizGen"
echo "======================================================================"
echo "Using conda environment: khoina_bizgen"
echo "Python: $CONDA_BIZGEN"
echo "Using GPU: cuda:$GPU_ID (via --device parameter)"
echo "Subset: $SUBSET"
echo "Dataset: $DATASET_NAME"
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
    echo "✓ BizGen generation completed successfully!"
    
    # Count generated images across all narrator folders
    OUTPUT_DATASET_DIR="output/$DATASET_NAME"
    if [ -d "$OUTPUT_DATASET_DIR" ]; then
        NUM_GENERATED=$(find "$OUTPUT_DATASET_DIR" -name "*.png" | grep -v "bbox" | grep -v "lcfg" | wc -l)
        NUM_FOLDERS=$(find "$OUTPUT_DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  Total images generated: $NUM_GENERATED"
        echo "  Total folders created: $NUM_FOLDERS"
        echo "  Generated images in: $BIZGEN_DIR/output/$DATASET_NAME/"
    fi
else
    echo ""
    echo "✗ BizGen generation failed!"
    cd - > /dev/null
    exit 1
fi

cd - > /dev/null

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "BizGen Generation Completed Successfully! - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  GPU                 : $GPU_ID"
echo "  Processed Range     : [$START_IDX, $END_IDX)"
echo "  Number of Images    : $NUM_IMAGES"
echo "  Dataset Name        : $DATASET_NAME"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Output Locations:"
echo "  Generated Images    : $BIZGEN_DIR/output/$DATASET_NAME/narrator*/"
echo "  Source Files        : $NARRATOR_FORMAT_DIR/wiki$(printf '%06d' $FIRST_FILE_IDX).json - wiki$(printf '%06d' $LAST_FILE_IDX).json"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"