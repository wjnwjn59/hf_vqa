#!/bin/bash

# ============================================================================
# BizGen Image Generation Script
# ============================================================================
# This script runs BizGen to generate images from narrator format data
#
# Usage:
#   bash scripts/run_bizgen.sh <GPU_ID> <START_FILE> <END_FILE>
#
# Example:
#   bash scripts/run_bizgen.sh 0 1 3      # GPU 0: files 1-2
#   bash scripts/run_bizgen.sh 1 3 6      # GPU 1: files 3-5
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================
if [ $# -ne 3 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    echo "Usage: $0 <GPU_ID> <START_FILE> <END_FILE>"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID     : GPU device ID (0, 1, 2, ...)"
    echo "  START_FILE : Start file index (inclusive, 1-based)"
    echo "  END_FILE   : End file index (exclusive, 1-based)"
    echo ""
    echo "Note: Each file contains 50 infographics"
    echo ""
    echo "Example:"
    echo "  $0 0 1 3      # GPU 0: files 1-2 (100 infographics)"
    echo "  $0 1 3 6      # GPU 1: files 3-5 (150 infographics)"
    echo ""
    exit 1
fi

GPU_ID=$1
START_FILE=$2
END_FILE=$3

# ============================================================================
# Configuration
# ============================================================================

# Conda environment path for BizGen
CONDA_BIZGEN="/opt/miniconda3/envs/bizgen/bin/python"

# Paths
NARRATOR_FORMAT_DIR="src/data/narrator/wiki"
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"

# Dataset settings
DATASET_NAME="squad_v2"

# Calculate parameters based on file indices
NUM_FILES=$((END_FILE - START_FILE))
NUM_INFOGRAPHICS=$((NUM_FILES * 50))

# Calculate infographic ID range
FIRST_INFOGRAPHIC_ID=$(((START_FILE - 1) * 50 + 1))
LAST_INFOGRAPHIC_ID=$(((END_FILE - 1) * 50))

# Calculate subset for bizgen (using file indices)
SUBSET="${START_FILE}:${END_FILE}"

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "BizGen Image Generation - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Start File Index    : $START_FILE"
echo "  End File Index      : $END_FILE (exclusive)"
echo "  Number of Files     : $NUM_FILES"
echo "  Total Infographics  : $NUM_INFOGRAPHICS"
echo "  Infographic ID Range: $FIRST_INFOGRAPHIC_ID - $LAST_INFOGRAPHIC_ID"
echo "  Expected Wiki Files : wiki$(printf '%06d' $START_FILE).json - wiki$(printf '%06d' $((END_FILE - 1))).json"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Paths:"
echo "  Narrator Format Dir : $NARRATOR_FORMAT_DIR"
echo "  BizGen Directory    : $BIZGEN_DIR"
echo "  Checkpoint Directory: $CKPT_DIR"
echo "  Output Dataset      : $DATASET_NAME"
echo ""
echo "======================================================================"
echo ""

# ============================================================================
# Generate Images with BizGen
# ============================================================================
echo "======================================================================"
echo "Generating Images with BizGen"
echo "======================================================================"
echo "Using conda environment: bizgen"
echo "Python: $CONDA_BIZGEN"
echo "Using GPU: $GPU_ID (CUDA device: cuda:$GPU_ID)"
echo "Subset: $SUBSET"
echo ""

cd "$BIZGEN_DIR"

$CONDA_BIZGEN inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "../narrator/wiki/" \
    --subset "$SUBSET" \
    --device "cuda:$GPU_ID" \
    --dataset_name "$DATASET_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ BizGen completed successfully!"
    
    # Calculate output directory based on dataset_name
    OUTPUT_DATASET_DIR="output/$DATASET_NAME"
    echo "  Generated images in: $BIZGEN_DIR/$OUTPUT_DATASET_DIR/"
    
    # Count generated images
    if [ -d "$OUTPUT_DATASET_DIR" ]; then
        NUM_GENERATED=$(find "$OUTPUT_DATASET_DIR" -name "*.png" -o -name "*.jpg" | wc -l)
        echo "  Total images generated: $NUM_GENERATED"
    fi
else
    echo ""
    echo "✗ BizGen failed!"
    cd - > /dev/null
    exit 1
fi

cd - > /dev/null

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "BizGen Completed Successfully! - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  GPU                 : $GPU_ID"
echo "  Processed Files     : [$START_FILE, $END_FILE)"
echo "  Number of Files     : $NUM_FILES"
echo "  Total Infographics  : $NUM_INFOGRAPHICS"
echo "  Infographic ID Range: $FIRST_INFOGRAPHIC_ID - $LAST_INFOGRAPHIC_ID"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Output Location:"
echo "  Generated Images    : $BIZGEN_DIR/output/$DATASET_NAME/narrator*/"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"
