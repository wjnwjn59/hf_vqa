#!/bin/bash

# ============================================================================
# BizGen Image Generation Pipeline
# ============================================================================
# This script only runs BizGen image generation step
# (assumes narrator format JSON files already exist)
#
# Usage:
#   bash scripts/run_bizgen_only.sh <GPU_ID> <START_FILE_IDX> <END_FILE_IDX> [DATASET_NAME]
#
# Example:
#   bash scripts/run_bizgen_only.sh 0 1 201 "squad_v2"       # GPU 0: files 1-200 (10,000 images), dataset "squad_v2"
#   bash scripts/run_bizgen_only.sh 1 201 401 "squad_v2"     # GPU 1: files 201-400 (10,000 images), dataset "squad_v2"
#   bash scripts/run_bizgen_only.sh 2 401 601 "my_data"      # GPU 2: files 401-600 (10,000 images), dataset "my_data"
#
# Note: Each file contains 50 images. File indices are 1-based.
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    echo "Usage: $0 <GPU_ID> <START_FILE_IDX> <END_FILE_IDX> [DATASET_NAME]"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID        : GPU device ID (0, 1, 2, ...)"
    echo "  START_FILE_IDX: Start file index (inclusive, 1-based)"
    echo "  END_FILE_IDX  : End file index (exclusive, 1-based)"
    echo "  DATASET_NAME  : Optional dataset name (default: squad_v2)"
    echo ""
    echo "Note: Each file contains 50 images. Use multiples of 50 for image counts."
    echo ""
    echo "Example:"
    echo "  $0 0 1 201               # GPU 0: files 1-200 (10,000 images) with default dataset"
    echo "  $0 1 201 401             # GPU 1: files 201-400 (10,000 images) with default dataset"
    echo "  $0 2 401 601 my_data     # GPU 2: files 401-600 (10,000 images) with custom dataset"
    echo ""
    exit 1
fi

GPU_ID=$1
START_FILE_IDX=$2
END_FILE_IDX=$3
DATASET_NAME=${4:-"squad_v2"}  # Default to squad_v2 if not provided

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
export PYTHONPATH="./:$PYTHONPATH"

# Conda environment path
CONDA_BIZGEN="/opt/miniconda3/envs/khoina_bizgen/bin/python"

# Paths
NARRATOR_FORMAT_DIR="src/data/create_data/output/narrator_format"
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"

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
echo "BizGen Image Generation - GPU $GPU_ID"
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

for ((i=START_FILE_IDX; i<END_FILE_IDX; i++)); do
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
    echo "  bash scripts/run_narrator_pipeline.sh $GPU_ID $START_FILE_IDX $END_FILE_IDX"
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
echo "  Processed File Range: [$START_FILE_IDX, $END_FILE_IDX)"
echo "  Number of Files     : $NUM_FILES"
echo "  Number of Images    : $NUM_IMAGES"
echo "  Dataset Name        : $DATASET_NAME"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Output Locations:"
echo "  Generated Images    : $BIZGEN_DIR/output/$DATASET_NAME/narrator*/"
echo "  Source Files        : $NARRATOR_FORMAT_DIR/wiki$(printf '%06d' $START_FILE_IDX).json - wiki$(printf '%06d' $((END_FILE_IDX-1))).json"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"