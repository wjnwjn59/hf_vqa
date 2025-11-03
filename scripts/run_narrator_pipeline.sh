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
    echo "  $0 0 1 3      # GPU 0: files 1-2 (infographic000001.json, infographic000002.json = 100 infographics)"
    echo "  $0 1 3 6      # GPU 1: files 3-5 (infographic000003.json - infographic000005.json = 150 infographics)"
    echo ""
    exit 1
fi

GPU_ID=$1
START_FILE=$2
END_FILE=$3

# ============================================================================
# Configuration
# ============================================================================
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Conda environment paths
CONDA_WIKI="/opt/miniconda3/envs/thinh_wiki/bin/python"
CONDA_BIZGEN="/opt/miniconda3/envs/bizgen/bin/python"

# Paths
SQUAD_TRAIN="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl"
TEMPLATE_PATH="./src/prompts/content_des_all.jinja"
INFOGRAPHIC_DIR="src/data/narrator/infographic"
NARRATOR_FORMAT_DIR="src/data/narrator/wiki"
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"

# Model settings
MODEL_NAME="unsloth/Qwen3-8B"
BATCH_SIZE=8
DATASET_TYPE="squad_v2"
SEED=42

# Calculate parameters based on file indices
NUM_FILES=$((END_FILE - START_FILE))
NUM_INFOGRAPHICS=$((NUM_FILES * 50))

# File indices for both generate and merge steps (same values)
GENERATE_START_FILE=$START_FILE
GENERATE_END_FILE=$END_FILE
MERGE_START_FILE=$START_FILE
MERGE_END_FILE=$END_FILE

# Calculate infographic ID range (corrected formula)
FIRST_INFOGRAPHIC_ID=$(((START_FILE - 1) * 50 + 1))
LAST_INFOGRAPHIC_ID=$(((END_FILE - 1) * 50))  # Changed: (END_FILE - 1) * 50 instead of END_FILE * 50

# Calculate subset for bizgen (using infographic IDs, exclusive end)
SUBSET="${GENERATE_START_FILE}:${GENERATE_END_FILE}"

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "Narrator Pipeline - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Start File Index    : $START_FILE"
echo "  End File Index      : $END_FILE (exclusive)"
echo "  Number of Files     : $NUM_FILES"
echo "  Total Infographics  : $NUM_INFOGRAPHICS"
echo "  Infographic ID Range: $FIRST_INFOGRAPHIC_ID - $LAST_INFOGRAPHIC_ID"
echo "  Expected Files      : infographic$(printf '%06d' $START_FILE).json - infographic$(printf '%06d' $((END_FILE - 1))).json"
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

export PYTHONPATH="./:$PYTHONPATH"

$CONDA_WIKI src/data/narrator/generate_infographic_data.py \
    --backend qwen \
    --model-name "$MODEL_NAME" \
    --input-data "$SQUAD_TRAIN" \
    --template-path "$TEMPLATE_PATH" \
    --dataset-type "$DATASET_TYPE" \
    --output-dir "$INFOGRAPHIC_DIR" \
    --start $GENERATE_START_FILE \
    --end $GENERATE_END_FILE \
    --batch-size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 1 completed successfully!"
    echo "  Generated files: infographic$(printf '%06d' $START_FILE).json - infographic$(printf '%06d' $((END_FILE - 1))).json"
    echo "  Infographic IDs: $FIRST_INFOGRAPHIC_ID - $LAST_INFOGRAPHIC_ID"
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
echo "Using file range: $MERGE_START_FILE to $MERGE_END_FILE (exclusive)"
echo ""

python src/data/narrator/merge_narrator_bboxes.py \
    --infographic-dir "$INFOGRAPHIC_DIR" \
    --output-dir "$NARRATOR_FORMAT_DIR" \
    --squad-file "$SQUAD_TRAIN" \
    --start $MERGE_START_FILE \
    --end $MERGE_END_FILE \
    --seed $SEED

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 2 completed successfully!"
    echo "  Generated wiki files: wiki$(printf '%06d' $MERGE_START_FILE).json - wiki$(printf '%06d' $((MERGE_END_FILE - 1))).json"
    echo "  Output directory: $NARRATOR_FORMAT_DIR"
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
echo "Using conda environment: bizgen"
echo "Python: $CONDA_BIZGEN"
echo "Using GPU: $GPU_ID"
echo "Subset: $SUBSET"
echo ""

cd "$BIZGEN_DIR"

$CONDA_BIZGEN inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "../narrator/wiki/" \
    --subset "$SUBSET" \
    --device "cuda:$GPU_ID" \
    --dataset_name "squad_v2"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 3 completed successfully!"
    
    # Calculate output directory based on dataset_name
    OUTPUT_DATASET_DIR="output/squad_v2"
    echo "  Generated images in: $BIZGEN_DIR/$OUTPUT_DATASET_DIR/"
    
    # Count generated images
    if [ -d "$OUTPUT_DATASET_DIR" ]; then
        NUM_GENERATED=$(find "$OUTPUT_DATASET_DIR" -name "*.png" -o -name "*.jpg" | wc -l)
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
echo "  Processed Files     : [$START_FILE, $END_FILE)"
echo "  Number of Files     : $NUM_FILES"
echo "  Total Infographics  : $NUM_INFOGRAPHICS"
echo "  Infographic ID Range: $FIRST_INFOGRAPHIC_ID - $LAST_INFOGRAPHIC_ID"
echo "  BizGen Subset       : $SUBSET"
echo ""
echo "Output Locations:"
echo "  1. Infographic JSON : $INFOGRAPHIC_DIR/infographic$(printf '%06d' $START_FILE).json - infographic$(printf '%06d' $((END_FILE - 1))).json"
echo "  2. Narrator Format  : $NARRATOR_FORMAT_DIR/wiki$(printf '%06d' $START_FILE).json - wiki$(printf '%06d' $((END_FILE - 1))).json"
echo "  3. Generated Images : $BIZGEN_DIR/output/squad_v2/narrator*/"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"
