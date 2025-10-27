#!/bin/bash

# ============================================================================
# Test Narrator BBox Pipeline - Generate Wiki Data and Images
# ============================================================================
# This script tests the complete pipeline:
# 1. Generate wiki data using generate_narrator_with_bbox.py
# 2. Generate images with BizGen
#
# Usage:
#   bash scripts/test_narrator_bbox_pipeline.sh <GPU_ID> <NUM_SAMPLES>
#
# Example:
#   bash scripts/test_narrator_bbox_pipeline.sh 0 10    # GPU 0: 10 samples
#   bash scripts/test_narrator_bbox_pipeline.sh 2 50    # GPU 2: 50 samples
#
# Note: Output folders will be prefixed with "test_" for easy identification
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================
if [ $# -ne 2 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    echo "Usage: $0 <GPU_ID> <NUM_SAMPLES>"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID      : GPU device ID (0, 1, 2, ...)"
    echo "  NUM_SAMPLES : Number of samples to generate (e.g., 10, 50, 100)"
    echo ""
    echo "Example:"
    echo "  $0 0 10     # GPU 0: generate 10 samples"
    echo "  $0 2 50     # GPU 2: generate 50 samples"
    echo ""
    exit 1
fi

GPU_ID=$1
NUM_SAMPLES=$2

# ============================================================================
# Validation
# ============================================================================
# Validate that GPU_ID is a non-negative integer
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU_ID must be a non-negative integer"
    exit 1
fi

# Validate that NUM_SAMPLES is a positive integer
if ! [[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_SAMPLES must be a positive integer >= 1"
    exit 1
fi

# ============================================================================
# Configuration
# ============================================================================
export PYTHONPATH="./:$PYTHONPATH"

# Conda environment paths
CONDA_WIKI="/opt/miniconda3/envs/thinh_wiki/bin/python"
CONDA_BIZGEN="/opt/miniconda3/envs/bizgen/bin/python"

# Input data
SQUAD_TRAIN="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl"

# Template paths
STAGE_A_TEMPLATE="./src/prompts/content_des_stage_1.jinja"
STAGE_B_TEMPLATE="./src/prompts/content_des_stage_2.jinja"
STAGE_C_TEMPLATE="./src/prompts/content_des_stage_3_with_bbox.jinja"
EXTRACTED_BBOXES="./src/data/narrator/extracted_bboxes.json"

# Test output directories (prefixed with "test_")
TEST_WIKI_DIR="/home/thinhnp/hf_vqa/src/data/create_data/output/test_wiki_data"
TEST_BIZGEN_DIR="./src/data/bizgen"
TEST_OUTPUT_NAME="test_narrator_bbox"

# Model settings
MODEL_NAME="unsloth/Qwen3-8B"
BATCH_SIZE=1
TEMPERATURE=0.7
MAX_RETRIES=2
GPU_MEMORY_UTIL=0.9

# BizGen settings
CKPT_DIR="checkpoints/lora/infographic"
DATASET_NAME="test_narrator_bbox"

# Calculate subset parameters for bbox generation
# We use subset 1 with the specified number of samples
START_SUBSET=1
END_SUBSET=2
BIZGEN_SUBSET="1:${NUM_SAMPLES}"

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "Test Narrator BBox Pipeline - GPU $GPU_ID"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Number of Samples   : $NUM_SAMPLES"
echo "  Model               : $MODEL_NAME"
echo "  Batch Size          : $BATCH_SIZE"
echo "  Temperature         : $TEMPERATURE"
echo "  Max Retries         : $MAX_RETRIES"
echo ""
echo "Paths:"
echo "  Input Dataset       : $SQUAD_TRAIN"
echo "  Stage 1 Template    : $STAGE_A_TEMPLATE"
echo "  Stage 2 Template    : $STAGE_B_TEMPLATE"
echo "  Stage 3 Template    : $STAGE_C_TEMPLATE"
echo "  Extracted BBoxes    : $EXTRACTED_BBOXES"
echo "  Wiki Output Dir     : $TEST_WIKI_DIR"
echo "  BizGen Directory    : $TEST_BIZGEN_DIR"
echo "  Dataset Name        : $DATASET_NAME"
echo ""
echo "======================================================================"
echo ""

# ============================================================================
# Clean Previous Test Results
# ============================================================================
echo "Cleaning previous test results..."

# Remove test wiki directory
if [ -d "$TEST_WIKI_DIR" ]; then
    echo "  Removing previous wiki data: $TEST_WIKI_DIR"
    rm -rf "$TEST_WIKI_DIR"
fi

# Remove test output from BizGen
BIZGEN_OUTPUT_DIR="$TEST_BIZGEN_DIR/output/$DATASET_NAME"
if [ -d "$BIZGEN_OUTPUT_DIR" ]; then
    echo "  Removing previous BizGen output: $BIZGEN_OUTPUT_DIR"
    rm -rf "$BIZGEN_OUTPUT_DIR"
fi

echo "  ✓ Cleanup completed"
echo ""

# ============================================================================
# Step 1: Generate Wiki Data using generate_narrator_with_bbox.py
# ============================================================================
echo "======================================================================"
echo "Step 1/2: Generating Wiki Data with BBox Matching"
echo "======================================================================"
echo "Using conda environment: thinh_wiki"
echo "Python: $CONDA_WIKI"
echo "Using GPU: $GPU_ID"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_WIKI src/data/narrator/generate_narrator_with_bbox.py \
    --model_name "$MODEL_NAME" \
    --input_data "$SQUAD_TRAIN" \
    --stage_a "$STAGE_A_TEMPLATE" \
    --stage_b "$STAGE_B_TEMPLATE" \
    --stage_c "$STAGE_C_TEMPLATE" \
    --extracted_bboxes "$EXTRACTED_BBOXES" \
    --output_dir "$TEST_WIKI_DIR" \
    --start $START_SUBSET \
    --end $END_SUBSET \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --max_retries $MAX_RETRIES \
    --gpu_memory_utilization $GPU_MEMORY_UTIL

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 1 completed successfully!"
    
    # Check generated wiki files
    if [ -d "$TEST_WIKI_DIR" ]; then
        WIKI_FILES=$(find "$TEST_WIKI_DIR" -name "wiki*.json" | wc -l)
        echo "  Generated wiki files: $WIKI_FILES"
        
        if [ $WIKI_FILES -gt 0 ]; then
            # Get total wiki entries
            TOTAL_ENTRIES=$(python3 -c "
import json
import glob
import os

total = 0
wiki_files = glob.glob('$TEST_WIKI_DIR/wiki*.json')
for wiki_file in wiki_files:
    try:
        with open(wiki_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                total += len(data)
            else:
                total += 1
    except:
        pass
print(total)
")
            echo "  Total wiki entries: $TOTAL_ENTRIES"
        else
            echo "  ✗ No wiki files were generated!"
            exit 1
        fi
    else
        echo "  ✗ Output directory was not created!"
        exit 1
    fi
else
    echo ""
    echo "✗ Step 1 failed!"
    exit 1
fi

# ============================================================================
# Step 2: Generate Images with BizGen
# ============================================================================
echo ""
echo "======================================================================"
echo "Step 2/2: Generating Images with BizGen"
echo "======================================================================"
echo "Using conda environment: bizgen"
echo "Python: $CONDA_BIZGEN"
echo "Using GPU: cuda:$GPU_ID"
echo "Subset: $BIZGEN_SUBSET"
echo ""

cd "$TEST_BIZGEN_DIR"

$CONDA_BIZGEN inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "$TEST_WIKI_DIR/" \
    --subset "$BIZGEN_SUBSET" \
    --dataset_name "$DATASET_NAME" \
    --device "cuda:$GPU_ID"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Step 2 completed successfully!"
    
    # Check generated images
    OUTPUT_DATASET_DIR="output/$DATASET_NAME"
    if [ -d "$OUTPUT_DATASET_DIR" ]; then
        NUM_IMAGES=$(find "$OUTPUT_DATASET_DIR" -name "*.png" | grep -v "bbox" | grep -v "lcfg" | wc -l)
        NUM_FOLDERS=$(find "$OUTPUT_DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  Generated images: $NUM_IMAGES"
        echo "  Generated folders: $NUM_FOLDERS"
        
        # Show some sample output files
        echo ""
        echo "Sample generated files:"
        find "$OUTPUT_DATASET_DIR" -name "*.png" | grep -v "bbox" | grep -v "lcfg" | head -5 | while read file; do
            echo "    $file"
        done
    else
        echo "  ✗ BizGen output directory was not created!"
        cd - > /dev/null
        exit 1
    fi
else
    echo ""
    echo "✗ Step 2 failed!"
    cd - > /dev/null
    exit 1
fi

cd - > /dev/null