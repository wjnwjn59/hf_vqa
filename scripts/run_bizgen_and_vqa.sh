#!/bin/bash

# Script to run BizGen inference and VQA generation for both easy and full datasets
# Usage: bash scripts/run_bizgen_and_vqa.sh

set -e  # Exit on error

# Configuration
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"

# Conda environment paths
CONDA_BIZGEN="/opt/miniconda3/envs/khoina_bizgen/bin/python"
CONDA_VQA="/opt/miniconda3/envs/thinh_wiki/bin/python"

# Paths
BIZGEN_DIR="./src/data/bizgen"
CKPT_DIR="checkpoints/lora/infographic"
WIKI_DIR_EASY="/home/thinhnp/hf_vqa/src/data/create_data/output/bizgen_format_easy"
WIKI_DIR_FULL="/home/thinhnp/hf_vqa/src/data/create_data/output/bizgen_format_full"
OUTPUT_DIR_EASY="output_easy"
OUTPUT_DIR_FULL="output_full"
VQA_SCRIPT="./src/data/vqa/generate_vqa_data.py"
TEMPLATE_PATH="./src/prompts/vqg.jinja"
MODEL_NAME="unsloth/Qwen2-VL-7B-Instruct"

# Parameters
SUBSET="0:516"
NUM_QUESTIONS=5
BATCH_SIZE=4
TEMPERATURE=0.7
TOP_P=0.9
MAX_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.9

echo "======================================================================"
echo "Starting BizGen Inference and VQA Generation Pipeline"
echo "======================================================================"
echo ""

# ==============================================================================
# Part 1: BizGen Inference - Easy Dataset
# ==============================================================================
echo "======================================================================"
echo "Part 1: Running BizGen Inference for EASY dataset"
echo "======================================================================"
echo "Using conda environment: khoina_bizgen"
echo "Python: $CONDA_BIZGEN"
echo "Input: $WIKI_DIR_EASY"
echo "Output: $BIZGEN_DIR/$OUTPUT_DIR_EASY"
echo "Subset: $SUBSET"
echo ""

cd "$BIZGEN_DIR"

$CONDA_BIZGEN inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "$WIKI_DIR_EASY" \
    --output_dir "$OUTPUT_DIR_EASY" \
    --subset "$SUBSET"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ BizGen inference for EASY dataset completed successfully!"
else
    echo ""
    echo "✗ BizGen inference for EASY dataset failed!"
    exit 1
fi

cd - > /dev/null

# ==============================================================================
# Part 2: BizGen Inference - Full Dataset
# ==============================================================================
echo ""
echo "======================================================================"
echo "Part 2: Running BizGen Inference for FULL dataset"
echo "======================================================================"
echo "Using conda environment: khoina_bizgen"
echo "Python: $CONDA_BIZGEN"
echo "Input: $WIKI_DIR_FULL"
echo "Output: $BIZGEN_DIR/$OUTPUT_DIR_FULL"
echo "Subset: $SUBSET"
echo ""

cd "$BIZGEN_DIR"

$CONDA_BIZGEN inference.py \
    --ckpt_dir "$CKPT_DIR" \
    --wiki_dir "$WIKI_DIR_FULL" \
    --output_dir "$OUTPUT_DIR_FULL" \
    --subset "$SUBSET"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ BizGen inference for FULL dataset completed successfully!"
else
    echo ""
    echo "✗ BizGen inference for FULL dataset failed!"
    exit 1
fi

cd - > /dev/null

# ==============================================================================
# Part 3: VQA Generation - Easy Dataset
# ==============================================================================
echo ""
echo "======================================================================"
echo "Part 3: Running VQA Generation for EASY dataset images"
echo "======================================================================"
echo "Switching to conda environment: thinh_wiki"
echo ""

# Get the generated images directory (assuming subset_0_516 format)
SUBSET_START=$(echo $SUBSET | cut -d':' -f1)
SUBSET_END=$(echo $SUBSET | cut -d':' -f2)
IMAGES_DIR_EASY="$BIZGEN_DIR/$OUTPUT_DIR_EASY/subset_${SUBSET_START}_${SUBSET_END}"
OUTPUT_VQA_EASY="$IMAGES_DIR_EASY/vqa_data.json"

echo "Images directory: $IMAGES_DIR_EASY"
echo "Output VQA file: $OUTPUT_VQA_EASY"
echo "Model: $MODEL_NAME"
echo "Questions per image: $NUM_QUESTIONS"
echo "Batch size: $BATCH_SIZE"
echo ""

$CONDA_VQA "$VQA_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --images_dir "$IMAGES_DIR_EASY" \
    --template_path "$TEMPLATE_PATH" \
    --output_path "$OUTPUT_VQA_EASY" \
    --num_questions "$NUM_QUESTIONS" \
    --batch_size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ VQA generation for EASY dataset completed successfully!"
else
    echo ""
    echo "✗ VQA generation for EASY dataset failed!"
    exit 1
fi

# ==============================================================================
# Part 4: VQA Generation - Full Dataset
# ==============================================================================
echo ""
echo "======================================================================"
echo "Part 4: Running VQA Generation for FULL dataset images"
echo "======================================================================"
echo "Using conda environment: thinh_wiki (already activated)"
echo ""

IMAGES_DIR_FULL="$BIZGEN_DIR/$OUTPUT_DIR_FULL/subset_${SUBSET_START}_${SUBSET_END}"
OUTPUT_VQA_FULL="$IMAGES_DIR_FULL/vqa_data.json"

echo "Images directory: $IMAGES_DIR_FULL"
echo "Output VQA file: $OUTPUT_VQA_FULL"
echo "Model: $MODEL_NAME"
echo "Questions per image: $NUM_QUESTIONS"
echo "Batch size: $BATCH_SIZE"
echo ""

$CONDA_VQA "$VQA_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --images_dir "$IMAGES_DIR_FULL" \
    --template_path "$TEMPLATE_PATH" \
    --output_path "$OUTPUT_VQA_FULL" \
    --num_questions "$NUM_QUESTIONS" \
    --batch_size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ VQA generation for FULL dataset completed successfully!"
else
    echo ""
    echo "✗ VQA generation for FULL dataset failed!"
    exit 1
fi

# ==============================================================================
# Final Summary
# ==============================================================================
echo ""
echo "======================================================================"
echo "Pipeline Completed Successfully!"
echo "======================================================================"
echo ""
echo "Results Summary:"
echo "----------------"
echo ""
echo "EASY Dataset:"
echo "  - Generated images: $IMAGES_DIR_EASY"
echo "  - VQA data: $OUTPUT_VQA_EASY"
echo "  - VQA summary: ${OUTPUT_VQA_EASY//.json/_summary.json}"
echo ""
echo "FULL Dataset:"
echo "  - Generated images: $IMAGES_DIR_FULL"
echo "  - VQA data: $OUTPUT_VQA_FULL"
echo "  - VQA summary: ${OUTPUT_VQA_FULL//.json/_summary.json}"
echo ""
echo "======================================================================"
