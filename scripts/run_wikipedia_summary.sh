#!/bin/bash

# Wikipedia Summary Generation Pipeline
# This script runs the complete pipeline for generating Wikipedia summaries

echo "=================================================="
echo "Wikipedia Summary Generation Pipeline"
echo "=================================================="

# Activate the thinh_wiki conda environment
echo "Activating thinh_wiki conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thinh_wiki

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate thinh_wiki environment"
    exit 1
fi

# Set default parameters
DATASET_PATH="${DATASET_PATH:-/home/thinhnp/hf_vqa/src/data/create_data/wikipedia/wikipedia_en_20231101}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-8B}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
START_WIKI="${START_WIKI:-0}"
END_WIKI="${END_WIKI:-}"

# Step 1: Extract full Wikipedia articles
echo ""
echo "Step 1: Extracting full Wikipedia articles..."
echo "=================================================="

python src/data/create_data/wikipedia/extract_wikipedia_full.py \
    --dataset_path "$DATASET_PATH" \
    --output_path src/data/create_data/wikipedia/wikipedia_full_processed.json \
    --min_words 500 \
    --max_samples "$MAX_SAMPLES" \
    --save_format json

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract Wikipedia articles"
    exit 1
fi

# Step 2: Generate summaries using Qwen
echo ""
echo "Step 2: Generating summaries using Qwen..."
echo "=================================================="

python src/data/create_data/qwen/generate_wikipedia_summary.py \
    --model_name "$MODEL_NAME" \
    --input_data src/data/create_data/wikipedia/wikipedia_full_processed.json \
    --template_path src/prompts/summary.jinja \
    --batch_size "$BATCH_SIZE" \
    --start-wiki "$START_WIKI" \
    ${END_WIKI:+--end-wiki "$END_WIKI"} \
    --output-dir src/data/create_data/output/summarize

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate summaries"
    exit 1
fi

echo ""
echo "=================================================="
echo "Pipeline completed successfully!"
echo "Check the output directory: src/data/create_data/output/summarize/"
echo "=================================================="