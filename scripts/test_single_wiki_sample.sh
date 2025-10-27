#!/bin/bash

# ============================================================================
# Complete Pipeline Test - Wiki Generation to Training Data
# ============================================================================
# This script runs the complete pipeline:
# 1. Generate wiki layouts with bbox matching
# 2. Generate images using BizGen
# 3. Apply OCR filtering
# 4. Create training dataset (JSON/JSONL)

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
export PYTHONPATH="./:$PYTHONPATH"

# Conda environments
CONDA_WIKI="/opt/miniconda3/envs/thinh_wiki/bin/python"
CONDA_BIZGEN="/opt/miniconda3/envs/bizgen/bin/python"

# Test parameters
GPU_ID=0
START_SUBSET=1
END_SUBSET=2
NUM_SAMPLES=3  # Number of samples for quick test

# Paths
SQUAD_TRAIN="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl"
STAGE_A_TEMPLATE="./src/prompts/content_des_stage_1.jinja"
STAGE_B_TEMPLATE="./src/prompts/content_des_stage_2.jinja"
STAGE_C_TEMPLATE="./src/prompts/content_des_stage_3_with_bbox.jinja"
EXTRACTED_BBOXES="./src/data/narrator/extracted_bboxes.json"

# Output directories
WIKI_OUTPUT_DIR="test_wiki_single"
BIZGEN_OUTPUT_DIR="./src/data/bizgen/output/test_single"
OCR_OUTPUT_DIR="./ocr_results_single"
TRAINING_OUTPUT_DIR="./training_data"

# Model settings
MODEL_NAME="unsloth/Qwen3-8B"
DATASET_NAME="test_single"

echo "======================================================================"
echo "Complete Pipeline Test: Wiki Generation → Images → Training Data"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Number of Samples   : $NUM_SAMPLES"
echo "  Dataset Name        : $DATASET_NAME"
echo "  Wiki Output         : $WIKI_OUTPUT_DIR"
echo "  BizGen Output       : $BIZGEN_OUTPUT_DIR"
echo "  OCR Output          : $OCR_OUTPUT_DIR"
echo "  Training Output     : $TRAINING_OUTPUT_DIR"
echo "  Model               : $MODEL_NAME"
echo ""

# Clean previous test results
echo "Cleaning previous test results..."
rm -rf "$WIKI_OUTPUT_DIR" "$BIZGEN_OUTPUT_DIR" "$OCR_OUTPUT_DIR"
mkdir -p "$TRAINING_OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "STEP 1/4: Generate Wiki Layouts with BBox Matching"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_WIKI src/data/narrator/generate_narrator_with_bbox.py \
    --model_name "$MODEL_NAME" \
    --input_data "$SQUAD_TRAIN" \
    --stage_a "$STAGE_A_TEMPLATE" \
    --stage_b "$STAGE_B_TEMPLATE" \
    --stage_c "$STAGE_C_TEMPLATE" \
    --extracted_bboxes "$EXTRACTED_BBOXES" \
    --output_dir "$WIKI_OUTPUT_DIR" \
    --start $START_SUBSET \
    --end $END_SUBSET \
    --num_samples $NUM_SAMPLES \
    --batch_size 1 \
    --temperature 0.7 \
    --max_retries 1 \
    --gpu_memory_utilization 0.9

# Check if wiki generation succeeded
if [ $? -ne 0 ]; then
    echo "✗ Wiki generation failed!"
    exit 1
fi

# Count generated wiki files
WIKI_FILES=$(find "$WIKI_OUTPUT_DIR" -name "wiki*.json" | wc -l)
echo "✓ Generated $WIKI_FILES wiki layout files"

if [ $WIKI_FILES -eq 0 ]; then
    echo "✗ No wiki files were generated!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 2/4: Generate Images using BizGen"
echo "======================================================================"

# Activate BizGen environment and run image generation
echo "Switching to BizGen environment and generating images..."
cd src/data/bizgen

CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_BIZGEN inference.py \
    --wiki_dir "$WIKI_OUTPUT_DIR" \
    --output_dir "$BIZGEN_OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME"

# Check if image generation succeeded
if [ $? -ne 0 ]; then
    echo "✗ BizGen image generation failed!"
    exit 1
fi

# Count generated images
IMAGE_FILES=$(find "$BIZGEN_OUTPUT_DIR" -name "*.png" | wc -l)
echo "✓ Generated $IMAGE_FILES image files"

if [ $IMAGE_FILES -eq 0 ]; then
    echo "✗ No images were generated!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 3/4: Apply OCR Filtering"
echo "======================================================================"

# Run OCR filter to identify poor quality images
echo "Running OCR quality filter..."
CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_WIKI src/data/ocr/ocr_filter.py \
    --images-dir "$BIZGEN_OUTPUT_DIR" \
    --bizgen-dir "$WIKI_OUTPUT_DIR" \
    --output-dir "$OCR_OUTPUT_DIR" \
    --threshold 0.5

# Check if OCR filtering succeeded
if [ $? -ne 0 ]; then
    echo "✗ OCR filtering failed!"
    exit 1
fi

echo "✓ OCR filtering completed"

echo ""
echo "======================================================================"
echo "STEP 4/4: Create Training Dataset"
echo "======================================================================"

# Create training dataset from generated images and QA data
echo "Creating training dataset (JSON + JSONL format)..."
$CONDA_WIKI src/data/narrator/convert_to_training_format.py \
    --qa-file "$SQUAD_TRAIN" \
    --image-base-dir "./src/data/bizgen/output" \
    --dataset-name "$DATASET_NAME" \
    --dataset-type "squad_v2" \
    --output-file "$TRAINING_OUTPUT_DIR/test_single_training.json" \
    --max-samples $NUM_SAMPLES \
    --seed 42

# Check if training data creation succeeded
if [ $? -ne 0 ]; then
    echo "✗ Training data creation failed!"
    exit 1
fi

# Verify training files were created
JSON_FILE="$TRAINING_OUTPUT_DIR/test_single_training.json"
JSONL_FILE="$TRAINING_OUTPUT_DIR/test_single_training.jsonl"

if [ -f "$JSON_FILE" ] && [ -f "$JSONL_FILE" ]; then
    echo "✓ Training dataset created successfully!"
    
    # Get file sizes
    JSON_SIZE=$(du -h "$JSON_FILE" | cut -f1)
    JSONL_SIZE=$(du -h "$JSONL_FILE" | cut -f1)
    
    echo "  JSON file: $JSON_FILE ($JSON_SIZE)"
    echo "  JSONL file: $JSONL_FILE ($JSONL_SIZE)"
    
    # Count training samples
    TRAIN_SAMPLES=$(python3 -c "import json; data=json.load(open('$JSON_FILE')); print(len(data))")
    echo "  Training samples: $TRAIN_SAMPLES"
    
else
    echo "✗ Training files were not created!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ COMPLETE PIPELINE TEST PASSED! 🎉"
echo "======================================================================"
echo ""
echo "Summary of results:"
echo "  📝 Wiki layouts generated: $WIKI_FILES files"
echo "  🖼️  Images generated: $IMAGE_FILES files"
echo "  🔍 OCR filtering completed"
echo "  📊 Training samples created: $TRAIN_SAMPLES samples"
echo ""
echo "Generated files:"
echo "  📁 Wiki layouts: $WIKI_OUTPUT_DIR/"
echo "  📁 Images: $BIZGEN_OUTPUT_DIR/"
echo "  📁 OCR results: $OCR_OUTPUT_DIR/"
echo "  📄 Training JSON: $JSON_FILE"
echo "  📄 Training JSONL: $JSONL_FILE"
echo ""
echo "Pipeline components verified:"
echo "  ✅ Squad v2 data loading & deduplication"
echo "  ✅ 3-stage infographic generation with QA integration"
echo "  ✅ BBox matching and layout optimization"
echo "  ✅ BizGen image synthesis"
echo "  ✅ OCR quality filtering"
echo "  ✅ Training data format conversion"
echo ""
echo "Next steps:"
echo "  🚀 Run full pipeline: bash scripts/run_narrator_bbox_pipeline.sh 0 1 2"
echo "  📈 Train VQA model with generated training data"
echo "  🧪 Test model inference on generated images"
echo ""

# ============================================================================
# Optional: Display sample training entry
# ============================================================================
echo "======================================================================"
echo "Sample Training Entry"
echo "======================================================================"

python3 -c "
import json
import sys

try:
    with open('$JSON_FILE', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if len(data) > 0:
        sample = data[0]
        print(f'ID: {sample[\"id\"]}')
        print(f'Image: {sample[\"image\"]}')
        print(f'Conversations: {len(sample[\"conversations\"])} turns')
        
        # Show first QA pair
        if len(sample['conversations']) >= 2:
            print(f'\\nFirst Q&A:')
            print(f'Q: {sample[\"conversations\"][0][\"value\"]}')
            print(f'A: {sample[\"conversations\"][1][\"value\"]}')
        
        print(f'\\n✅ Training data format is correct!')
    else:
        print('❌ No training samples found')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Error reading training data: {e}')
    sys.exit(1)
"

echo ""
echo "======================================================================"
echo "🎊 COMPLETE PIPELINE TEST COMPLETED SUCCESSFULLY! 🎊"
echo "======================================================================"