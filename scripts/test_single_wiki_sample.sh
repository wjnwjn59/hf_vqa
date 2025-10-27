#!/bin/bash

# ============================================================================
# Test Single Wiki Sample - Quick Test
# ============================================================================
# This script tests the wiki bbox pipeline with just 1 sample for debugging

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
export PYTHONPATH="./:$PYTHONPATH"

# Conda environment
CONDA_WIKI="/opt/miniconda3/envs/thinh_wiki/bin/python"

# Test parameters
GPU_ID=2
START_SUBSET=1
END_SUBSET=2
NUM_SAMPLES=10  # Only 1 sample for quick test

# Paths
SQUAD_TRAIN="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl"
STAGE_A_TEMPLATE="./src/prompts/content_des_stage_1.jinja"
STAGE_B_TEMPLATE="./src/prompts/content_des_stage_2.jinja"
STAGE_C_TEMPLATE="./src/prompts/content_des_stage_3_with_bbox.jinja"
EXTRACTED_BBOXES="./src/data/narrator/extracted_bboxes.json"
TEST_OUTPUT_DIR="test_wiki_single"

# Model settings
MODEL_NAME="unsloth/Qwen3-8B"

echo "======================================================================"
echo "Testing Single Wiki Sample"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  GPU ID              : $GPU_ID"
echo "  Number of Samples   : $NUM_SAMPLES"
echo "  Output Directory    : $TEST_OUTPUT_DIR"
echo "  Model               : $MODEL_NAME"
echo ""

# Clean previous test results
if [ -d "$TEST_OUTPUT_DIR" ]; then
    echo "Cleaning previous test results..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

echo "Starting single sample test..."
echo ""

# ============================================================================
# Run the test
# ============================================================================
CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_WIKI src/data/narrator/generate_narrator_with_bbox.py \
    --model_name "$MODEL_NAME" \
    --input_data "$SQUAD_TRAIN" \
    --stage_a "$STAGE_A_TEMPLATE" \
    --stage_b "$STAGE_B_TEMPLATE" \
    --stage_c "$STAGE_C_TEMPLATE" \
    --extracted_bboxes "$EXTRACTED_BBOXES" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --start $START_SUBSET \
    --end $END_SUBSET \
    --num_samples $NUM_SAMPLES \
    --batch_size 1 \
    --temperature 0.7 \
    --max_retries 1 \
    --gpu_memory_utilization 0.9

# ============================================================================
# Check Results
# ============================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Single sample test completed!"
    
    # Check if output directory exists
    if [ -d "$TEST_OUTPUT_DIR" ]; then
        echo ""
        echo "Checking output files..."
        
        # Count wiki files
        WIKI_FILES=$(find "$TEST_OUTPUT_DIR" -name "wiki*.json" | wc -l)
        echo "  Wiki files generated: $WIKI_FILES"
        
        if [ $WIKI_FILES -gt 0 ]; then
            # Get the first wiki file
            FIRST_WIKI=$(find "$TEST_OUTPUT_DIR" -name "wiki*.json" | head -1)
            echo "  First wiki file: $FIRST_WIKI"
            
            # Display file size
            FILE_SIZE=$(du -h "$FIRST_WIKI" | cut -f1)
            echo "  File size: $FILE_SIZE"
            
            echo ""
            echo "======================================================================"
            echo "Analyzing Wiki Structure"
            echo "======================================================================"
            
            # Analyze the wiki structure
            python3 -c "
import json
import sys

try:
    with open('$FIRST_WIKI', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'File format: {type(data)}')
    
    if isinstance(data, list):
        print(f'Number of wiki entries: {len(data)}')
        
        if len(data) > 0:
            wiki_entry = data[0]
            print(f'\\nFirst wiki entry structure:')
            print(f'  Keys: {list(wiki_entry.keys())}')
            
            if 'index' in wiki_entry:
                print(f'  Wiki index: {wiki_entry[\"index\"]}')
            
            if 'layers_all' in wiki_entry:
                layers = wiki_entry['layers_all']
                print(f'  Number of layers: {len(layers)}')
                
                print(f'  Layer breakdown:')
                for i, layer in enumerate(layers):
                    category = layer.get('category', 'unknown')
                    caption = layer.get('caption', '')[:50] + '...' if len(layer.get('caption', '')) > 50 else layer.get('caption', '')
                    print(f'    Layer {i}: {category} - \"{caption}\"')
            
            if 'full_image_caption' in wiki_entry:
                caption = wiki_entry['full_image_caption']
                print(f'\\n  Full image caption length: {len(caption)} characters')
                print(f'  Caption preview: {caption[:100]}...')
            
            if 'layout_quality' in wiki_entry:
                quality = wiki_entry['layout_quality']
                print(f'\\n  Layout quality:')
                print(f'    Quality score: {quality.get(\"quality_score\", \"N/A\")}')
                print(f'    Passes quality: {quality.get(\"passes_quality\", \"N/A\")}')
            
            print(f'\\n✓ Wiki structure is correct!')
            
        else:
            print('✗ Empty wiki file')
            sys.exit(1)
    else:
        print('✗ Invalid wiki file format')
        sys.exit(1)
        
except Exception as e:
    print(f'✗ Error analyzing wiki file: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "======================================================================"
                echo "✓ Single Sample Test PASSED! 🎉"
                echo "======================================================================"
                echo ""
                echo "The wiki bbox pipeline is working correctly!"
                echo ""
                echo "Output file: $FIRST_WIKI"
                echo ""
                echo "Next steps:"
                echo "  1. Run larger test with more samples"
                echo "  2. Run full pipeline: bash scripts/run_narrator_bbox_pipeline.sh 0 1 2"
                echo "  3. Use generated wiki files with BizGen"
                echo ""
            else
                echo ""
                echo "✗ Wiki structure analysis failed"
                exit 1
            fi
        else
            echo "  ✗ No wiki files were generated"
            exit 1
        fi
    else
        echo "  ✗ Output directory was not created"
        exit 1
    fi
else
    echo ""
    echo "✗ Single sample test failed!"
    exit 1
fi

echo "======================================================================"
echo "Test completed!"
echo "======================================================================"