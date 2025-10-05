#!/bin/bash

# Script to run generate_infographic_data.py on 9 GPUs in parallel
# Assumes total of 232000 Wikipedia entries to be processed

TOTAL_ENTRIES=232000
NUM_GPUS=9
ENTRIES_PER_GPU=$((TOTAL_ENTRIES / NUM_GPUS))

echo "Total Wikipedia entries: $TOTAL_ENTRIES"
echo "Number of GPUs: $NUM_GPUS" 
echo "Entries per GPU: $ENTRIES_PER_GPU"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set PYTHONPATH for proper imports
export PYTHONPATH="./:$PYTHONPATH"

# Launch parallel processes for each GPU
for gpu_id in {0..8}; do
    START_IDX=$((gpu_id * ENTRIES_PER_GPU))
    END_IDX=$(((gpu_id + 1) * ENTRIES_PER_GPU))
    
    # For the last GPU, process remaining entries
    if [ $gpu_id -eq 8 ]; then
        END_IDX=$TOTAL_ENTRIES
    fi
    
    echo "GPU $gpu_id: Processing Wikipedia entries $START_IDX to $((END_IDX-1))"
    
    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python src/data/create_data/qwen/generate_infographic_data.py \
        --model_name "unsloth/Qwen3-8B" \
        --input_data "/home/thinhnp/hf_vqa/src/data/create_data/wikipedia/wikipedia_processed.json" \
        --template_path "/home/thinhnp/hf_vqa/src/prompts/bizgen.jinja" \
        --start-wiki $START_IDX \
        --end-wiki $END_IDX \
        --batch_size 8 \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_tokens 8192 \
        --gpu_memory_utilization 0.9 \
        > "logs/generate_gpu${gpu_id}.log" 2>&1 &
done

# Wait for all background processes to complete
wait

echo "All GPU processes completed!"

# Count total output files
OUTPUT_DIR="src/data/create_data/output/infographic"
FILE_COUNT=$(ls -1 $OUTPUT_DIR/infographic*.json 2>/dev/null | wc -l)
echo "Total infographic files generated: $FILE_COUNT"

# Calculate expected file count (50 entries per file)
EXPECTED_FILES=$((TOTAL_ENTRIES / 50))
echo "Expected files: $EXPECTED_FILES"

if [ $FILE_COUNT -eq $EXPECTED_FILES ]; then
    echo "✅ Success: Generated expected number of files"
else
    echo "⚠️  Warning: File count mismatch"
fi

# Show file size distribution
echo -e "\nFile size distribution (first 10 files):"
ls -lah $OUTPUT_DIR/infographic*.json 2>/dev/null | head -10