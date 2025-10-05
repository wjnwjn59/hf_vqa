TOTAL_IMAGES=232000
NUM_GPUS=9
IMAGES_PER_GPU=$((TOTAL_IMAGES / NUM_GPUS))

echo "Total images: $TOTAL_IMAGES"
echo "Number of GPUs: $NUM_GPUS" 
echo "Images per GPU: $IMAGES_PER_GPU"

# Launch parallel processes for each GPU
for gpu_id in {0..8}; do
    START_IDX=$((gpu_id * IMAGES_PER_GPU))
    END_IDX=$(((gpu_id + 1) * IMAGES_PER_GPU))
    
    # For the last GPU, process remaining images
    if [ $gpu_id -eq 8 ]; then
        END_IDX=$TOTAL_IMAGES
    fi
    
    echo "GPU $gpu_id: Processing images $START_IDX to $((END_IDX-1))"
    
    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python src/data/create_data/qwen/merge_infographic_bboxes.py \
        --infographic-dir "src/data/create_data/output/infographic" \
        --start-wiki $START_IDX \
        --end-wiki $END_IDX \
        --seed $((42 + gpu_id)) \
        > "logs/merge_gpu${gpu_id}.log" 2>&1 &
done

# Wait for all background processes to complete
wait

echo "All GPU processes completed!"

# Count total output files
OUTPUT_DIR="src/data/create_data/output/bizgen_format"
FILE_COUNT=$(ls -1 $OUTPUT_DIR/wiki*.json 2>/dev/null | wc -l)
echo "Total wiki files generated: $FILE_COUNT"

# Calculate expected file count (50 images per file)
EXPECTED_FILES=$((TOTAL_IMAGES / 50))
echo "Expected files: $EXPECTED_FILES"

if [ $FILE_COUNT -eq $EXPECTED_FILES ]; then
    echo "Success: Generated expected number of files"
else
    echo "Warning: File count mismatch"
fi