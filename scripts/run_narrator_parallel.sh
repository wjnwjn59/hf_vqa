#!/bin/bash

# ============================================================================
# Parallel Narrator Pipeline Launcher
# ============================================================================
# This script launches the complete narrator pipeline on multiple GPUs in parallel:
# 1. Generate 3-stage infographic data using Qwen
# 2. Merge bounding boxes 
# 3. Generate images with BizGen
# 4. Create training datasets (JSON/JSONL format)
#
# Usage:
#   bash scripts/run_narrator_parallel.sh
#
# Configuration:
#   Edit the GPU_CONFIGS array below to set ranges for each GPU
#
# Post-processing:
#   After completion, run: python scripts/merge_training_data.py
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

# Define ranges for each GPU
# Format: "GPU_ID START_SUBSET END_SUBSET"
# Note: Each subset contains 50 images
GPU_CONFIGS=(
    # "0 1 20"
    "1 1 20"
    "2 20 40"
)

# Log directory
LOG_DIR="logs/narrator_pipeline"
mkdir -p "$LOG_DIR"

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "Parallel Narrator Pipeline Launcher"
echo "======================================================================"
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Log Directory: $LOG_DIR"
echo ""
echo "GPU Configuration:"
echo "------------------"
for config in "${GPU_CONFIGS[@]}"; do
    read -r gpu start end <<< "$config"
    num_subsets=$((end - start))
    num_images=$((num_subsets * 50))
    echo "  GPU $gpu: subsets [$start, $end) → $num_subsets subsets ($num_images images)"
done
echo ""
echo "======================================================================"
echo ""

# ============================================================================
# Launch Jobs
# ============================================================================
PIDS=()
LOG_FILES=()

for config in "${GPU_CONFIGS[@]}"; do
    read -r gpu start end <<< "$config"
    
    LOG_FILE="$LOG_DIR/gpu${gpu}_subsets_${start}_${end}_${TIMESTAMP}.log"
    LOG_FILES+=("$LOG_FILE")
    
    echo "Launching GPU $gpu: subsets [$start, $end)"
    echo "  Log file: $LOG_FILE"
    
    # Run the pipeline script in background
    bash scripts/run_narrator_pipeline.sh $gpu $start $end > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    
    echo "  PID: $PID"
    echo ""
    
    # Small delay to avoid race conditions
    sleep 2
done

echo "======================================================================"
echo "All jobs launched!"
echo "======================================================================"
echo ""
echo "Running processes:"
for i in "${!PIDS[@]}"; do
    read -r gpu start end <<< "${GPU_CONFIGS[$i]}"
    echo "  GPU $gpu (PID ${PIDS[$i]}): subsets [$start, $end)"
done
echo ""
echo "Log files:"
for log in "${LOG_FILES[@]}"; do
    echo "  - $log"
done
echo ""
echo "======================================================================"
echo ""
echo "Monitoring progress..."
echo "You can tail the logs with:"
echo "  tail -f $LOG_DIR/*_${TIMESTAMP}.log"
echo ""
echo "Or monitor individual GPUs:"
for i in "${!PIDS[@]}"; do
    read -r gpu start end <<< "${GPU_CONFIGS[$i]}"
    echo "  GPU $gpu: tail -f ${LOG_FILES[$i]}"
done
echo ""
echo "======================================================================"

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo "Waiting for all jobs to complete..."
echo "(Press Ctrl+C to cancel all jobs)"
echo ""

# Trap SIGINT to kill all child processes
trap 'echo ""; echo "Caught Ctrl+C, killing all jobs..."; for pid in "${PIDS[@]}"; do kill $pid 2>/dev/null || true; done; exit 1' INT

FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    read -r gpu start end <<< "${GPU_CONFIGS[$i]}"
    
    echo "Waiting for GPU $gpu (PID $PID)..."
    
    if wait $PID; then
        echo "  ✓ GPU $gpu completed successfully"
    else
        echo "  ✗ GPU $gpu failed (check ${LOG_FILES[$i]})"
        FAILED=$((FAILED + 1))
    fi
done

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "All Jobs Completed!"
echo "======================================================================"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All GPUs completed successfully!"
    echo ""
    
    # Calculate totals
    TOTAL_SUBSETS=0
    TOTAL_IMAGES=0
    for config in "${GPU_CONFIGS[@]}"; do
        read -r gpu start end <<< "$config"
        num_subsets=$((end - start))
        num_images=$((num_subsets * 50))
        TOTAL_SUBSETS=$((TOTAL_SUBSETS + num_subsets))
        TOTAL_IMAGES=$((TOTAL_IMAGES + num_images))
    done
    
    echo "Summary:"
    echo "  Total Subsets Generated: $TOTAL_SUBSETS"
    echo "  Total Images Generated : $TOTAL_IMAGES"
    echo "  Number of GPUs Used    : ${#GPU_CONFIGS[@]}"
    echo ""
    echo "Output Locations:"
    echo "  Infographic v2 JSON: src/data/create_data/output/infographic_v2/"
    echo "  Narrator Format v2 : src/data/create_data/output/narrator_format_v2/"
    echo "  Generated Images   : src/data/bizgen/output/squad_v2_new/"
    echo "  Training Data      : training_data/narrator_*_training.json"
    echo ""
    echo "Next Steps:"
    echo "  1. Merge all training data files: python scripts/merge_training_data.py"
    echo "  2. Run OCR filtering: python src/data/ocr/ocr_filter.py"
    echo "  3. Train VQA model with generated data"
else
    echo "✗ $FAILED GPU(s) failed"
    echo ""
    echo "Check the log files for details:"
    for log in "${LOG_FILES[@]}"; do
        echo "  - $log"
    done
    exit 1
fi

echo ""
echo "Log files saved in: $LOG_DIR"
echo ""
echo "======================================================================"
echo "All Done! 🎉"
echo "======================================================================"