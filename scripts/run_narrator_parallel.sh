#!/bin/bash

# ============================================================================
# Parallel Narrator Pipeline Launcher
# ============================================================================
# This script launches the narrator pipeline on multiple GPUs in parallel
#
# Usage:
#   bash scripts/run_narrator_parallel.sh
#
# Configuration:
#   Edit the GPU_CONFIGS array below to set ranges for each GPU
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

# Define ranges for each GPU
# Format: "GPU_ID START_FILE_IDX END_FILE_IDX"
# Note: Each file contains 50 images
GPU_CONFIGS=(
    "0 1 201"       # GPU 0: files 1-200 (10,000 images)
    "1 201 401"     # GPU 1: files 201-400 (10,000 images)  
    "2 401 601"     # GPU 2: files 401-600 (10,000 images)
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
    num_files=$((end - start))
    num_images=$((num_files * 50))
    echo "  GPU $gpu: files [$start, $end) → $num_files files ($num_images images)"
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
    
    LOG_FILE="$LOG_DIR/gpu${gpu}_files_${start}_${end}_${TIMESTAMP}.log"
    LOG_FILES+=("$LOG_FILE")
    
    echo "Launching GPU $gpu: files [$start, $end)"
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
    echo "  GPU $gpu (PID ${PIDS[$i]}): files [$start, $end)"
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
    TOTAL_FILES=0
    TOTAL_IMAGES=0
    for config in "${GPU_CONFIGS[@]}"; do
        read -r gpu start end <<< "$config"
        num_files=$((end - start))
        num_images=$((num_files * 50))
        TOTAL_FILES=$((TOTAL_FILES + num_files))
        TOTAL_IMAGES=$((TOTAL_IMAGES + num_images))
    done
    
    echo "Summary:"
    echo "  Total Files Generated  : $TOTAL_FILES"
    echo "  Total Images Generated : $TOTAL_IMAGES"
    echo "  Number of GPUs Used    : ${#GPU_CONFIGS[@]}"
    echo ""
    echo "Output Locations:"
    echo "  Infographic JSON  : src/data/create_data/output/infographic/"
    echo "  Narrator Format   : src/data/create_data/output/narrator_format/"
    echo "  Generated Images  : src/data/bizgen/output/"
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
