#!/bin/bash

# ============================================================================
# Parallel BizGen Image Generation Launcher
# ============================================================================
# This script launches BizGen image generation on multiple GPUs in parallel
#
# Usage:
#   bash scripts/run_bizgen_parallel.sh
#
# Configuration:
#   Edit the GPU_CONFIGS array below to set ranges for each GPU
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

# Define ranges for each GPU
# Format: "GPU_ID START_FILE END_FILE"
# Note: Each file contains 50 infographics
# Example: "0 1 21" means GPU 0 processes files 1-20 (1000 infographics)
GPU_CONFIGS=(
    "0 1 2"     # GPU 0: files 1-1 (50 infographics)
    "2 2 3"     # GPU 2: files 2-2 (50 infographics)
)

# Log directory
LOG_DIR="logs/bizgen"
mkdir -p "$LOG_DIR"

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "Parallel BizGen Image Generation Launcher"
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
    num_infographics=$((num_files * 50))
    first_id=$(((start - 1) * 50 + 1))
    last_id=$(((end - 1) * 50))
    echo "  GPU $gpu: files [$start, $end) → $num_files files ($num_infographics infographics, IDs $first_id-$last_id)"
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
    
    # Run the BizGen script in background
    bash scripts/run_bizgen.sh $gpu $start $end > "$LOG_FILE" 2>&1 &
    
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
    TOTAL_INFOGRAPHICS=0
    for config in "${GPU_CONFIGS[@]}"; do
        read -r gpu start end <<< "$config"
        num_files=$((end - start))
        num_infographics=$((num_files * 50))
        TOTAL_FILES=$((TOTAL_FILES + num_files))
        TOTAL_INFOGRAPHICS=$((TOTAL_INFOGRAPHICS + num_infographics))
    done
    
    echo "Summary:"
    echo "  Total Files Processed     : $TOTAL_FILES"
    echo "  Total Infographics        : $TOTAL_INFOGRAPHICS"
    echo "  Number of GPUs Used       : ${#GPU_CONFIGS[@]}"
    echo ""
    echo "Output Location:"
    echo "  Generated Images: src/data/bizgen/output/squad_v2/"
    echo ""
    echo "Next Steps:"
    echo "  1. Verify generated images in: src/data/bizgen/output/squad_v2/"
    echo "  2. Run OCR filtering (optional): python src/data/ocr/ocr_filter.py"
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
