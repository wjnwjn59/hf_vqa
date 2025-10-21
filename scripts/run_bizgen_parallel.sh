#!/bin/bash

# ============================================================================
# Parallel BizGen Image Generation Launcher
# ============================================================================
# This script launches BizGen image generation on multiple GPUs in parallel
# (assumes narrator format JSON files already exist)
#
# Usage:
#   bash scripts/run_bizgen_parallel.sh [DATASET_NAME]
#
# Example:
#   bash scripts/run_bizgen_parallel.sh           # Use default dataset name "squad_v2"
#   bash scripts/run_bizgen_parallel.sh my_data  # Use custom dataset name "my_data"
#
# Configuration:
#   Edit the GPU_CONFIGS array below to set file ranges for each GPU
#   Note: Each file contains 50 images. File indices are 1-based.
# ============================================================================

set -e

# ============================================================================
# Parse Arguments
# ============================================================================
DATASET_NAME=${1:-"squad_v2"}  # Default to squad_v2 if not provided

# ============================================================================
# Configuration
# ============================================================================

# Define ranges for each GPU
# Format: "GPU_ID START_FILE_IDX END_FILE_IDX"
# Note: Each file contains 50 images
GPU_CONFIGS=(
    "0 83 90"     # GPU 0: files 1-200 (10,000 images)
    "1 91 100"     # GPU 1: files 201-400 (10,000 images)  
    # "2 90 100"    # GPU 2: files 401-600 (10,000 images)
)

# Log directory
LOG_DIR="logs/bizgen_only"
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
echo "Dataset Name: $DATASET_NAME"
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
# Check Prerequisites
# ============================================================================
echo "Checking prerequisites..."

NARRATOR_FORMAT_DIR="src/data/create_data/output/narrator_format"
if [ ! -d "$NARRATOR_FORMAT_DIR" ]; then
    echo "✗ Error: Narrator format directory not found: $NARRATOR_FORMAT_DIR"
    echo "  Please run the full pipeline first to generate narrator format files."
    exit 1
fi

# Check if any expected files exist
TOTAL_EXPECTED=0
TOTAL_MISSING=0

for config in "${GPU_CONFIGS[@]}"; do
    read -r gpu start end <<< "$config"
    
    for ((i=start; i<end; i++)); do
        WIKI_FILE="$NARRATOR_FORMAT_DIR/wiki$(printf '%06d' $i).json"
        TOTAL_EXPECTED=$((TOTAL_EXPECTED + 1))
        
        if [ ! -f "$WIKI_FILE" ]; then
            TOTAL_MISSING=$((TOTAL_MISSING + 1))
        fi
    done
done

if [ $TOTAL_MISSING -gt 0 ]; then
    echo "✗ Warning: $TOTAL_MISSING out of $TOTAL_EXPECTED expected files are missing."
    echo "  Some GPUs may fail. Consider running the full pipeline first."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ All $TOTAL_EXPECTED narrator format files found"
fi

echo ""

# ============================================================================
# Launch Jobs
# ============================================================================
PIDS=()
LOG_FILES=()

for config in "${GPU_CONFIGS[@]}"; do
    read -r gpu start end <<< "$config"
    
    LOG_FILE="$LOG_DIR/gpu${gpu}_${start}_${end}_${DATASET_NAME}_${TIMESTAMP}.log"
    LOG_FILES+=("$LOG_FILE")
    
    echo "Launching GPU $gpu: files [$start, $end) for dataset '$DATASET_NAME'"
    echo "  Log file: $LOG_FILE"
    
    # Run the BizGen-only script in background
    bash scripts/run_bizgen_only.sh $gpu $start $end "$DATASET_NAME" > "$LOG_FILE" 2>&1 &
    
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
echo "  tail -f $LOG_DIR/*_${DATASET_NAME}_${TIMESTAMP}.log"
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
    echo "  Total Files Used       : $TOTAL_FILES"
    echo "  Total Images Generated : $TOTAL_IMAGES"
    echo "  Number of GPUs Used    : ${#GPU_CONFIGS[@]}"
    echo "  Dataset Name           : $DATASET_NAME"
    echo ""
    echo "Output Location:"
    echo "  Generated Images : src/data/bizgen/output/$DATASET_NAME/"
    echo ""
    
    # Try to count actual generated images
    OUTPUT_DIR="src/data/bizgen/output/$DATASET_NAME"
    if [ -d "$OUTPUT_DIR" ]; then
        ACTUAL_IMAGES=$(find "$OUTPUT_DIR" -name "*.png" | grep -v "bbox" | grep -v "lcfg" | wc -l)
        ACTUAL_FOLDERS=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  Actual Images Generated: $ACTUAL_IMAGES"
        echo "  Actual Folders Created : $ACTUAL_FOLDERS"
    fi
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