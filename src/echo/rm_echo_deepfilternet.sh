#!/bin/bash

# conda activate captions
# pip install torch==2.2.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
# pip install deepfilternet

dirUser="/home/sameer"
dirProj="$dirUser/Shared/Sync/Private/Work/Projects/qadrai"

# --- CPU and Environment Management Configuration ---

# Function to get the number of logical cores (using nproc or an alternative)
get_num_cores() {
    # Check for 'nproc' (Linux systems)
    if command -v nproc &> /dev/null; then
        nproc
    # Check for 'sysctl' (macOS/BSD)
    elif command -v sysctl &> /dev/null; then
        sysctl -n hw.ncpu
    else
        # Fallback to a safe minimum if command is not found
        echo 2
    fi
}

# 1. Determine total cores
TOTAL_CORES=$(get_num_cores)

# 2. Calculate the number of cores to use (half of the total, minimum 1)
# Bash arithmetic truncates, which is fine for integer division.
CORES_TO_USE=$(( TOTAL_CORES / 2 ))

# Ensure we use at least 1 core
if [ "$CORES_TO_USE" -lt 1 ]; then
    CORES_TO_USE=1
fi

echo "Total CPU Cores Detected: $TOTAL_CORES"
echo "Limiting script to: $CORES_TO_USE threads (Half of available cores)"

# --- Critical Environment Variables for PyTorch/TensorFlow Stability ---

# A. Limit intra-operation parallelism (OpenMP/MKL, primary control for PyTorch/NumPy)
export OMP_NUM_THREADS=$CORES_TO_USE
export MKL_NUM_THREADS=$CORES_TO_USE
export OPENBLAS_NUM_THREADS=$CORES_TO_USE
export NUMEXPR_NUM_THREADS=$CORES_TO_USE

# B. Disable aggressive thread binding (can help prevent crashes in nested parallel regions)
export KMP_AFFINITY=disabled

# C. Force CPU-only execution (Crucial for stability in environments where CUDA is available but not desired)
# This prevents PyTorch/TensorFlow from crashing due to failed GPU device initialization.
export CUDA_VISIBLE_DEVICES=""

# This command processes the file and handles reverb/noise automatically
deepFilter /run/user/1000/gvfs/smb-share:server=latitude.local,share=shared/viz/mine/aajrumiyyah/SharhAjroomiya/01.mp3 -o "$dirProj/output/deepfilternet"
