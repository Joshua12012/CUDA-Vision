#!/usr/bin/env bash
################################################################################
# CUDA-Vision: High-Throughput GPU & Heterogeneous Vision Pipeline
# Author: Joshua Deniese
# Coursera GPU Specialization Capstone Project
#
# Automated build, execution, and artifact generation script.
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "   CUDA-VISION: AUTOMATED CAPSTONE RUNNER & ARTIFACT GENERATOR        "
echo "   Author: Joshua Deniese                                             "
echo "======================================================================"

# Ensure directories exist
mkdir -p bin data/input data/output artifacts/logs artifacts/data artifacts/images

# Build binary if not already built or if rebuild requested
if [ ! -f "bin/cuda_vision" ] || [ "$1" == "--rebuild" ]; then
    echo ">>> Building project with Makefile..."
    make all
fi

# If arguments were provided by user, forward directly to executable
if [ $# -gt 0 ] && [ "$1" != "--rebuild" ]; then
    echo ">>> Forwarding arguments to bin/cuda_vision: $@"
    ./bin/cuda_vision "$@"
    exit 0
fi

# Default: Run full capstone pipeline
echo ""
echo ">>> Step 1/3: Running Image and Batched Signal Processing Pipeline..."
./bin/cuda_vision --filter all --save-all

echo ""
echo ">>> Step 2/3: Running Multi-Scale Scaling Benchmarks..."
./bin/cuda_vision --benchmark

echo ""
echo ">>> Step 3/3: Packaging Proof of Execution Artifacts..."
make package

echo ""
echo "======================================================================"
echo "   CAPSTONE EXECUTION AND PACKAGING COMPLETED SUCCESSFULLY!           "
echo "======================================================================"
echo "Generated Artifacts:"
echo "  - Proof Archive:   proof_of_execution.tar.gz"
echo "  - Processed Images: artifacts/images/ (Before/After & Filter Stages)"
echo "  - CSV Data:        artifacts/data/ (Raw & Filtered Signals, Benchmarks)"
echo "  - Execution Logs:  artifacts/logs/execution_run.log"
echo ""
echo "Ready for Coursera Peer Review submission!"
