################################################################################
# CUDA-Vision: High-Throughput GPU & Heterogeneous Vision Pipeline
# Author: Joshua Deniese
# Coursera GPU Specialization Capstone Project
################################################################################

# Compiler setup
CXX = g++
NVCC ?= $(shell which nvcc 2>/dev/null)
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wno-missing-field-initializers -fopenmp -Iinclude -Ilib
CUDAFLAGS = -O3 -std=c++17 -Xcompiler "-fopenmp -Wall -Wno-missing-field-initializers" -Iinclude -Ilib

# Target paths
BIN_DIR = bin
TARGET_CPU = $(BIN_DIR)/cuda_vision_cpu
TARGET_GPU = $(BIN_DIR)/cuda_vision_gpu
TARGET_DEFAULT = $(BIN_DIR)/cuda_vision

# Source files
CORE_SRCS = src/image_buffer.cpp src/signal_buffer.cpp src/cpu/cpu_engine.cpp src/benchmark.cpp
CUDA_SRCS = src/cuda/cuda_kernels.cu

.PHONY: all cpu gpu run benchmark test package clean help

# Auto-detect whether CUDA / nvcc is available
all:
	@mkdir -p $(BIN_DIR)
	@if [ -n "$(NVCC)" ] && [ -x "$(NVCC)" ]; then \
		echo ">>> CUDA compiler (nvcc) detected: $(NVCC). Building GPU target..."; \
		$(MAKE) gpu; \
	else \
		echo ">>> No CUDA compiler found. Building optimized Heterogeneous CPU target..."; \
		$(MAKE) cpu; \
	fi

# Build optimized CPU target with OpenMP & CUDA Grid/Block Simulation
cpu:
	@mkdir -p $(BIN_DIR)
	@echo ">>> Compiling CPU Heterogeneous Architecture..."
	$(CXX) $(CXXFLAGS) $(CORE_SRCS) src/main.cpp -o $(TARGET_CPU)
	@cp -f $(TARGET_CPU) $(TARGET_DEFAULT)
	@echo ">>> Successfully built $(TARGET_CPU) and $(TARGET_DEFAULT)!"

# Build native NVIDIA CUDA target with NVCC
gpu:
	@mkdir -p $(BIN_DIR)
	@if [ -z "$(NVCC)" ]; then \
		echo "Error: nvcc not found in PATH. Install CUDA Toolkit or run 'make cpu'."; \
		exit 1; \
	fi
	@echo ">>> Compiling CUDA GPU Target with $(NVCC)..."
	$(NVCC) $(CUDAFLAGS) -DCUDA_ENABLED $(CUDA_SRCS) $(CORE_SRCS) src/main.cpp -o $(TARGET_GPU) -lcudart
	@cp -f $(TARGET_GPU) $(TARGET_DEFAULT)
	@echo ">>> Successfully built $(TARGET_GPU) and $(TARGET_DEFAULT)!"

# Execute pipeline and generate artifacts
run: all
	@echo ">>> Executing Capstone Vision & Signal Pipeline..."
	./$(TARGET_DEFAULT) --filter all --save-all

# Execute multi-scale benchmark suite
benchmark: all
	@echo ">>> Running Multi-Scale Performance Benchmark Suite..."
	./$(TARGET_DEFAULT) --benchmark

# Run verification and correctness tests
test: all
	@echo ">>> Running Algorithm Equivalence and Verification Tests..."
	./$(TARGET_DEFAULT) --iterations 1 --save-all

# Package proof of execution artifacts for Coursera submission
package: run
	@echo ">>> Packaging Proof of Execution Artifacts into tar.gz and zip..."
	tar -czvf proof_of_execution.tar.gz data/input data/output artifacts
	@which zip >/dev/null 2>&1 && zip -r proof_of_execution.zip data/input data/output artifacts || true
	@echo ">>> Packaging complete: proof_of_execution.tar.gz ready for upload!"

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)/*
	@echo ">>> Build directory cleaned."

help:
	@echo "Available make targets:"
	@echo "  make            - Auto-detect CUDA or CPU and build executable."
	@echo "  make cpu        - Build CPU-accelerated (OpenMP + CUDA emulation) target."
	@echo "  make gpu        - Build native NVIDIA CUDA target (requires nvcc)."
	@echo "  make run        - Execute complete image & signal pipeline and save artifacts."
	@echo "  make benchmark  - Run multi-scale benchmark and export CSV results."
	@echo "  make test       - Run correctness verification test."
	@echo "  make package    - Package all artifacts into proof_of_execution.tar.gz."
	@echo "  make clean      - Remove compiled binaries."
