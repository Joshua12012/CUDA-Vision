// Copyright 2026 Joshua Deniese
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CUDA_VISION_COMMON_H_
#define CUDA_VISION_COMMON_H_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cuda_vision {

// Maximum dimensions and kernel bounds for static/shared memory allocations.
constexpr int kMaxFilterRadius = 15;
constexpr int kMaxFilterSize = 2 * kMaxFilterRadius + 1;
constexpr int kDefaultBlockDimX = 16;
constexpr int kDefaultBlockDimY = 16;
constexpr int kWarpSize = 32;

// Supported image and signal processing filters.
enum class FilterType {
  kGrayscale,
  kGaussianBlur,
  kSobelEdges,
  kBilateralFilter,
  kUnsharpMask,
  kBatchedFir,
  kAll
};

// Supported execution backends.
enum class ExecutionMode {
  kAuto,
  kCpuSingle,
  kCpuMulti,
  kGpu,
  kBenchmark
};

// Structure holding performance and kernel execution statistics.
struct KernelStats {
  std::string filter_name;
  std::string backend_name;
  int width = 0;
  int height = 0;
  int channels = 0;
  int batch_size = 1;
  int kernel_radius = 1;
  double elapsed_ms = 0.0;
  double throughput_mpixels_sec = 0.0;
  double memory_bandwidth_gb_sec = 0.0;
  double gflops = 0.0;
  double mse_vs_ref = 0.0;
  double psnr_vs_ref = 0.0;
};

// Pipeline configuration options passed from CLI.
struct PipelineConfig {
  std::string input_image_path = "data/input/Lena.png";
  std::string output_image_path = "data/output/result.png";
  std::string output_dir = "data/output";
  std::string artifacts_dir = "artifacts";
  FilterType filter = FilterType::kAll;
  ExecutionMode mode = ExecutionMode::kAuto;
  int kernel_radius = 2;              // Default 5x5 filter (radius = 2)
  float sigma_spatial = 1.5f;          // Spatial Gaussian variance
  float sigma_range = 0.1f;            // Range/photometric variance for bilateral
  int iterations = 5;                 // Timed benchmarking iterations
  int num_threads = 0;                // 0 = auto-detect max threads
  int signal_batch_count = 1000;      // Batch count for 1D signals
  int signal_length = 4096;           // Samples per signal
  int fir_taps = 64;                  // FIR filter tap count
  bool save_all = true;               // Save intermediate filter outputs
  bool run_benchmark = false;         // Run full multi-resolution benchmark
  bool verbose = true;                // Verbose logging
};

// Helper function to convert FilterType to human-readable string.
inline std::string FilterTypeToString(FilterType filter) {
  switch (filter) {
    case FilterType::kGrayscale:
      return "Grayscale Conversion";
    case FilterType::kGaussianBlur:
      return "Gaussian Blur 2D";
    case FilterType::kSobelEdges:
      return "Sobel Edge Detection";
    case FilterType::kBilateralFilter:
      return "Bilateral Edge-Preserving Filter";
    case FilterType::kUnsharpMask:
      return "Unsharp Masking (Sharpen)";
    case FilterType::kBatchedFir:
      return "Batched 1D FIR Filtering";
    case FilterType::kAll:
      return "Complete Processing Pipeline (All Filters)";
    default:
      return "Unknown";
  }
}

// Helper function to convert ExecutionMode to string.
inline std::string ExecutionModeToString(ExecutionMode mode) {
  switch (mode) {
    case ExecutionMode::kAuto:
      return "Auto (Prefer GPU, fallback to CPU)";
    case ExecutionMode::kCpuSingle:
      return "CPU Single-Threaded";
    case ExecutionMode::kCpuMulti:
      return "CPU Multi-Threaded (OpenMP)";
    case ExecutionMode::kGpu:
      return "NVIDIA CUDA GPU Acceleration";
    case ExecutionMode::kBenchmark:
      return "Multi-Scale Benchmark Mode";
    default:
      return "Unknown";
  }
}

// High-resolution timer utility for microsecond-accuracy benchmarking.
class ScopedTimer {
 public:
  ScopedTimer() : start_(std::chrono::high_resolution_clock::now()) {}

  void Reset() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  double ElapsedMilliseconds() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

  double ElapsedMicroseconds() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start_).count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}  // namespace cuda_vision

#endif  // CUDA_VISION_COMMON_H_
