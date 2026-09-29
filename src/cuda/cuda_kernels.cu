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

#include "cuda_vision/cuda_kernels.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s (code: %d)\n", __FILE__,  \
                   __LINE__, cudaGetErrorString(err), static_cast<int>(err)); \
    }                                                                         \
  } while (0)

namespace cuda_vision {

// Constant memory caches for zero bank-conflict broadcast across warps.
__constant__ float c_filter_weights[kMaxFilterSize * kMaxFilterSize];
__constant__ float c_fir_taps[512];

// Tile block configuration for 2D image processing.
constexpr int kTileDimX = 16;
constexpr int kTileDimY = 16;
constexpr int kMaxTileHalo = 4;
constexpr int kSharedWidth = kTileDimX + 2 * kMaxTileHalo;
constexpr int kSharedHeight = kTileDimY + 2 * kMaxTileHalo;

// ============================================================================
// CUDA Kernel 1: Coalesced RGB/RGBA to Grayscale Conversion
// ============================================================================
__global__ void RgbToGrayscaleKernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int pixel_idx = y * width + x;
    if (channels == 1) {
      output[pixel_idx] = input[pixel_idx];
    } else {
      // Planar input layout: [c * height * width + pixel_idx]
      float r = input[0 * height * width + pixel_idx];
      float g = input[1 * height * width + pixel_idx];
      float b = (channels > 2) ? input[2 * height * width + pixel_idx] : g;

      // Rec. 601 luma standard weights
      float luma = 0.299f * r + 0.587f * g + 0.114f * b;
      output[pixel_idx] = fminf(fmaxf(luma, 0.0f), 1.0f);
    }
  }
}

// ============================================================================
// CUDA Kernel 2: Tiled 2D Gaussian Convolution with Shared Memory & Halo
// ============================================================================
__global__ void GaussianBlurTiledKernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int width, int height, int radius) {
  __shared__ float s_tile[kSharedHeight][kSharedWidth];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int gx = blockIdx.x * blockDim.x + tx;
  int gy = blockIdx.y * blockDim.y + ty;

  // Shared tile coordinate corresponding to center
  int sx = tx + radius;
  int sy = ty + radius;

  // 1. Cooperative Load: Center element
  int clamped_gx = min(max(gx, 0), width - 1);
  int clamped_gy = min(max(gy, 0), height - 1);
  s_tile[sy][sx] = input[clamped_gy * width + clamped_gx];

  // 2. Cooperative Load: Top/Bottom Halo
  if (ty < radius) {
    // Top halo
    int top_y = min(max(gy - radius, 0), height - 1);
    s_tile[ty][sx] = input[top_y * width + clamped_gx];

    // Bottom halo
    int bot_y = min(max(gy + blockDim.y, 0), height - 1);
    s_tile[sy + blockDim.y][sx] = input[bot_y * width + clamped_gx];
  }

  // 3. Cooperative Load: Left/Right Halo
  if (tx < radius) {
    // Left halo
    int left_x = min(max(gx - radius, 0), width - 1);
    s_tile[sy][tx] = input[clamped_gy * width + left_x];

    // Right halo
    int right_x = min(max(gx + blockDim.x, 0), width - 1);
    s_tile[sy][sx + blockDim.x] = input[clamped_gy * width + right_x];
  }

  // 4. Cooperative Load: 4 Corners
  if (tx < radius && ty < radius) {
    // Top-Left
    int tl_x = min(max(gx - radius, 0), width - 1);
    int tl_y = min(max(gy - radius, 0), height - 1);
    s_tile[ty][tx] = input[tl_y * width + tl_x];

    // Top-Right
    int tr_x = min(max(gx + blockDim.x, 0), width - 1);
    int tr_y = min(max(gy - radius, 0), height - 1);
    s_tile[ty][sx + blockDim.x] = input[tr_y * width + tr_x];

    // Bottom-Left
    int bl_x = min(max(gx - radius, 0), width - 1);
    int bl_y = min(max(gy + blockDim.y, 0), height - 1);
    s_tile[sy + blockDim.y][tx] = input[bl_y * width + bl_x];

    // Bottom-Right
    int br_x = min(max(gx + blockDim.x, 0), width - 1);
    int br_y = min(max(gy + blockDim.y, 0), height - 1);
    s_tile[sy + blockDim.y][sx + blockDim.x] = input[br_y * width + br_x];
  }

  __syncthreads();

  // Perform 2D convolution purely out of fast shared memory
  if (gx < width && gy < height) {
    float sum = 0.0f;
    int filter_size = 2 * radius + 1;

    for (int ky = -radius; ky <= radius; ++ky) {
      for (int kx = -radius; kx <= radius; ++kx) {
        float weight = c_filter_weights[(ky + radius) * filter_size + (kx + radius)];
        sum += s_tile[sy + ky][sx + kx] * weight;
      }
    }
    output[gy * width + gx] = sum;
  }
}

// ============================================================================
// CUDA Kernel 3: Tiled Sobel Edge Gradient Magnitude & Orientation
// ============================================================================
__global__ void SobelEdgeTiledKernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int width, int height) {
  __shared__ float s_tile[kTileDimY + 2][kTileDimX + 2];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int gx = blockIdx.x * blockDim.x + tx;
  int gy = blockIdx.y * blockDim.y + ty;

  int sx = tx + 1;
  int sy = ty + 1;

  // Load center
  int cx = min(max(gx, 0), width - 1);
  int cy = min(max(gy, 0), height - 1);
  s_tile[sy][sx] = input[cy * width + cx];

  // Load boundaries (1-pixel radius)
  if (ty == 0) {
    s_tile[0][sx] = input[min(max(gy - 1, 0), height - 1) * width + cx];
    s_tile[kTileDimY + 1][sx] = input[min(max(gy + kTileDimY, 0), height - 1) * width + cx];
  }
  if (tx == 0) {
    s_tile[sy][0] = input[cy * width + min(max(gx - 1, 0), width - 1)];
    s_tile[sy][kTileDimX + 1] = input[cy * width + min(max(gx + kTileDimX, 0), width - 1)];
  }
  if (tx == 0 && ty == 0) {
    s_tile[0][0] = input[min(max(gy - 1, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)];
    s_tile[0][kTileDimX + 1] = input[min(max(gy - 1, 0), height - 1) * width + min(max(gx + kTileDimX, 0), width - 1)];
    s_tile[kTileDimY + 1][0] = input[min(max(gy + kTileDimY, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)];
    s_tile[kTileDimY + 1][kTileDimX + 1] = input[min(max(gy + kTileDimY, 0), height - 1) * width + min(max(gx + kTileDimX, 0), width - 1)];
  }

  __syncthreads();

  if (gx < width && gy < height) {
    // 3x3 Sobel Operator:
    // Gx: [-1  0  1]    Gy: [-1 -2 -1]
    //     [-2  0  2]        [ 0  0  0]
    //     [-1  0  1]        [ 1  2  1]
    float p00 = s_tile[sy - 1][sx - 1];
    float p01 = s_tile[sy - 1][sx];
    float p02 = s_tile[sy - 1][sx + 1];
    float p10 = s_tile[sy][sx - 1];
    float p12 = s_tile[sy][sx + 1];
    float p20 = s_tile[sy + 1][sx - 1];
    float p21 = s_tile[sy + 1][sx];
    float p22 = s_tile[sy + 1][sx + 1];

    float gx_val = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
    float gy_val = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

    float magnitude = sqrtf(gx_val * gx_val + gy_val * gy_val);
    output[gy * width + gx] = fminf(magnitude, 1.0f);
  }
}

// ============================================================================
// CUDA Kernel 4: Tiled Non-linear Bilateral Edge-Preserving Filter
// ============================================================================
__global__ void BilateralFilterTiledKernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int width, int height,
                                           int radius, float sigma_spatial,
                                           float sigma_range) {
  __shared__ float s_tile[kSharedHeight][kSharedWidth];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int gx = blockIdx.x * blockDim.x + tx;
  int gy = blockIdx.y * blockDim.y + ty;

  int sx = tx + radius;
  int sy = ty + radius;

  // Load center and halo with boundary clamping
  int cx = min(max(gx, 0), width - 1);
  int cy = min(max(gy, 0), height - 1);
  s_tile[sy][sx] = input[cy * width + cx];

  if (ty < radius) {
    s_tile[ty][sx] = input[min(max(gy - radius, 0), height - 1) * width + cx];
    s_tile[sy + blockDim.y][sx] = input[min(max(gy + blockDim.y, 0), height - 1) * width + cx];
  }
  if (tx < radius) {
    s_tile[sy][tx] = input[cy * width + min(max(gx - radius, 0), width - 1)];
    s_tile[sy][sx + blockDim.x] = input[cy * width + min(max(gx + blockDim.x, 0), width - 1)];
  }
  if (tx < radius && ty < radius) {
    s_tile[ty][tx] = input[min(max(gy - radius, 0), height - 1) * width + min(max(gx - radius, 0), width - 1)];
    s_tile[ty][sx + blockDim.x] = input[min(max(gy - radius, 0), height - 1) * width + min(max(gx + blockDim.x, 0), width - 1)];
    s_tile[sy + blockDim.y][tx] = input[min(max(gy + blockDim.y, 0), height - 1) * width + min(max(gx - radius, 0), width - 1)];
    s_tile[sy + blockDim.y][sx + blockDim.x] = input[min(max(gy + blockDim.y, 0), height - 1) * width + min(max(gx + blockDim.x, 0), width - 1)];
  }

  __syncthreads();

  if (gx < width && gy < height) {
    float center_val = s_tile[sy][sx];
    float sum_val = 0.0f;
    float sum_weight = 0.0f;
    float two_spatial_sq = 2.0f * sigma_spatial * sigma_spatial;
    float two_range_sq = 2.0f * sigma_range * sigma_range;

    for (int ky = -radius; ky <= radius; ++ky) {
      for (int kx = -radius; kx <= radius; ++kx) {
        float neighbor_val = s_tile[sy + ky][sx + kx];
        float spatial_dist_sq = static_cast<float>(kx * kx + ky * ky);
        float range_diff = neighbor_val - center_val;
        float range_dist_sq = range_diff * range_diff;

        float weight = expf(-spatial_dist_sq / two_spatial_sq - range_dist_sq / two_range_sq);
        sum_val += neighbor_val * weight;
        sum_weight += weight;
      }
    }

    output[gy * width + gx] = (sum_weight > 1e-7f) ? (sum_val / sum_weight) : center_val;
  }
}

// ============================================================================
// CUDA Kernel 5: High-Throughput Batched 1D FIR Filtering
// ============================================================================
__global__ void BatchedFirSignalKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int num_signals, int signal_length,
                                       int num_taps) {
  int signal_idx = blockIdx.y;
  int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (signal_idx < num_signals && sample_idx < signal_length) {
    const float* sig_in = input + static_cast<size_t>(signal_idx) * signal_length;
    float* sig_out = output + static_cast<size_t>(signal_idx) * signal_length;

    float acc = 0.0f;
    int half_taps = (num_taps - 1) / 2;

    for (int k = 0; k < num_taps; ++k) {
      int tap_offset = k - half_taps;
      int src_idx = sample_idx + tap_offset;
      src_idx = min(max(src_idx, 0), signal_length - 1);
      acc += sig_in[src_idx] * c_fir_taps[k];
    }

    sig_out[sample_idx] = acc;
  }
}

// ============================================================================
// CUDA Kernel 6: Warp-Level Reduction for Image Statistics
// ============================================================================
__inline__ __device__ float WarpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void ImageReductionKernel(const float* __restrict__ input,
                                     float* __restrict__ block_sums,
                                     int total_elements) {
  __shared__ float s_warp_sums[32];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  float local_sum = 0.0f;
  for (int i = gid; i < total_elements; i += stride) {
    local_sum += input[i];
  }

  // Warp-level shuffle reduction
  local_sum = WarpReduceSum(local_sum);

  int lane = tid % 32;
  int warp_id = tid / 32;
  if (lane == 0) {
    s_warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  // Block leader reduces warp results
  if (tid < 32) {
    float block_sum = (tid < (blockDim.x / 32)) ? s_warp_sums[tid] : 0.0f;
    block_sum = WarpReduceSum(block_sum);
    if (tid == 0) {
      block_sums[blockIdx.x] = block_sum;
    }
  }
}

// ============================================================================
// Host Implementation of CudaEngine
// ============================================================================
bool IsCudaAvailable() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return (err == cudaSuccess && count > 0);
}

std::string GetCudaDeviceInformation() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    return "No CUDA-capable GPU detected (CPU mode active).";
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::ostringstream oss;
  oss << "Device: " << prop.name << "\n"
      << "Compute Capability: " << prop.major << "." << prop.minor << "\n"
      << "Multiprocessors (SMs): " << prop.multiProcessorCount << "\n"
      << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n"
      << "Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n"
      << "Warp Size: " << prop.warpSize;
  return oss.str();
}

CudaEngine::CudaEngine() : device_id_(0), is_initialized_(false) {}

CudaEngine::~CudaEngine() {
  if (is_initialized_) {
    cudaDeviceReset();
  }
}

bool CudaEngine::Initialize(int device_id) {
  if (!IsCudaAvailable()) {
    std::cerr << "[CudaEngine] Cannot initialize: No CUDA device found." << std::endl;
    return false;
  }
  device_id_ = device_id;
  CUDA_CHECK(cudaSetDevice(device_id_));
  is_initialized_ = true;
  return true;
}

KernelStats CudaEngine::RunGrayscale(const ImageBuffer& input, ImageBuffer* output) {
  KernelStats stats;
  stats.filter_name = "Grayscale Conversion";
  stats.backend_name = "CUDA GPU";
  stats.width = input.width();
  stats.height = input.height();
  stats.channels = input.channels();

  *output = ImageBuffer(input.width(), input.height(), 1);

  if (!is_initialized_) return stats;

  float *d_in = nullptr, *d_out = nullptr;
  size_t in_bytes = input.size_bytes();
  size_t out_bytes = output->size_bytes();

  CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

  CUDA_CHECK(cudaMemcpy(d_in, input.Data(), in_bytes, cudaMemcpyHostToDevice));

  dim3 block(kDefaultBlockDimX, kDefaultBlockDimY);
  dim3 grid((input.width() + block.x - 1) / block.x,
            (input.height() + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  RgbToGrayscaleKernel<<<grid, block>>>(d_in, d_out, input.width(), input.height(), input.channels());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  stats.elapsed_ms = ms;
  stats.throughput_mpixels_sec = (input.total_pixels() / (ms * 1e-3)) / 1e6;

  CUDA_CHECK(cudaMemcpy(output->Data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return stats;
}

KernelStats CudaEngine::RunGaussianBlur(const ImageBuffer& input, int radius, float sigma,
                                        ImageBuffer* output) {
  KernelStats stats;
  stats.filter_name = "Gaussian Blur (Shared Memory Tiled)";
  stats.backend_name = "CUDA GPU";
  stats.width = input.width();
  stats.height = input.height();
  stats.channels = 1;
  stats.kernel_radius = radius;

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);

  if (!is_initialized_) return stats;

  int filter_size = 2 * radius + 1;
  std::vector<float> h_weights(filter_size * filter_size);
  float sum = 0.0f;
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      float w = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
      h_weights[(y + radius) * filter_size + (x + radius)] = w;
      sum += w;
    }
  }
  for (float& w : h_weights) w /= sum;

  CUDA_CHECK(cudaMemcpyToSymbol(c_filter_weights, h_weights.data(),
                                h_weights.size() * sizeof(float)));

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, gray.size_bytes()));
  CUDA_CHECK(cudaMalloc(&d_out, output->size_bytes()));
  CUDA_CHECK(cudaMemcpy(d_in, gray.Data(), gray.size_bytes(), cudaMemcpyHostToDevice));

  dim3 block(kTileDimX, kTileDimY);
  dim3 grid((gray.width() + block.x - 1) / block.x,
            (gray.height() + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  GaussianBlurTiledKernel<<<grid, block>>>(d_in, d_out, gray.width(), gray.height(), radius);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  stats.elapsed_ms = ms;
  stats.throughput_mpixels_sec = (gray.total_pixels() / (ms * 1e-3)) / 1e6;

  CUDA_CHECK(cudaMemcpy(output->Data(), d_out, output->size_bytes(), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return stats;
}

KernelStats CudaEngine::RunSobelEdges(const ImageBuffer& input, ImageBuffer* output) {
  KernelStats stats;
  stats.filter_name = "Sobel Edge Detection (Shared Memory)";
  stats.backend_name = "CUDA GPU";
  stats.width = input.width();
  stats.height = input.height();
  stats.channels = 1;

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);

  if (!is_initialized_) return stats;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, gray.size_bytes()));
  CUDA_CHECK(cudaMalloc(&d_out, output->size_bytes()));
  CUDA_CHECK(cudaMemcpy(d_in, gray.Data(), gray.size_bytes(), cudaMemcpyHostToDevice));

  dim3 block(kTileDimX, kTileDimY);
  dim3 grid((gray.width() + block.x - 1) / block.x,
            (gray.height() + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  SobelEdgeTiledKernel<<<grid, block>>>(d_in, d_out, gray.width(), gray.height());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  stats.elapsed_ms = ms;
  stats.throughput_mpixels_sec = (gray.total_pixels() / (ms * 1e-3)) / 1e6;

  CUDA_CHECK(cudaMemcpy(output->Data(), d_out, output->size_bytes(), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return stats;
}

KernelStats CudaEngine::RunBilateralFilter(const ImageBuffer& input, int radius,
                                           float sigma_spatial, float sigma_range,
                                           ImageBuffer* output) {
  KernelStats stats;
  stats.filter_name = "Bilateral Filter";
  stats.backend_name = "CUDA GPU";
  stats.width = input.width();
  stats.height = input.height();
  stats.channels = 1;
  stats.kernel_radius = radius;

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);

  if (!is_initialized_) return stats;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, gray.size_bytes()));
  CUDA_CHECK(cudaMalloc(&d_out, output->size_bytes()));
  CUDA_CHECK(cudaMemcpy(d_in, gray.Data(), gray.size_bytes(), cudaMemcpyHostToDevice));

  dim3 block(kTileDimX, kTileDimY);
  dim3 grid((gray.width() + block.x - 1) / block.x,
            (gray.height() + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  BilateralFilterTiledKernel<<<grid, block>>>(d_in, d_out, gray.width(), gray.height(),
                                              radius, sigma_spatial, sigma_range);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  stats.elapsed_ms = ms;
  stats.throughput_mpixels_sec = (gray.total_pixels() / (ms * 1e-3)) / 1e6;

  CUDA_CHECK(cudaMemcpy(output->Data(), d_out, output->size_bytes(), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return stats;
}

KernelStats CudaEngine::RunUnsharpMask(const ImageBuffer& input, int radius, float sigma,
                                       float amount, ImageBuffer* output) {
  ImageBuffer blurred;
  KernelStats stats = RunGaussianBlur(input, radius, sigma, &blurred);
  stats.filter_name = "Unsharp Masking (Sharpen)";

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);

  // Sharpen: output = input + amount * (input - blurred)
  for (int y = 0; y < gray.height(); ++y) {
    for (int x = 0; x < gray.width(); ++x) {
      float orig = gray.At(y, x);
      float blur = blurred.At(y, x);
      float sharp = orig + amount * (orig - blur);
      output->At(y, x) = std::clamp(sharp, 0.0f, 1.0f);
    }
  }
  return stats;
}

KernelStats CudaEngine::RunBatchedFir(const SignalBatch& input,
                                      const std::vector<float>& taps,
                                      SignalBatch* output) {
  KernelStats stats;
  stats.filter_name = "Batched 1D FIR Filtering";
  stats.backend_name = "CUDA GPU";
  stats.batch_size = input.num_signals();
  stats.width = input.signal_length();

  *output = input.CloneEmpty();
  if (!is_initialized_) return stats;

  CUDA_CHECK(cudaMemcpyToSymbol(c_fir_taps, taps.data(), taps.size() * sizeof(float)));

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, input.size_bytes()));
  CUDA_CHECK(cudaMalloc(&d_out, output->size_bytes()));
  CUDA_CHECK(cudaMemcpy(d_in, input.Data(), input.size_bytes(), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((input.signal_length() + block.x - 1) / block.x, input.num_signals());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  BatchedFirSignalKernel<<<grid, block>>>(d_in, d_out, input.num_signals(),
                                          input.signal_length(), static_cast<int>(taps.size()));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  stats.elapsed_ms = ms;
  stats.throughput_mpixels_sec = (input.total_samples() / (ms * 1e-3)) / 1e6;

  CUDA_CHECK(cudaMemcpy(output->Data(), d_out, output->size_bytes(), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return stats;
}

KernelStats CudaEngine::RunStreamPipelineBatch(const std::vector<ImageBuffer>& batch_inputs,
                                               int radius, float sigma,
                                               std::vector<ImageBuffer>* batch_outputs) {
  KernelStats stats;
  stats.filter_name = "Asynchronous Multi-Stream Batch Pipeline";
  stats.backend_name = "CUDA GPU Streams";
  stats.batch_size = static_cast<int>(batch_inputs.size());

  batch_outputs->clear();
  batch_outputs->resize(batch_inputs.size());

  if (!is_initialized_ || batch_inputs.empty()) return stats;

  constexpr int kNumStreams = 4;
  cudaStream_t streams[kNumStreams];
  for (int i = 0; i < kNumStreams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  // Pre-calculate weights and copy to constant memory
  int filter_size = 2 * radius + 1;
  std::vector<float> h_weights(filter_size * filter_size);
  float sum = 0.0f;
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      float w = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
      h_weights[(y + radius) * filter_size + (x + radius)] = w;
      sum += w;
    }
  }
  for (float& w : h_weights) w /= sum;
  CUDA_CHECK(cudaMemcpyToSymbol(c_filter_weights, h_weights.data(),
                                h_weights.size() * sizeof(float)));

  int w = batch_inputs[0].width();
  int h = batch_inputs[0].height();
  size_t img_bytes = static_cast<size_t>(w) * h * sizeof(float);

  float* d_in[kNumStreams];
  float* d_out[kNumStreams];
  for (int i = 0; i < kNumStreams; ++i) {
    CUDA_CHECK(cudaMalloc(&d_in[i], img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out[i], img_bytes));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  dim3 block(kTileDimX, kTileDimY);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

  for (size_t i = 0; i < batch_inputs.size(); ++i) {
    int stream_id = i % kNumStreams;
    (*batch_outputs)[i] = ImageBuffer(w, h, 1);

    CUDA_CHECK(cudaMemcpyAsync(d_in[stream_id], batch_inputs[i].Data(), img_bytes,
                               cudaMemcpyHostToDevice, streams[stream_id]));

    GaussianBlurTiledKernel<<<grid, block, 0, streams[stream_id]>>>(
        d_in[stream_id], d_out[stream_id], w, h, radius);

    CUDA_CHECK(cudaMemcpyAsync((*batch_outputs)[i].Data(), d_out[stream_id], img_bytes,
                               cudaMemcpyDeviceToHost, streams[stream_id]));
  }

  for (int i = 0; i < kNumStreams; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    CUDA_CHECK(cudaFree(d_in[i]));
    CUDA_CHECK(cudaFree(d_out[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  stats.elapsed_ms = ms;
  stats.throughput_mpixels_sec = (static_cast<double>(w) * h * batch_inputs.size() / (ms * 1e-3)) / 1e6;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return stats;
}

}  // namespace cuda_vision
