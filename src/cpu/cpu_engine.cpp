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

#include "cuda_vision/cpu_engine.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cuda_vision {

CpuEngine::CpuEngine(int num_threads) {
  if (num_threads <= 0) {
    unsigned int hw = std::thread::hardware_concurrency();
    num_threads_ = (hw > 0) ? static_cast<int>(hw) : 4;
  } else {
    num_threads_ = num_threads;
  }
#ifdef _OPENMP
  omp_set_num_threads(num_threads_);
#endif
}

void CpuEngine::SetNumThreads(int threads) {
  num_threads_ = (threads > 0) ? threads : 1;
#ifdef _OPENMP
  omp_set_num_threads(num_threads_);
#endif
}

std::string CpuEngine::GetCpuInformation() const {
  std::ostringstream oss;
  oss << "CPU Threads: " << num_threads_ << " (Hardware Concurrency: "
      << std::thread::hardware_concurrency() << ")\n";
#ifdef _OPENMP
  oss << "OpenMP Version: " << _OPENMP << " (Enabled)";
#else
  oss << "OpenMP: Disabled (Single-threaded fallback)";
#endif
  return oss.str();
}

KernelStats CpuEngine::RunGrayscale(const ImageBuffer& input, ImageBuffer* output,
                                    bool multi_threaded) {
  KernelStats stats;
  stats.filter_name = "Grayscale Conversion";
  stats.backend_name = multi_threaded ? "CPU OpenMP Multi-Threaded" : "CPU Single-Threaded";
  stats.width = input.width();
  stats.height = input.height();
  stats.channels = input.channels();

  *output = ImageBuffer(input.width(), input.height(), 1);
  int w = input.width();
  int h = input.height();
  int c = input.channels();

  ScopedTimer timer;

  if (c == 1) {
    std::copy(input.Data(), input.Data() + input.total_pixels(), output->Data());
  } else {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(multi_threaded)
#endif
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float r = input.At(0, y, x);
        float g = input.At(1, y, x);
        float b = (c > 2) ? input.At(2, y, x) : g;
        float luma = 0.299f * r + 0.587f * g + 0.114f * b;
        output->At(0, y, x) = std::clamp(luma, 0.0f, 1.0f);
      }
    }
  }

  stats.elapsed_ms = timer.ElapsedMilliseconds();
  stats.throughput_mpixels_sec = (input.total_pixels() / (stats.elapsed_ms * 1e-3)) / 1e6;
  return stats;
}

KernelStats CpuEngine::RunGaussianBlur(const ImageBuffer& input, int radius, float sigma,
                                       ImageBuffer* output, bool multi_threaded) {
  KernelStats stats;
  stats.filter_name = "Gaussian Blur 2D";
  stats.backend_name = multi_threaded ? "CPU OpenMP Multi-Threaded" : "CPU Single-Threaded";
  stats.kernel_radius = radius;

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);
  int w = gray.width();
  int h = gray.height();

  stats.width = w;
  stats.height = h;
  stats.channels = 1;

  // Precompute 2D Gaussian weights
  int filter_size = 2 * radius + 1;
  std::vector<float> kernel(filter_size * filter_size);
  float sum = 0.0f;
  for (int ky = -radius; ky <= radius; ++ky) {
    for (int kx = -radius; kx <= radius; ++kx) {
      float weight = std::exp(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
      kernel[(ky + radius) * filter_size + (kx + radius)] = weight;
      sum += weight;
    }
  }
  for (float& val : kernel) val /= sum;

  ScopedTimer timer;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(multi_threaded)
#endif
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      for (int ky = -radius; ky <= radius; ++ky) {
        int ny = std::clamp(y + ky, 0, h - 1);
        for (int kx = -radius; kx <= radius; ++kx) {
          int nx = std::clamp(x + kx, 0, w - 1);
          float weight = kernel[(ky + radius) * filter_size + (kx + radius)];
          acc += gray.At(0, ny, nx) * weight;
        }
      }
      output->At(0, y, x) = std::clamp(acc, 0.0f, 1.0f);
    }
  }

  stats.elapsed_ms = timer.ElapsedMilliseconds();
  stats.throughput_mpixels_sec = (gray.total_pixels() / (stats.elapsed_ms * 1e-3)) / 1e6;
  return stats;
}

KernelStats CpuEngine::RunSobelEdges(const ImageBuffer& input, ImageBuffer* output,
                                     bool multi_threaded) {
  KernelStats stats;
  stats.filter_name = "Sobel Edge Detection";
  stats.backend_name = multi_threaded ? "CPU OpenMP Multi-Threaded" : "CPU Single-Threaded";

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);
  int w = gray.width();
  int h = gray.height();

  stats.width = w;
  stats.height = h;
  stats.channels = 1;

  ScopedTimer timer;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(multi_threaded)
#endif
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int y_prev = std::max(y - 1, 0);
      int y_next = std::min(y + 1, h - 1);
      int x_prev = std::max(x - 1, 0);
      int x_next = std::min(x + 1, w - 1);

      float p00 = gray.At(0, y_prev, x_prev);
      float p01 = gray.At(0, y_prev, x);
      float p02 = gray.At(0, y_prev, x_next);
      float p10 = gray.At(0, y, x_prev);
      float p12 = gray.At(0, y, x_next);
      float p20 = gray.At(0, y_next, x_prev);
      float p21 = gray.At(0, y_next, x);
      float p22 = gray.At(0, y_next, x_next);

      float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
      float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

      float mag = std::sqrt(gx * gx + gy * gy);
      output->At(0, y, x) = std::clamp(mag, 0.0f, 1.0f);
    }
  }

  stats.elapsed_ms = timer.ElapsedMilliseconds();
  stats.throughput_mpixels_sec = (gray.total_pixels() / (stats.elapsed_ms * 1e-3)) / 1e6;
  return stats;
}

KernelStats CpuEngine::RunBilateralFilter(const ImageBuffer& input, int radius,
                                          float sigma_spatial, float sigma_range,
                                          ImageBuffer* output, bool multi_threaded) {
  KernelStats stats;
  stats.filter_name = "Bilateral Filter";
  stats.backend_name = multi_threaded ? "CPU OpenMP Multi-Threaded" : "CPU Single-Threaded";
  stats.kernel_radius = radius;

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);
  int w = gray.width();
  int h = gray.height();

  stats.width = w;
  stats.height = h;
  stats.channels = 1;

  float two_spatial_sq = 2.0f * sigma_spatial * sigma_spatial;
  float two_range_sq = 2.0f * sigma_range * sigma_range;

  ScopedTimer timer;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) if(multi_threaded)
#endif
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float center_val = gray.At(0, y, x);
      float sum_val = 0.0f;
      float sum_weight = 0.0f;

      for (int ky = -radius; ky <= radius; ++ky) {
        int ny = std::clamp(y + ky, 0, h - 1);
        for (int kx = -radius; kx <= radius; ++kx) {
          int nx = std::clamp(x + kx, 0, w - 1);
          float neighbor_val = gray.At(0, ny, nx);

          float spatial_dist_sq = static_cast<float>(kx * kx + ky * ky);
          float range_diff = neighbor_val - center_val;
          float range_dist_sq = range_diff * range_diff;

          float weight = std::exp(-spatial_dist_sq / two_spatial_sq - range_dist_sq / two_range_sq);
          sum_val += neighbor_val * weight;
          sum_weight += weight;
        }
      }

      output->At(0, y, x) = (sum_weight > 1e-7f) ? (sum_val / sum_weight) : center_val;
    }
  }

  stats.elapsed_ms = timer.ElapsedMilliseconds();
  stats.throughput_mpixels_sec = (gray.total_pixels() / (stats.elapsed_ms * 1e-3)) / 1e6;
  return stats;
}

KernelStats CpuEngine::RunUnsharpMask(const ImageBuffer& input, int radius, float sigma,
                                      float amount, ImageBuffer* output,
                                      bool multi_threaded) {
  ImageBuffer blurred;
  KernelStats stats = RunGaussianBlur(input, radius, sigma, &blurred, multi_threaded);
  stats.filter_name = "Unsharp Masking (Sharpen)";

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);
  int w = gray.width();
  int h = gray.height();

  ScopedTimer timer;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(multi_threaded)
#endif
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float orig = gray.At(0, y, x);
      float blur = blurred.At(0, y, x);
      float sharp = orig + amount * (orig - blur);
      output->At(0, y, x) = std::clamp(sharp, 0.0f, 1.0f);
    }
  }

  stats.elapsed_ms += timer.ElapsedMilliseconds();
  stats.throughput_mpixels_sec = (gray.total_pixels() / (stats.elapsed_ms * 1e-3)) / 1e6;
  return stats;
}

KernelStats CpuEngine::RunBatchedFir(const SignalBatch& input,
                                     const std::vector<float>& taps,
                                     SignalBatch* output, bool multi_threaded) {
  KernelStats stats;
  stats.filter_name = "Batched 1D FIR Filtering";
  stats.backend_name = multi_threaded ? "CPU OpenMP Multi-Threaded" : "CPU Single-Threaded";
  stats.batch_size = input.num_signals();
  stats.width = input.signal_length();

  *output = input.CloneEmpty();
  int num_sigs = input.num_signals();
  int sig_len = input.signal_length();
  int num_taps = static_cast<int>(taps.size());
  int half_taps = (num_taps - 1) / 2;

  ScopedTimer timer;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if(multi_threaded)
#endif
  for (int s = 0; s < num_sigs; ++s) {
    for (int t = 0; t < sig_len; ++t) {
      float acc = 0.0f;
      for (int k = 0; k < num_taps; ++k) {
        int src_t = std::clamp(t + (k - half_taps), 0, sig_len - 1);
        acc += input.At(s, src_t) * taps[k];
      }
      output->At(s, t) = acc;
    }
  }

  stats.elapsed_ms = timer.ElapsedMilliseconds();
  stats.throughput_mpixels_sec = (input.total_samples() / (stats.elapsed_ms * 1e-3)) / 1e6;
  return stats;
}

KernelStats CpuEngine::RunCudaEmulatedGaussianBlur(const ImageBuffer& input, int radius,
                                                   float sigma, ImageBuffer* output) {
  KernelStats stats;
  stats.filter_name = "CUDA Tiled Kernel (CPU Emulation Mode)";
  stats.backend_name = "CUDA Emulated Shared-Memory Tiles";
  stats.kernel_radius = radius;

  ImageBuffer gray = (input.channels() > 1) ? input.ToGrayscale() : input;
  *output = ImageBuffer(gray.width(), gray.height(), 1);
  int w = gray.width();
  int h = gray.height();

  stats.width = w;
  stats.height = h;
  stats.channels = 1;

  // Precompute 2D Gaussian weights
  int filter_size = 2 * radius + 1;
  std::vector<float> kernel(filter_size * filter_size);
  float sum = 0.0f;
  for (int ky = -radius; ky <= radius; ++ky) {
    for (int kx = -radius; kx <= radius; ++kx) {
      float weight = std::exp(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
      kernel[(ky + radius) * filter_size + (kx + radius)] = weight;
      sum += weight;
    }
  }
  for (float& val : kernel) val /= sum;

  ScopedTimer timer;

  // Emulate CUDA Thread Blocks and Shared Memory Tiling
  constexpr int kBlockDimX = 16;
  constexpr int kBlockDimY = 16;
  int grid_dim_x = (w + kBlockDimX - 1) / kBlockDimX;
  int grid_dim_y = (h + kBlockDimY - 1) / kBlockDimY;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
  for (int by = 0; by < grid_dim_y; ++by) {
    for (int bx = 0; bx < grid_dim_x; ++bx) {
      // Allocate simulated block shared memory tile
      int shared_w = kBlockDimX + 2 * radius;
      int shared_h = kBlockDimY + 2 * radius;
      std::vector<float> s_tile(shared_w * shared_h);

      // Cooperative tile & halo loading
      for (int ty = 0; ty < kBlockDimY; ++ty) {
        for (int tx = 0; tx < kBlockDimX; ++tx) {
          int gx = bx * kBlockDimX + tx;
          int gy = by * kBlockDimY + ty;
          int sx = tx + radius;
          int sy = ty + radius;

          // Center
          int cgx = std::clamp(gx, 0, w - 1);
          int cgy = std::clamp(gy, 0, h - 1);
          s_tile[sy * shared_w + sx] = gray.At(0, cgy, cgx);

          // Halos
          if (ty < radius) {
            s_tile[ty * shared_w + sx] = gray.At(0, std::clamp(gy - radius, 0, h - 1), cgx);
            s_tile[(sy + kBlockDimY) * shared_w + sx] =
                gray.At(0, std::clamp(gy + kBlockDimY, 0, h - 1), cgx);
          }
          if (tx < radius) {
            s_tile[sy * shared_w + tx] = gray.At(0, cgy, std::clamp(gx - radius, 0, w - 1));
            s_tile[sy * shared_w + (sx + kBlockDimX)] =
                gray.At(0, cgy, std::clamp(gx + kBlockDimX, 0, w - 1));
          }
          if (tx < radius && ty < radius) {
            s_tile[ty * shared_w + tx] =
                gray.At(0, std::clamp(gy - radius, 0, h - 1), std::clamp(gx - radius, 0, w - 1));
            s_tile[ty * shared_w + (sx + kBlockDimX)] =
                gray.At(0, std::clamp(gy - radius, 0, h - 1), std::clamp(gx + kBlockDimX, 0, w - 1));
            s_tile[(sy + kBlockDimY) * shared_w + tx] =
                gray.At(0, std::clamp(gy + kBlockDimY, 0, h - 1), std::clamp(gx - radius, 0, w - 1));
            s_tile[(sy + kBlockDimY) * shared_w + (sx + kBlockDimX)] =
                gray.At(0, std::clamp(gy + kBlockDimY, 0, h - 1), std::clamp(gx + kBlockDimX, 0, w - 1));
          }
        }
      }

      // __syncthreads() simulation

      // Compute convolution from shared memory
      for (int ty = 0; ty < kBlockDimY; ++ty) {
        for (int tx = 0; tx < kBlockDimX; ++tx) {
          int gx = bx * kBlockDimX + tx;
          int gy = by * kBlockDimY + ty;
          if (gx < w && gy < h) {
            int sx = tx + radius;
            int sy = ty + radius;
            float acc = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
              for (int kx = -radius; kx <= radius; ++kx) {
                float weight = kernel[(ky + radius) * filter_size + (kx + radius)];
                acc += s_tile[(sy + ky) * shared_w + (sx + kx)] * weight;
              }
            }
            output->At(0, gy, gx) = std::clamp(acc, 0.0f, 1.0f);
          }
        }
      }
    }
  }

  stats.elapsed_ms = timer.ElapsedMilliseconds();
  stats.throughput_mpixels_sec = (gray.total_pixels() / (stats.elapsed_ms * 1e-3)) / 1e6;
  return stats;
}

}  // namespace cuda_vision
