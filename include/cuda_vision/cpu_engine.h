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

#ifndef CUDA_VISION_CPU_ENGINE_H_
#define CUDA_VISION_CPU_ENGINE_H_

#include <string>
#include <vector>

#include "cuda_vision/common.h"
#include "cuda_vision/image_buffer.h"
#include "cuda_vision/signal_buffer.h"

namespace cuda_vision {

// CPU-based high-performance execution engine supporting both single-threaded reference
// and OpenMP multi-threaded SIMD implementations, as well as CUDA grid/block emulation.
class CpuEngine {
 public:
  explicit CpuEngine(int num_threads = 0);
  ~CpuEngine() = default;

  // Configuration.
  void SetNumThreads(int threads);
  int num_threads() const { return num_threads_; }
  std::string GetCpuInformation() const;

  // Filter execution methods.
  KernelStats RunGrayscale(const ImageBuffer& input, ImageBuffer* output,
                           bool multi_threaded = true);

  KernelStats RunGaussianBlur(const ImageBuffer& input, int radius, float sigma,
                              ImageBuffer* output, bool multi_threaded = true);

  KernelStats RunSobelEdges(const ImageBuffer& input, ImageBuffer* output,
                            bool multi_threaded = true);

  KernelStats RunBilateralFilter(const ImageBuffer& input, int radius,
                                 float sigma_spatial, float sigma_range,
                                 ImageBuffer* output, bool multi_threaded = true);

  KernelStats RunUnsharpMask(const ImageBuffer& input, int radius, float sigma,
                             float amount, ImageBuffer* output,
                             bool multi_threaded = true);

  KernelStats RunBatchedFir(const SignalBatch& input, const std::vector<float>& taps,
                            SignalBatch* output, bool multi_threaded = true);

  // Emulates CUDA 2D thread-block/grid scheduling and shared memory tiling
  // directly on the CPU to mathematically verify GPU algorithm equivalence.
  KernelStats RunCudaEmulatedGaussianBlur(const ImageBuffer& input, int radius,
                                          float sigma, ImageBuffer* output);

 private:
  int num_threads_ = 1;
};

}  // namespace cuda_vision

#endif  // CUDA_VISION_CPU_ENGINE_H_
