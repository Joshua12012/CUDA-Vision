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

#ifndef CUDA_VISION_CUDA_KERNELS_H_
#define CUDA_VISION_CUDA_KERNELS_H_

#include <string>
#include <vector>

#include "cuda_vision/common.h"
#include "cuda_vision/image_buffer.h"
#include "cuda_vision/signal_buffer.h"

namespace cuda_vision {

// Check if a compatible CUDA-enabled GPU device is detected at runtime.
bool IsCudaAvailable();

// Query and return human-readable CUDA device properties and specs.
std::string GetCudaDeviceInformation();

// CUDA Engine wrapper executing kernels on NVIDIA GPU hardware.
class CudaEngine {
 public:
  CudaEngine();
  ~CudaEngine();

  // Initialize CUDA context and allocate device memory buffers.
  bool Initialize(int device_id = 0);

  // Kernel execution methods on device.
  KernelStats RunGrayscale(const ImageBuffer& input, ImageBuffer* output);

  KernelStats RunGaussianBlur(const ImageBuffer& input, int radius, float sigma,
                              ImageBuffer* output);

  KernelStats RunSobelEdges(const ImageBuffer& input, ImageBuffer* output);

  KernelStats RunBilateralFilter(const ImageBuffer& input, int radius,
                                 float sigma_spatial, float sigma_range,
                                 ImageBuffer* output);

  KernelStats RunUnsharpMask(const ImageBuffer& input, int radius, float sigma,
                             float amount, ImageBuffer* output);

  KernelStats RunBatchedFir(const SignalBatch& input, const std::vector<float>& taps,
                            SignalBatch* output);

  // Asynchronous multi-stream batch pipeline demonstration.
  KernelStats RunStreamPipelineBatch(const std::vector<ImageBuffer>& batch_inputs,
                                     int radius, float sigma,
                                     std::vector<ImageBuffer>* batch_outputs);

 private:
  int device_id_ = 0;
  bool is_initialized_ = false;
};

}  // namespace cuda_vision

#endif  // CUDA_VISION_CUDA_KERNELS_H_
