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

#ifndef CUDA_VISION_IMAGE_BUFFER_H_
#define CUDA_VISION_IMAGE_BUFFER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace cuda_vision {

// 2D image buffer managing normalized single-precision floating point pixels.
// Pixels are stored in planar or interleaved order in range [0.0f, 1.0f].
class ImageBuffer {
 public:
  // Default constructor creates empty buffer.
  ImageBuffer();

  // Construct with given dimensions and channel count, initialized to zero.
  ImageBuffer(int width, int height, int channels);

  // Copy constructor and assignment.
  ImageBuffer(const ImageBuffer& other);
  ImageBuffer& operator=(const ImageBuffer& other);

  // Move constructor and assignment.
  ImageBuffer(ImageBuffer&& other) noexcept;
  ImageBuffer& operator=(ImageBuffer&& other) noexcept;

  ~ImageBuffer() = default;

  // Factory methods.
  static ImageBuffer LoadFromFile(const std::string& file_path);
  static ImageBuffer CreateSyntheticPattern(int width, int height, int pattern_type = 0);

  // Save current image buffer to PNG format on disk.
  bool SaveToPng(const std::string& file_path) const;

  // Convert multi-channel RGB/RGBA buffer to 1-channel Grayscale.
  ImageBuffer ToGrayscale() const;

  // Clones dimensions and channels without copying contents.
  ImageBuffer CloneEmpty() const;

  // Metric calculation functions.
  static double CalculateMse(const ImageBuffer& img_a, const ImageBuffer& img_b);
  static double CalculatePsnr(const ImageBuffer& img_a, const ImageBuffer& img_b);

  // Data accessors.
  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  size_t total_pixels() const { return static_cast<size_t>(width_) * height_; }
  size_t total_elements() const { return total_pixels() * channels_; }
  size_t size_bytes() const { return total_elements() * sizeof(float); }
  bool empty() const { return data_.empty(); }

  float* Data() { return data_.data(); }
  const float* Data() const { return data_.data(); }

  // Pixel accessor for planar representation: [c * height * width + y * width + x]
  float& At(int c, int y, int x) {
    return data_[(c * height_ + y) * width_ + x];
  }

  const float& At(int c, int y, int x) const {
    return data_[(c * height_ + y) * width_ + x];
  }

  // 1-channel convenience accessor.
  float& At(int y, int x) {
    return data_[y * width_ + x];
  }

  const float& At(int y, int x) const {
    return data_[y * width_ + x];
  }

 private:
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;
  std::vector<float> data_;
};

}  // namespace cuda_vision

#endif  // CUDA_VISION_IMAGE_BUFFER_H_
