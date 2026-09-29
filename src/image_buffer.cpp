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

#include "cuda_vision/image_buffer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace cuda_vision {

ImageBuffer::ImageBuffer() : width_(0), height_(0), channels_(0) {}

ImageBuffer::ImageBuffer(int width, int height, int channels)
    : width_(width),
      height_(height),
      channels_(channels),
      data_(static_cast<size_t>(width) * height * channels, 0.0f) {}

ImageBuffer::ImageBuffer(const ImageBuffer& other)
    : width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      data_(other.data_) {}

ImageBuffer& ImageBuffer::operator=(const ImageBuffer& other) {
  if (this != &other) {
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    data_ = other.data_;
  }
  return *this;
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      data_(std::move(other.data_)) {
  other.width_ = 0;
  other.height_ = 0;
  other.channels_ = 0;
}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
  if (this != &other) {
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    data_ = std::move(other.data_);
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
  }
  return *this;
}

ImageBuffer ImageBuffer::CloneEmpty() const {
  return ImageBuffer(width_, height_, channels_);
}

ImageBuffer ImageBuffer::LoadFromFile(const std::string& file_path) {
  int w = 0;
  int h = 0;
  int c = 0;
  unsigned char* raw = stbi_load(file_path.c_str(), &w, &h, &c, 0);
  if (!raw) {
    std::cerr << "[ImageBuffer] Error loading image file: " << file_path
              << " (" << stbi_failure_reason() << ")" << std::endl;
    return ImageBuffer();
  }

  // Support 1, 3, or 4 channels (clamp 4 channels to 3 RGB if preferred, or keep 4).
  int target_channels = (c >= 3) ? 3 : 1;
  ImageBuffer img(w, h, target_channels);

  // Convert interleaved uint8 [0..255] to planar float [0.0f..1.0f]
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int src_idx = (y * w + x) * c;
      if (target_channels == 1) {
        img.At(0, y, x) = static_cast<float>(raw[src_idx]) / 255.0f;
      } else {
        img.At(0, y, x) = static_cast<float>(raw[src_idx + 0]) / 255.0f;
        img.At(1, y, x) = static_cast<float>(raw[src_idx + 1]) / 255.0f;
        img.At(2, y, x) = static_cast<float>(raw[src_idx + 2]) / 255.0f;
      }
    }
  }

  stbi_image_free(raw);
  return img;
}

bool ImageBuffer::SaveToPng(const std::string& file_path) const {
  if (empty()) {
    std::cerr << "[ImageBuffer] Cannot save empty image to " << file_path << std::endl;
    return false;
  }

  std::vector<unsigned char> interleaved(total_pixels() * channels_);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int dst_idx = (y * width_ + x) * channels_;
      for (int c = 0; c < channels_; ++c) {
        float val = At(c, y, x);
        float clamped = std::clamp(val, 0.0f, 1.0f);
        interleaved[dst_idx + c] = static_cast<unsigned char>(std::round(clamped * 255.0f));
      }
    }
  }

  int stride_bytes = width_ * channels_;
  int ret = stbi_write_png(file_path.c_str(), width_, height_, channels_,
                           interleaved.data(), stride_bytes);
  return ret != 0;
}

ImageBuffer ImageBuffer::ToGrayscale() const {
  if (empty()) {
    return ImageBuffer();
  }
  if (channels_ == 1) {
    return *this;
  }

  ImageBuffer gray(width_, height_, 1);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      float r = At(0, y, x);
      float g = At(1, y, x);
      float b = (channels_ > 2) ? At(2, y, x) : g;
      // Rec. 601 standard weights
      float luma = 0.299f * r + 0.587f * g + 0.114f * b;
      gray.At(0, y, x) = std::clamp(luma, 0.0f, 1.0f);
    }
  }
  return gray;
}

ImageBuffer ImageBuffer::CreateSyntheticPattern(int width, int height, int pattern_type) {
  ImageBuffer img(width, height, 1);
  float cx = width / 2.0f;
  float cy = height / 2.0f;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float val = 0.0f;
      if (pattern_type == 0) {
        // Concentric sinusoidal rings (high-frequency frequency response benchmark)
        float dx = x - cx;
        float dy = y - cy;
        float dist = std::sqrt(dx * dx + dy * dy);
        val = 0.5f + 0.5f * std::sin(dist * 0.15f);
      } else if (pattern_type == 1) {
        // High-contrast geometric edges (Sobel benchmark)
        bool check_x = ((x / 32) % 2) == 0;
        bool check_y = ((y / 32) % 2) == 0;
        val = (check_x ^ check_y) ? 0.9f : 0.1f;
      } else {
        // Linear smooth gradient
        val = static_cast<float>(x + y) / static_cast<float>(width + height);
      }
      img.At(0, y, x) = val;
    }
  }
  return img;
}

double ImageBuffer::CalculateMse(const ImageBuffer& img_a, const ImageBuffer& img_b) {
  if (img_a.width() != img_b.width() || img_a.height() != img_b.height() ||
      img_a.channels() != img_b.channels()) {
    std::cerr << "[ImageBuffer] Dimension mismatch in CalculateMse" << std::endl;
    return -1.0;
  }

  double sum_sq = 0.0;
  size_t n = img_a.total_elements();
  const float* a = img_a.Data();
  const float* b = img_b.Data();

  for (size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum_sq += diff * diff;
  }
  return sum_sq / static_cast<double>(n);
}

double ImageBuffer::CalculatePsnr(const ImageBuffer& img_a, const ImageBuffer& img_b) {
  double mse = CalculateMse(img_a, img_b);
  if (mse < 0.0) return -1.0;
  if (mse < 1e-12) return 100.0;  // Identical signals
  double max_val = 1.0;  // Normalized scale
  return 10.0 * std::log10((max_val * max_val) / mse);
}

}  // namespace cuda_vision
