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

#include "cuda_vision/signal_buffer.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cuda_vision {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

SignalBatch::SignalBatch() : num_signals_(0), signal_length_(0) {}

SignalBatch::SignalBatch(int num_signals, int signal_length)
    : num_signals_(num_signals),
      signal_length_(signal_length),
      data_(static_cast<size_t>(num_signals) * signal_length, 0.0f) {}

SignalBatch::SignalBatch(const SignalBatch& other)
    : num_signals_(other.num_signals_),
      signal_length_(other.signal_length_),
      data_(other.data_) {}

SignalBatch& SignalBatch::operator=(const SignalBatch& other) {
  if (this != &other) {
    num_signals_ = other.num_signals_;
    signal_length_ = other.signal_length_;
    data_ = other.data_;
  }
  return *this;
}

SignalBatch::SignalBatch(SignalBatch&& other) noexcept
    : num_signals_(other.num_signals_),
      signal_length_(other.signal_length_),
      data_(std::move(other.data_)) {
  other.num_signals_ = 0;
  other.signal_length_ = 0;
}

SignalBatch& SignalBatch::operator=(SignalBatch&& other) noexcept {
  if (this != &other) {
    num_signals_ = other.num_signals_;
    signal_length_ = other.signal_length_;
    data_ = std::move(other.data_);
    other.num_signals_ = 0;
    other.signal_length_ = 0;
  }
  return *this;
}

SignalBatch SignalBatch::CloneEmpty() const {
  return SignalBatch(num_signals_, signal_length_);
}

SignalBatch SignalBatch::CreateSyntheticBatch(int num_signals, int signal_length,
                                              float sampling_rate_hz) {
  SignalBatch batch(num_signals, signal_length);

  for (int s = 0; s < num_signals; ++s) {
    // Generate distinct fundamental frequency per signal channel
    float base_freq = 220.0f + static_cast<float>(s % 32) * 20.0f;
    float phase_shift = static_cast<float>(s) * 0.1f;

    for (int t = 0; t < signal_length; ++t) {
      float time_sec = static_cast<float>(t) / sampling_rate_hz;

      // Base carrier tone + 3rd harmonic + high frequency sensor noise
      float signal = 0.6f * std::sin(2.0f * static_cast<float>(kPi) * base_freq * time_sec + phase_shift);
      signal += 0.25f * std::sin(2.0f * static_cast<float>(kPi) * (3.0f * base_freq) * time_sec);
      // High-frequency interference / noise at 8000 Hz
      signal += 0.15f * std::sin(2.0f * static_cast<float>(kPi) * 8000.0f * time_sec);

      batch.At(s, t) = signal;
    }
  }

  return batch;
}

std::vector<float> SignalBatch::CreateLowPassFirKernel(int num_taps,
                                                      float cutoff_normalized) {
  std::vector<float> kernel(num_taps, 0.0f);
  int mid = (num_taps - 1) / 2;
  double sum = 0.0;

  for (int i = 0; i < num_taps; ++i) {
    double n = i - mid;
    // Sinc function
    double sinc = 0.0;
    if (std::abs(n) < 1e-7) {
      sinc = 2.0 * cutoff_normalized;
    } else {
      sinc = std::sin(2.0 * kPi * cutoff_normalized * n) / (kPi * n);
    }

    // Hamming window: 0.54 - 0.46 * cos(2 * pi * i / (M - 1))
    double hamming = 0.54 - 0.46 * std::cos(2.0 * kPi * i / (num_taps - 1));
    kernel[i] = static_cast<float>(sinc * hamming);
    sum += kernel[i];
  }

  // Normalize coefficients to achieve unity DC gain
  if (std::abs(sum) > 1e-7) {
    for (int i = 0; i < num_taps; ++i) {
      kernel[i] = static_cast<float>(kernel[i] / sum);
    }
  }

  return kernel;
}

bool SignalBatch::SaveToCsv(const std::string& file_path,
                            int max_signals_to_export) const {
  std::ofstream out(file_path);
  if (!out.is_open()) {
    std::cerr << "[SignalBatch] Unable to open CSV file for writing: "
              << file_path << std::endl;
    return false;
  }

  int signals_to_write = std::min(num_signals_, max_signals_to_export);

  // Write CSV Header
  out << "sample_index";
  for (int s = 0; s < signals_to_write; ++s) {
    out << ",signal_" << s;
  }
  out << "\n";

  // Write Samples
  out << std::fixed << std::setprecision(6);
  for (int t = 0; t < signal_length_; ++t) {
    out << t;
    for (int s = 0; s < signals_to_write; ++s) {
      out << "," << At(s, t);
    }
    out << "\n";
  }

  return true;
}

double SignalBatch::CalculateBatchRms() const {
  if (data_.empty()) return 0.0;
  double sum_sq = 0.0;
  for (float val : data_) {
    sum_sq += static_cast<double>(val) * val;
  }
  return std::sqrt(sum_sq / static_cast<double>(data_.size()));
}

}  // namespace cuda_vision
