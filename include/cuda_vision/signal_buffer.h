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

#ifndef CUDA_VISION_SIGNAL_BUFFER_H_
#define CUDA_VISION_SIGNAL_BUFFER_H_

#include <cstddef>
#include <string>
#include <vector>

namespace cuda_vision {

// Represents a batch of 1D time-series signals for high-throughput stream processing.
class SignalBatch {
 public:
  SignalBatch();
  SignalBatch(int num_signals, int signal_length);

  // Copy and move constructors/operators.
  SignalBatch(const SignalBatch& other);
  SignalBatch& operator=(const SignalBatch& other);
  SignalBatch(SignalBatch&& other) noexcept;
  SignalBatch& operator=(SignalBatch&& other) noexcept;

  ~SignalBatch() = default;

  // Factory generator: creates a batch of synthetic multi-channel audio/sensor signals
  // with harmonic frequencies, phase variations, and additive noise.
  static SignalBatch CreateSyntheticBatch(int num_signals, int signal_length,
                                          float sampling_rate_hz = 44100.0f);

  // Generates 1D FIR Low-Pass filter coefficients using windowed sinc (Hamming window).
  static std::vector<float> CreateLowPassFirKernel(int num_taps, float cutoff_normalized);

  // CSV I/O for Coursera proof of execution artifacts.
  bool SaveToCsv(const std::string& file_path, int max_signals_to_export = 10) const;
  static SignalBatch LoadFromCsv(const std::string& file_path);

  // Clones empty batch with same dimensions.
  SignalBatch CloneEmpty() const;

  // Accessors.
  int num_signals() const { return num_signals_; }
  int signal_length() const { return signal_length_; }
  size_t total_samples() const {
    return static_cast<size_t>(num_signals_) * signal_length_;
  }
  size_t size_bytes() const { return total_samples() * sizeof(float); }
  bool empty() const { return data_.empty(); }

  float* Data() { return data_.data(); }
  const float* Data() const { return data_.data(); }

  float& At(int signal_idx, int sample_idx) {
    return data_[static_cast<size_t>(signal_idx) * signal_length_ + sample_idx];
  }

  const float& At(int signal_idx, int sample_idx) const {
    return data_[static_cast<size_t>(signal_idx) * signal_length_ + sample_idx];
  }

  // Compute Root Mean Square (RMS) energy across the batch.
  double CalculateBatchRms() const;

 private:
  int num_signals_ = 0;
  int signal_length_ = 0;
  std::vector<float> data_;
};

}  // namespace cuda_vision

#endif  // CUDA_VISION_SIGNAL_BUFFER_H_
