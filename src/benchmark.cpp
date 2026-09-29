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

#include "cuda_vision/benchmark.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace cuda_vision {

BenchmarkSuite::BenchmarkSuite(const PipelineConfig& config)
    : config_(config), cpu_engine_(config.num_threads) {}

void BenchmarkSuite::RunFullBenchmark() {
  results_.clear();
  std::cout << "\n======================================================================\n"
            << "              STARTING ENTERPRISE PERFORMANCE BENCHMARK               \n"
            << "======================================================================\n";
  std::cout << cpu_engine_.GetCpuInformation() << "\n";

  std::vector<int> test_sizes = {512, 1024, 2048};

  for (int size : test_sizes) {
    std::cout << "\n>>> Benchmarking Resolution: " << size << "x" << size << " ("
              << (size * size) / 1000000.0 << " MPixels) <<<\n";

    ImageBuffer test_img = ImageBuffer::CreateSyntheticPattern(size, size, 0);

    // 1. Grayscale
    {
      ImageBuffer ref_out, multi_out;
      KernelStats single = cpu_engine_.RunGrayscale(test_img, &ref_out, false);
      KernelStats multi = cpu_engine_.RunGrayscale(test_img, &multi_out, true);
      multi.mse_vs_ref = ImageBuffer::CalculateMse(ref_out, multi_out);
      multi.psnr_vs_ref = ImageBuffer::CalculatePsnr(ref_out, multi_out);

      results_.push_back(single);
      results_.push_back(multi);
      std::cout << "  [Grayscale] Single: " << std::fixed << std::setprecision(2)
                << single.elapsed_ms << " ms (" << single.throughput_mpixels_sec
                << " MP/s) | Multi: " << multi.elapsed_ms << " ms ("
                << multi.throughput_mpixels_sec << " MP/s) -> Speedup: "
                << (single.elapsed_ms / multi.elapsed_ms) << "x\n";
    }

    // 2. Gaussian Blur (5x5, radius 2)
    {
      ImageBuffer ref_out, multi_out, emu_out;
      KernelStats single = cpu_engine_.RunGaussianBlur(test_img, config_.kernel_radius,
                                                       config_.sigma_spatial, &ref_out, false);
      KernelStats multi = cpu_engine_.RunGaussianBlur(test_img, config_.kernel_radius,
                                                      config_.sigma_spatial, &multi_out, true);
      KernelStats emu = cpu_engine_.RunCudaEmulatedGaussianBlur(test_img, config_.kernel_radius,
                                                               config_.sigma_spatial, &emu_out);

      multi.mse_vs_ref = ImageBuffer::CalculateMse(ref_out, multi_out);
      multi.psnr_vs_ref = ImageBuffer::CalculatePsnr(ref_out, multi_out);
      emu.mse_vs_ref = ImageBuffer::CalculateMse(ref_out, emu_out);
      emu.psnr_vs_ref = ImageBuffer::CalculatePsnr(ref_out, emu_out);

      results_.push_back(single);
      results_.push_back(multi);
      results_.push_back(emu);
      std::cout << "  [Gaussian Blur] Single: " << single.elapsed_ms << " ms | Multi: "
                << multi.elapsed_ms << " ms | CUDA Tile Emulation: " << emu.elapsed_ms
                << " ms (PSNR: " << emu.psnr_vs_ref << " dB) -> Speedup: "
                << (single.elapsed_ms / multi.elapsed_ms) << "x\n";
    }

    // 3. Sobel Edge Detection
    {
      ImageBuffer ref_out, multi_out;
      KernelStats single = cpu_engine_.RunSobelEdges(test_img, &ref_out, false);
      KernelStats multi = cpu_engine_.RunSobelEdges(test_img, &multi_out, true);
      multi.mse_vs_ref = ImageBuffer::CalculateMse(ref_out, multi_out);
      multi.psnr_vs_ref = ImageBuffer::CalculatePsnr(ref_out, multi_out);

      results_.push_back(single);
      results_.push_back(multi);
      std::cout << "  [Sobel Edges]  Single: " << single.elapsed_ms << " ms | Multi: "
                << multi.elapsed_ms << " ms -> Speedup: "
                << (single.elapsed_ms / multi.elapsed_ms) << "x\n";
    }

    // 4. Bilateral Filter
    {
      ImageBuffer ref_out, multi_out;
      KernelStats single = cpu_engine_.RunBilateralFilter(test_img, config_.kernel_radius,
                                                         config_.sigma_spatial, config_.sigma_range,
                                                         &ref_out, false);
      KernelStats multi = cpu_engine_.RunBilateralFilter(test_img, config_.kernel_radius,
                                                        config_.sigma_spatial, config_.sigma_range,
                                                        &multi_out, true);
      multi.mse_vs_ref = ImageBuffer::CalculateMse(ref_out, multi_out);
      multi.psnr_vs_ref = ImageBuffer::CalculatePsnr(ref_out, multi_out);

      results_.push_back(single);
      results_.push_back(multi);
      std::cout << "  [Bilateral]    Single: " << single.elapsed_ms << " ms | Multi: "
                << multi.elapsed_ms << " ms -> Speedup: "
                << (single.elapsed_ms / multi.elapsed_ms) << "x\n";
    }
  }

  // 5. Batched 1D Signal Processing
  std::cout << "\n>>> Benchmarking Batched 1D FIR Filtering ("
            << config_.signal_batch_count << " signals x "
            << config_.signal_length << " samples = "
            << (config_.signal_batch_count * config_.signal_length) / 1000000.0
            << " MSamples) <<<\n";
  {
    SignalBatch batch = SignalBatch::CreateSyntheticBatch(config_.signal_batch_count,
                                                          config_.signal_length);
    std::vector<float> taps = SignalBatch::CreateLowPassFirKernel(config_.fir_taps, 0.15f);

    SignalBatch ref_out, multi_out;
    KernelStats single = cpu_engine_.RunBatchedFir(batch, taps, &ref_out, false);
    KernelStats multi = cpu_engine_.RunBatchedFir(batch, taps, &multi_out, true);

    results_.push_back(single);
    results_.push_back(multi);
    std::cout << "  [Batched FIR]  Single: " << single.elapsed_ms << " ms ("
              << single.throughput_mpixels_sec << " MSamples/s) | Multi: "
              << multi.elapsed_ms << " ms (" << multi.throughput_mpixels_sec
              << " MSamples/s) -> Speedup: "
              << (single.elapsed_ms / multi.elapsed_ms) << "x\n";
  }

  std::cout << "\n======================================================================\n"
            << "                     BENCHMARK SUITE COMPLETED                        \n"
            << "======================================================================\n";
}

bool BenchmarkSuite::ExportToCsv(const std::string& csv_path) const {
  std::ofstream out(csv_path);
  if (!out.is_open()) {
    std::cerr << "[BenchmarkSuite] Error writing to " << csv_path << std::endl;
    return false;
  }

  out << "filter_name,backend,width,height,channels,radius,elapsed_ms,"
      << "throughput_mpixels_sec,mse_vs_ref,psnr_db\n";

  for (const auto& r : results_) {
    out << "\"" << r.filter_name << "\",\""
        << r.backend_name << "\","
        << r.width << ","
        << r.height << ","
        << r.channels << ","
        << r.kernel_radius << ","
        << std::fixed << std::setprecision(3) << r.elapsed_ms << ","
        << std::fixed << std::setprecision(2) << r.throughput_mpixels_sec << ","
        << std::scientific << std::setprecision(6) << r.mse_vs_ref << ","
        << std::fixed << std::setprecision(2) << r.psnr_vs_ref << "\n";
  }

  return true;
}

void BenchmarkSuite::PrintSummaryTable() const {
  std::cout << "\n"
            << "----------------------------------------------------------------------------------------------------\n"
            << std::left << std::setw(28) << "Kernel Filter"
            << std::left << std::setw(28) << "Backend"
            << std::left << std::setw(14) << "Resolution"
            << std::right << std::setw(12) << "Time (ms)"
            << std::right << std::setw(16) << "Throughput(MP/s)"
            << "\n"
            << "----------------------------------------------------------------------------------------------------\n";

  for (const auto& r : results_) {
    std::string dim = (r.height > 0)
                          ? (std::to_string(r.width) + "x" + std::to_string(r.height))
                          : (std::to_string(r.batch_size) + "x" + std::to_string(r.width));
    std::cout << std::left << std::setw(28) << r.filter_name
              << std::left << std::setw(28) << r.backend_name
              << std::left << std::setw(14) << dim
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << r.elapsed_ms
              << std::right << std::setw(16) << std::fixed << std::setprecision(2) << r.throughput_mpixels_sec
              << "\n";
  }
  std::cout << "----------------------------------------------------------------------------------------------------\n\n";
}

}  // namespace cuda_vision
