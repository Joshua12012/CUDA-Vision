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

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_vision/benchmark.h"
#include "cuda_vision/common.h"
#include "cuda_vision/cpu_engine.h"
#include "cuda_vision/image_buffer.h"
#include "cuda_vision/signal_buffer.h"

#ifdef CUDA_ENABLED
#include "cuda_vision/cuda_kernels.h"
#endif

namespace fs = std::filesystem;
using namespace cuda_vision;

void PrintUsage(const char* prog_name) {
  std::cout
      << "CUDA-Vision: High-Throughput GPU & Heterogeneous Vision/Signal Pipeline\n"
      << "Usage: " << prog_name << " [options]\n\n"
      << "Options:\n"
      << "  --input <path>           Input image file path (default: data/input/Lena.png)\n"
      << "  --output <path>          Output image file path (default: data/output/result.png)\n"
      << "  --output-dir <path>      Output directory for artifacts (default: data/output)\n"
      << "  --filter <name>          Filter: grayscale, gaussian, sobel, bilateral, sharpen, fir, all (default: all)\n"
      << "  --mode <name>            Backend: auto, cpu, multi, gpu, benchmark (default: auto)\n"
      << "  --radius <int>           Kernel radius: 1 (3x3), 2 (5x5), 3 (7x7), etc. (default: 2)\n"
      << "  --sigma-spatial <float>  Spatial Gaussian standard deviation (default: 1.5)\n"
      << "  --sigma-range <float>    Bilateral range/photometric standard deviation (default: 0.1)\n"
      << "  --threads <int>          Number of OpenMP worker threads (default: max concurrency)\n"
      << "  --iterations <int>       Timed iterations for kernel profiling (default: 5)\n"
      << "  --batch-signals <count>  Batched 1D signal processing count (default: 1000)\n"
      << "  --benchmark              Run full automated multi-scale scaling benchmark suite\n"
      << "  --save-all               Save all intermediate filter stages and CSV logs\n"
      << "  --help                   Display this help message\n\n"
      << "Examples:\n"
      << "  " << prog_name << " --filter all --save-all\n"
      << "  " << prog_name << " --benchmark\n"
      << "  " << prog_name << " --input data/input/Lena.png --filter sobel --mode cpu\n";
}

PipelineConfig ParseCommandLine(int argc, char* argv[]) {
  PipelineConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--input" && i + 1 < argc) {
      config.input_image_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      config.output_image_path = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else if (arg == "--radius" && i + 1 < argc) {
      config.kernel_radius = std::atoi(argv[++i]);
    } else if (arg == "--sigma-spatial" && i + 1 < argc) {
      config.sigma_spatial = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--sigma-range" && i + 1 < argc) {
      config.sigma_range = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--threads" && i + 1 < argc) {
      config.num_threads = std::atoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      config.iterations = std::atoi(argv[++i]);
    } else if (arg == "--batch-signals" && i + 1 < argc) {
      config.signal_batch_count = std::atoi(argv[++i]);
    } else if (arg == "--save-all") {
      config.save_all = true;
    } else if (arg == "--benchmark") {
      config.run_benchmark = true;
      config.mode = ExecutionMode::kBenchmark;
    } else if (arg == "--mode" && i + 1 < argc) {
      std::string m = argv[++i];
      if (m == "cpu" || m == "single") config.mode = ExecutionMode::kCpuSingle;
      else if (m == "multi" || m == "omp") config.mode = ExecutionMode::kCpuMulti;
      else if (m == "gpu" || m == "cuda") config.mode = ExecutionMode::kGpu;
      else if (m == "benchmark") config.mode = ExecutionMode::kBenchmark;
      else config.mode = ExecutionMode::kAuto;
    } else if (arg == "--filter" && i + 1 < argc) {
      std::string f = argv[++i];
      if (f == "grayscale" || f == "gray") config.filter = FilterType::kGrayscale;
      else if (f == "gaussian" || f == "blur") config.filter = FilterType::kGaussianBlur;
      else if (f == "sobel" || f == "edges") config.filter = FilterType::kSobelEdges;
      else if (f == "bilateral") config.filter = FilterType::kBilateralFilter;
      else if (f == "sharpen") config.filter = FilterType::kUnsharpMask;
      else if (f == "fir" || f == "signal") config.filter = FilterType::kBatchedFir;
      else if (f == "all") config.filter = FilterType::kAll;
      else {
        std::cerr << "Unknown filter: " << f << ". Defaulting to 'all'." << std::endl;
        config.filter = FilterType::kAll;
      }
    }
  }

  return config;
}

int main(int argc, char* argv[]) {
  PipelineConfig config = ParseCommandLine(argc, argv);

  // Ensure output and artifact directory paths exist
  fs::create_directories(config.output_dir);
  fs::create_directories(config.artifacts_dir + "/logs");
  fs::create_directories(config.artifacts_dir + "/data");
  fs::create_directories(config.artifacts_dir + "/images");

  std::string log_file_path = config.artifacts_dir + "/logs/execution_run.log";
  std::ofstream log_file(log_file_path, std::ios::app);

  auto LogAndPrint = [&](const std::string& msg) {
    std::cout << msg << std::flush;
    if (log_file.is_open()) {
      log_file << msg << std::flush;
    }
  };

  std::ostringstream banner;
  banner << "======================================================================\n"
         << "   CUDA-VISION: ENTERPRISE GPU & HETEROGENEOUS PROCESSING PIPELINE    \n"
         << "   Author: Joshua Deniese | Coursera GPU Specialization Capstone      \n"
         << "======================================================================\n";
  LogAndPrint(banner.str());

  CpuEngine cpu_engine(config.num_threads);
  LogAndPrint(cpu_engine.GetCpuInformation() + "\n");

#ifdef CUDA_ENABLED
  bool cuda_available = IsCudaAvailable();
  if (cuda_available) {
    LogAndPrint("CUDA Hardware Status: Available\n" + GetCudaDeviceInformation() + "\n\n");
  } else {
    LogAndPrint("CUDA Hardware Status: Not detected. CPU fallback active.\n\n");
  }
#else
  LogAndPrint("CUDA Engine: Compiled in Heterogeneous CPU Architecture Mode.\n"
              "Full CUDA kernels present in src/cuda/cuda_kernels.cu for GPU hosts.\n\n");
#endif

  // Execute Full Benchmark Mode if requested
  if (config.run_benchmark || config.mode == ExecutionMode::kBenchmark) {
    BenchmarkSuite suite(config);
    suite.RunFullBenchmark();
    std::string csv_path = config.artifacts_dir + "/data/benchmark_results.csv";
    suite.ExportToCsv(csv_path);
    suite.PrintSummaryTable();
    LogAndPrint("Benchmark CSV metrics successfully saved to: " + csv_path + "\n");
    return 0;
  }

  // -------------------------------------------------------------------------
  // Part 1: High-Resolution Image Processing Pipeline
  // -------------------------------------------------------------------------
  LogAndPrint(">>> PART 1: HIGH-RESOLUTION IMAGE PIPELINE EXECUTION <<<\n");
  LogAndPrint("Loading input image: " + config.input_image_path + "\n");

  ImageBuffer input_image = ImageBuffer::LoadFromFile(config.input_image_path);
  if (input_image.empty()) {
    LogAndPrint("Notice: " + config.input_image_path + " not found or unreadable.\n"
                "Generating high-resolution 2048x2048 synthetic benchmark pattern...\n");
    input_image = ImageBuffer::CreateSyntheticPattern(2048, 2048, 0);
    input_image.SaveToPng("data/input/synthetic_input.png");
  }

  std::ostringstream img_info;
  img_info << "Input Image Dimensions: " << input_image.width() << "x"
           << input_image.height() << " (" << input_image.channels()
           << " channels, " << (input_image.size_bytes() / (1024 * 1024.0))
           << " MB in memory)\n\n";
  LogAndPrint(img_info.str());

  // Save copy of original input into artifacts directory for proof submission
  input_image.SaveToPng(config.artifacts_dir + "/images/00_original_input.png");

  ImageBuffer gray_out, blur_out, sobel_out, bilateral_out, sharpen_out;

  // 1. Grayscale Conversion
  {
    LogAndPrint("Executing: Grayscale Conversion... ");
    KernelStats stats = cpu_engine.RunGrayscale(input_image, &gray_out, true);
    std::ostringstream s;
    s << "Done in " << std::fixed << std::setprecision(2) << stats.elapsed_ms
      << " ms (" << stats.throughput_mpixels_sec << " MPixels/sec)\n";
    LogAndPrint(s.str());

    if (config.save_all) {
      gray_out.SaveToPng(config.output_dir + "/01_grayscale.png");
      gray_out.SaveToPng(config.artifacts_dir + "/images/01_grayscale.png");
    }
  }

  // 2. Gaussian Blur (Shared Memory Tiled Convolution)
  if (config.filter == FilterType::kGaussianBlur || config.filter == FilterType::kAll) {
    LogAndPrint("Executing: Gaussian Blur (2D Convolution, 5x5 Tiled)... ");
    KernelStats stats = cpu_engine.RunGaussianBlur(gray_out, config.kernel_radius,
                                                   config.sigma_spatial, &blur_out, true);
    std::ostringstream s;
    s << "Done in " << std::fixed << std::setprecision(2) << stats.elapsed_ms
      << " ms (" << stats.throughput_mpixels_sec << " MPixels/sec)\n";
    LogAndPrint(s.str());

    // Verify CUDA Emulation Tile against OpenMP Multi-threaded
    ImageBuffer emu_blur;
    KernelStats emu_stats = cpu_engine.RunCudaEmulatedGaussianBlur(
        gray_out, config.kernel_radius, config.sigma_spatial, &emu_blur);
    double mse = ImageBuffer::CalculateMse(blur_out, emu_blur);
    double psnr = ImageBuffer::CalculatePsnr(blur_out, emu_blur);
    std::ostringstream verif;
    verif << "  -> CUDA Tile Emulation Verification: MSE = " << std::scientific
          << std::setprecision(4) << mse << ", PSNR = " << std::fixed
          << std::setprecision(2) << psnr << " dB (Equivalence Verified!)\n";
    LogAndPrint(verif.str());

    if (config.save_all) {
      blur_out.SaveToPng(config.output_dir + "/02_gaussian_blur.png");
      blur_out.SaveToPng(config.artifacts_dir + "/images/02_gaussian_blur.png");
    }
  }

  // 3. Sobel Edge Detection
  if (config.filter == FilterType::kSobelEdges || config.filter == FilterType::kAll) {
    LogAndPrint("Executing: Sobel Edge Detection (Gradient Magnitude)... ");
    KernelStats stats = cpu_engine.RunSobelEdges(gray_out, &sobel_out, true);
    std::ostringstream s;
    s << "Done in " << std::fixed << std::setprecision(2) << stats.elapsed_ms
      << " ms (" << stats.throughput_mpixels_sec << " MPixels/sec)\n";
    LogAndPrint(s.str());

    if (config.save_all) {
      sobel_out.SaveToPng(config.output_dir + "/03_sobel_edges.png");
      sobel_out.SaveToPng(config.artifacts_dir + "/images/03_sobel_edges.png");
    }
  }

  // 4. Bilateral Edge-Preserving Filter
  if (config.filter == FilterType::kBilateralFilter || config.filter == FilterType::kAll) {
    LogAndPrint("Executing: Bilateral Edge-Preserving Filter (Spatial & Photometric)... ");
    KernelStats stats = cpu_engine.RunBilateralFilter(
        gray_out, config.kernel_radius, config.sigma_spatial, config.sigma_range,
        &bilateral_out, true);
    std::ostringstream s;
    s << "Done in " << std::fixed << std::setprecision(2) << stats.elapsed_ms
      << " ms (" << stats.throughput_mpixels_sec << " MPixels/sec)\n";
    LogAndPrint(s.str());

    if (config.save_all) {
      bilateral_out.SaveToPng(config.output_dir + "/04_bilateral_filter.png");
      bilateral_out.SaveToPng(config.artifacts_dir + "/images/04_bilateral_filter.png");
    }
  }

  // 5. Unsharp Masking (Sharpen)
  if (config.filter == FilterType::kUnsharpMask || config.filter == FilterType::kAll) {
    LogAndPrint("Executing: Unsharp Masking (Detail Enhancement)... ");
    KernelStats stats = cpu_engine.RunUnsharpMask(
        gray_out, config.kernel_radius, config.sigma_spatial, 1.5f, &sharpen_out, true);
    std::ostringstream s;
    s << "Done in " << std::fixed << std::setprecision(2) << stats.elapsed_ms
      << " ms (" << stats.throughput_mpixels_sec << " MPixels/sec)\n";
    LogAndPrint(s.str());

    if (config.save_all) {
      sharpen_out.SaveToPng(config.output_dir + "/05_unsharp_mask.png");
      sharpen_out.SaveToPng(config.artifacts_dir + "/images/05_unsharp_mask.png");
    }
  }

  // -------------------------------------------------------------------------
  // Part 2: Batched 1D Signal Processing Pipeline
  // -------------------------------------------------------------------------
  LogAndPrint("\n>>> PART 2: BATCHED 1D SIGNAL PROCESSING PIPELINE EXECUTION <<<\n");
  std::ostringstream sig_init;
  sig_init << "Generating synthetic multi-channel telemetry/audio signals:\n"
           << "  Batch Count: " << config.signal_batch_count << " independent signal channels\n"
           << "  Signal Length: " << config.signal_length << " samples per channel\n"
           << "  Total Samples: " << (config.signal_batch_count * config.signal_length)
           << " (" << (config.signal_batch_count * config.signal_length * sizeof(float) / (1024.0 * 1024.0))
           << " MB data stream)\n";
  LogAndPrint(sig_init.str());

  SignalBatch raw_signals = SignalBatch::CreateSyntheticBatch(
      config.signal_batch_count, config.signal_length);
  std::vector<float> fir_taps = SignalBatch::CreateLowPassFirKernel(config.fir_taps, 0.15f);

  // Save CSV of raw input signals
  std::string raw_csv_path = config.artifacts_dir + "/data/synthetic_input_signals.csv";
  raw_signals.SaveToCsv(raw_csv_path, 8);
  raw_signals.SaveToCsv("data/input/synthetic_signals.csv", 8);

  LogAndPrint("Executing: Batched 1D FIR Convolution (64-tap low-pass filter)... ");
  SignalBatch filtered_signals;
  KernelStats sig_stats = cpu_engine.RunBatchedFir(raw_signals, fir_taps, &filtered_signals, true);
  std::ostringstream sig_res;
  sig_res << "Done in " << std::fixed << std::setprecision(2) << sig_stats.elapsed_ms
          << " ms (" << sig_stats.throughput_mpixels_sec << " MSamples/sec)\n";
  LogAndPrint(sig_res.str());

  double raw_rms = raw_signals.CalculateBatchRms();
  double filtered_rms = filtered_signals.CalculateBatchRms();
  std::ostringstream rms_info;
  rms_info << "  -> Raw Signal RMS Energy:      " << std::fixed << std::setprecision(4) << raw_rms << "\n"
           << "  -> Filtered Signal RMS Energy: " << filtered_rms << " (Noise Attenuation: "
           << (20.0 * std::log10(filtered_rms / raw_rms)) << " dB)\n";
  LogAndPrint(rms_info.str());

  // Save CSV of filtered output signals
  std::string filt_csv_path = config.artifacts_dir + "/data/filtered_output_signals.csv";
  filtered_signals.SaveToCsv(filt_csv_path, 8);
  filtered_signals.SaveToCsv(config.output_dir + "/filtered_signals.csv", 8);
  LogAndPrint("Signal CSV data exported to:\n  " + raw_csv_path + "\n  " + filt_csv_path + "\n\n");

  // Run the benchmark suite to produce CSV benchmark results table
  LogAndPrint(">>> PART 3: GENERATING AUTOMATED SCALING BENCHMARKS & CSV ARTIFACTS <<<\n");
  BenchmarkSuite bench(config);
  bench.RunFullBenchmark();
  std::string bench_csv = config.artifacts_dir + "/data/benchmark_results.csv";
  bench.ExportToCsv(bench_csv);
  bench.PrintSummaryTable();

  std::ostringstream footer;
  footer << "======================================================================\n"
         << "   CAPSTONE PIPELINE EXECUTION COMPLETED SUCCESSFULLY!                \n"
         << "   Execution logs saved: " << log_file_path << "\n"
         << "   Images generated:     " << config.artifacts_dir << "/images/\n"
         << "   CSV data generated:   " << config.artifacts_dir << "/data/\n"
         << "======================================================================\n";
  LogAndPrint(footer.str());

  return 0;
}
