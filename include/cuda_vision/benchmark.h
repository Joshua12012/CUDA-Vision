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

#ifndef CUDA_VISION_BENCHMARK_H_
#define CUDA_VISION_BENCHMARK_H_

#include <string>
#include <vector>

#include "cuda_vision/common.h"
#include "cuda_vision/cpu_engine.h"

namespace cuda_vision {

// Automated performance benchmarking and CSV reporting harness.
class BenchmarkSuite {
 public:
  explicit BenchmarkSuite(const PipelineConfig& config);
  ~BenchmarkSuite() = default;

  // Run full multi-scale benchmarks and generate CSV and log artifacts.
  void RunFullBenchmark();

  // Export benchmark results to CSV.
  bool ExportToCsv(const std::string& csv_path) const;

  // Print results table to console.
  void PrintSummaryTable() const;

  const std::vector<KernelStats>& results() const { return results_; }

 private:
  PipelineConfig config_;
  CpuEngine cpu_engine_;
  std::vector<KernelStats> results_;
};

}  // namespace cuda_vision

#endif  // CUDA_VISION_BENCHMARK_H_
