# CUDA-Vision: Capstone Video Demonstration & Presentation Script

**Project Title:** CUDA-Accelerated High-Throughput Vision & Signal Processing Pipeline  
**Author:** Joshua Deniese  
**Specialization:** Coursera GPU Specialization: CUDA at Scale for the Enterprise  
**Target Video Length:** 7:30 - 8:00 Minutes (Meets Rubric requirement: 5 to 10 minutes)  

---

## Preparation Checklist Before Recording
1. **Screen Layout:** Open your slide deck (`PRESENTATION_SLIDES.md`) or slides in one window, and a clean Linux terminal in another.
2. **Terminal Directory:** Ensure you are in `/home/joshua/cuda-adv-lib`.
3. **Artifacts Ready:** Make sure `make all` has been run and images in `artifacts/images/` can be viewed if desired.
4. **Recording Tool:** OBS Studio, SimpleScreenRecorder, Zoom, or your favorite screen recorder capturing microphone audio.

---

## Timed Video Script & Walkthrough

### [0:00 - 0:45] Slide 1: Introduction & Title
**[Visual: Show Slide 1 - Title Slide]**

> "Hello everyone, and welcome to my capstone project presentation for the Coursera GPU Specialization: CUDA at Scale for the Enterprise. My name is Joshua Deniese, and today I am excited to present **CUDA-Vision: An Enterprise-Scale High-Throughput GPU and Heterogeneous Vision and Signal Processing Pipeline**."
>
> "In this presentation, I will walk you through the motivation behind our pipeline, the mathematical formulation of our kernels, the GPU architectural optimizations we implemented—including shared-memory tiling, constant memory caching, and warp shuffles—as well as a live demonstration of our code running on large-scale image and signal datasets."

---

### [0:45 - 1:30] Slide 2: Project Motivation & Enterprise Context
**[Visual: Show Slide 2]**

> "In enterprise settings such as automated manufacturing inspection, medical imaging, and real-time acoustic telemetry, systems must process high-resolution visual streams and thousands of concurrent sensor signals with microsecond latency."
>
> "However, standard 2D convolution and signal processing algorithms quickly become memory-bandwidth bound when implemented naively on GPUs. For example, a standard 5 by 5 spatial filter repeatedly accesses high-latency global DRAM up to 25 times per output pixel."
>
> "The goal of this capstone project was to design an enterprise-grade pipeline that eliminates redundant memory traffic using on-chip shared memory tiles, leverages constant cache broadcasting, utilizes intra-warp register reductions, and provides full cross-platform deployability with bitwise mathematical verification."

---

### [1:30 - 2:30] Slide 3 & Slide 4: System Architecture & Algorithmic Foundations
**[Visual: Show Slide 3 & Slide 4 - Architecture Diagram]**

> "Let's look at the system architecture. CUDA-Vision is built with a modular, dual-engine design. At the top level, a unified C++17 CLI driver manages arguments, memory allocations, and configuration."
>
> "The pipeline targets two execution backends:
> First, a native NVIDIA CUDA engine targeting GPU clusters, and second, an optimized Heterogeneous CPU engine utilizing OpenMP multi-threading and an emulated CUDA grid-and-block scheduler. This allows teams to develop, profile, and verify GPU code on developer workstations without dedicated NVIDIA hardware, while retaining identical algorithm semantics."
>
> "The pipeline supports five foundational algorithms:
> 1. Coalesced Rec. 601 Grayscale conversion.
> 2. Tiled 2D Gaussian blur.
> 3. Sobel 2D gradient edge detection.
> 4. Nonlinear bilateral edge-preserving smoothing.
> 5. Batched 1D windowed-sinc FIR filtering for multi-channel sensor telemetry."

---

### [2:30 - 3:45] Slide 5 & Slide 6: CUDA Kernel Optimizations & Memory Hierarchy
**[Visual: Show Slide 5 & Slide 6 - Code Architecture]**

> "Now let's examine the GPU kernel design in `src/cuda/cuda_kernels.cu`."
>
> "To eliminate the global memory bottleneck in 2D convolution, we implemented a 2D shared-memory tiling strategy. Thread blocks are configured as 16 by 16 threads. Each block cooperatively stages an on-chip tile of size 20 by 20 elements into `__shared__` memory."
>
> "Our implementation cooperatively loads the center elements, top and bottom halos, left and right halos, and diagonal corners, clamping out-of-bounds coordinates to replicate edges without branch divergence. After a block-wide `__syncthreads()` barrier, all 25 kernel multiplications execute entirely from fast on-chip shared memory, reducing global DRAM transactions by over 85%."
>
> "In addition, uniform filter coefficients are placed in `__constant__` memory. When all 32 threads in a warp access the same filter weight, hardware constant cache broadcasts the value to all lanes in a single clock cycle with zero bank conflicts. For statistics, we use `__shfl_down_sync` warp-level register shuffles, avoiding block-level shared memory synchronization."

---

### [3:45 - 4:45] Slide 7 & Slide 8: Streams & Heterogeneous CPU Parity
**[Visual: Show Slide 7 & Slide 8]**

> "For high-volume throughput, we implemented an asynchronous multi-stream pipeline using `cudaStream_t` and pinned host memory. By overlapping host-to-device transfers of chunk N plus 1 with kernel execution of chunk N and device-to-host transfers of chunk N minus 1, we hide PCIe transfer latencies."
>
> "Crucially, for cross-platform engineering, we implemented a CPU emulation engine. The CPU engine models the exact 2D thread block grid, allocates local shared-memory tiles, executes halo loads, and verifies results. When we compare the multi-threaded reference with the tiled kernel, we achieve a Mean Squared Error of zero and a Peak Signal-to-Noise Ratio of 100 dB, proving exact mathematical equivalence."

---

### [4:45 - 6:15] Live Terminal Demonstration
**[Visual: Switch to Fullscreen Linux Terminal]**

> "Now, let's switch to the live terminal to see the pipeline in action."

**Action: Type and run:**
```bash
./run.sh
```

**[Wait for terminal execution and explain the live output as it scrolls]**

> "As you can see, our automated runner immediately starts by displaying the system configuration and OpenMP threads. In Part 1, it ingests the high-resolution Lena benchmark image—1666 by 1250 pixels, over 23 megabytes in memory."
>
> "It sequentially executes our five filter stages:
> - Grayscale conversion finishes in 14.6 milliseconds, achieving over 141 Megapixels per second.
> - Gaussian blur runs in 33.1 milliseconds at 62.8 Megapixels per second. Notice the output line: 'CUDA Tile Emulation Verification: MSE = 0.0000, PSNR = 100.00 dB'—confirming bitwise equivalence!
> - Sobel edge detection finishes in just 6.0 milliseconds, running at over 345 Megapixels per second!
> - Bilateral filtering and unsharp masking complete cleanly."
>
> "Next, in Part 2, the pipeline executes our batched 1D signal processing engine. It generates 1,000 independent telemetry channels, each with 4,096 samples—representing over 4 million time-series data points. It applies a 64-tap windowed-sinc FIR filter in just 75 milliseconds, achieving over 54 Mega-samples per second and suppressing high-frequency sensor noise by 1.78 dB."
>
> "Finally, in Part 3, the multi-scale benchmark suite evaluates resolutions from 512 by 512 up to 2048 by 2048, and automatically packages all proof-of-execution artifacts into `proof_of_execution.tar.gz`."

---

### [6:15 - 7:00] Slide 10: Performance Benchmarks & Scaling Analysis
**[Visual: Switch back to Slide 10 - Benchmark Table]**

> "Let's analyze the quantitative performance results from our benchmark run."
>
> "On a 2048 by 2048 image—over 4 million pixels—our CPU OpenMP multi-threaded Gaussian blur achieved a 2.7x speedup over single-threaded execution. Even more impressively, our tiled shared-memory emulation achieved a **5.66x speedup** (25 milliseconds vs 141 milliseconds), proving that optimizing memory locality and eliminating redundant fetches delivers enormous gains even on CPU caches."
>
> "For compute-bound filters like the Bilateral filter, OpenMP multi-threading achieved a **4.25x speedup** on our 4-core, 8-thread system, scaling from 772 milliseconds down to 181 milliseconds."

---

### [7:00 - 7:45] Slide 11 & Slide 12: Engineering Challenges, Next Steps & Conclusion
**[Visual: Show Slide 11 & Slide 12]**

> "Throughout this project, we overcame several key engineering challenges:
> First, resolving halo cell synchronization race conditions by enforcing a clean two-phase cooperative load and barrier. Second, removing bulky third-party dependencies by embedding lightweight `stb_image` single-header decoders, making the codebase 100% self-contained."
>
> "Looking ahead to next steps, I plan to extend this pipeline by integrating NVIDIA TensorRT for INT8 quantization on NVIDIA Jetson edge devices, and utilizing NCCL for distributed multi-GPU scaling across compute clusters."
>
> "In conclusion, CUDA-Vision provides a complete, enterprise-grade, high-throughput pipeline that satisfies all rubric requirements: a clean repository adhering to Google C++ Style Guide standards, concrete proof of execution across large images and 4-million-point signal streams, and thorough documentation."
>
> "Thank you very much for your time and for reviewing my capstone project!"
