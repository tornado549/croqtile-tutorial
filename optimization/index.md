# Performance Tuning Demos

This section presents real-world kernel optimization case studies. Each case starts from a working baseline, profiles it to identify bottlenecks, and applies optimization patterns to push performance toward hardware peak.

## Getting Started

Before diving into individual cases, read the profiling setup section — it covers the tooling and methodology used throughout:

- [Setting Up: TimerOption, TFLOPS, and HW Efficiency](setup-profiling.md)

## Case Studies

### [Dense GEMM FP16](matmul-f16/index.md)

Optimizing a half-precision matrix multiply from ~208 TFLOPS to 382+ TFLOPS on H800 PCIe. Covers warp specialization, multi-stage pipelining, split-output, persistent kernels, and tile scheduling strategies.

### [Sparse GEMM: FP16 and FP8 E4M3](gemm-sp/index.md)

Structured 2:4 sparse GEMM on **4096×8192×8192**: FP16 from **368** to **655 TFLOPS** (iter143), FP8 E4M3 from **671** to **1127 TFLOPS** (iter068). Covers metadata/TMA staging, warp specialization, 3-stage pipelines, barrier tuning, and the `.co`-to-`.cu` boundary.

### [Block-Scaled GEMM FP8](blockscale-gemm/index.md)

FP8 GEMM with per-block scaling factors. Covers scale DMA staging, warp-specialized blockscale pipelines, and transposed scale patterns.
