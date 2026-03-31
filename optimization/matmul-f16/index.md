# Dense GEMM FP16: Optimization Case Study

This case study walks a Hopper (SM90a) half-precision matrix multiply from the Choreo benchmark baseline to near–cuBLAS-class throughput on **H800 PCIe** (114 SMs). The narrative is intentionally linear: establish a measured baseline, read the hardware story from TFLOPS and occupancy, then apply patterns from the Lv0 tutorial—[warp specialization](../../tutorial/ch06-warpspec.md), [multi-stage pipelining](../../tutorial/ch03-pipeline.md), and [persistent tiling](../../tutorial/ch07-persistent.md)—only when profiling supports them.

**Numbers that anchor the story**

| Milestone | TFLOPS | Setting |
|-----------|--------|---------|
| Baseline (main) | 208.7 | 8192³, 1 producer / 1 consumer (1p1c), WN=128, 4 stages |
| Best shipped | **382.5** | 8192³, 1p2c split-output, WN=152, non-persistent |
| Improvement | **+83%** | Same problem size vs. baseline |

Hardware context: **1513 TFLOPS** is a common FP16 tensor peak headline for this class of GPU; **cuBLAS** on this stack lands near **~380 TFLOPS**, which is the practical ceiling the hand-tuned kernels chase.

**How to read the series**

1. [Baseline and profiling](baseline-analysis.md) — what the default `matmul_f16_dyn_sm90.co` and early warpspec kernels are doing wrong at 8192³.
2. [Optimization patterns](pattern-optimizations.md) — tile width (WN), pipeline depth (stages), split-output 1p2c, compiler flags (`stmatrix`, `ptx-barrier`, TMA cluster awareness), and when each pattern wins.
3. [AI-tune last mile](aitune-last-mile.md) — sweep methodology at 8192³, the WN=168 occupancy cliff, and shipped kernels (iter048, iter050, iter057, iter061).

Source of truth for raw iteration tables: `choreo/benchmark/performance/matmul/README_matmul_f16_aitune_2026-03-23.md`.

Representative `.co` sources in the Choreo tree include `benchmark/performance/matmul/matmul_f16_dyn_sm90.co` (dynamic-tile baseline), `matmul_f16_dyn_sm90_warpspec_1p1c.co`, `matmul_f16_dyn_sm90_warpspec_1p2c.co`, and the dated AI-tune variants (`*_iter048_*`, `*_iter050_*`, `*_iter057_*`, `*_iter061_*`).

**Prerequisites**

Skim [TMA and swizzle](../../tutorial/ch04-tma-swizzle.md) and [WGMMA](../../tutorial/ch05-mma.md) before Ch6—the case study assumes you already know **why** operand layouts match between **`tma.copy`** and **`mma.load.swiz`**.

**Method**

Each chapter follows the same loop: quote **TFLOPS** at a fixed problem size, name the **limiter** (occupancy, pipeline bubbles, output contention), apply **one** pattern, re-measure. No pattern is adopted on intuition alone.
