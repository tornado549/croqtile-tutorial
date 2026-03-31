# Sparse GEMM: FP16 and FP8 E4M3 Case Study

This case study follows **structured 2:4 sparse matrix multiply** on Hopper (SM90a), measured on **H800 PCIe** (114 SMs). We treat **FP16** and **FP8 E4M3** (accumulate to FP16) as one narrative: same sparsity pattern, same metadata-driven loads, but different tensor paths and ceilings—so optimization patterns **transfer**, while the dominant bottleneck shifts.

**Problem shape (both dtypes)**

| Field | Value |
|-------|--------|
| M × N × K | 4096 × 8192 × 8192 |
| Sparsity | NVIDIA 2:4 (four consecutive weights, two kept) |
| Peak reference (FP8) | **3026 TFLOPS** (H800 PCIe headline) |
| Peak reference (FP16 dense) | **1513 TFLOPS** (context for accumulators) |

**End-to-end results**

| Variant | Baseline TFLOPS | Best TFLOPS | Gain vs baseline |
|---------|-----------------|-------------|------------------|
| **FP16** sparse GEMM | 368 | **655** (iter143) | **+78%** |
| **E4M3** sparse GEMM | 671 | **1127** (iter068) | **+68%** |

The FP16 line peaks lower in absolute TFLOPS but shows how **metadata bandwidth**, **TMA layout**, and **pipeline depth** compound; E4M3 starts from a stronger baseline and benefits especially from **software pipelining** and **early-empty / barrier** refinements.

**How to read the series**

1. [Baseline and sparse GEMM background](baseline-analysis.md) — 2:4 structure, why metadata matters, and what the Choreo baselines are doing.
2. [Optimization patterns](pattern-optimizations.md) — patterns that apply to both F16 and E4M3, with dtype-specific notes.
3. [AI-tune last mile](aitune-last-mile.md) — `.co` vs `.cu`, iter120 vs iter143 on FP16, and how automated sweeps explore the space.

Sources of truth for iteration tables: `choreo/benchmark/performance/gemm_sp/README_gemm_sp_f16_aitune_2026-03-25.md` and `README_e4m3_aitune_2026-03-21.md`. Representative artifacts live under `benchmark/performance/gemm_sp/` (`.co` variants and per-iteration `.cu` subfolders).

**Methodology (shared with other optimization chapters)**

Measurements use the same harness described in [Setting up: TimerOption, TFLOPS, and HW efficiency](../setup-profiling.md): warmup, repeated timed runs, and TFLOPS derived from the effective operation count for **2:4** (nonzeros only). Compare sparse results against **FP8** or **FP16** peaks only to sanity-check scale—not to claim “percent of dense GEMM.”

**Tutorial cross-links**

Warp specialization ([Ch6](../../tutorial/ch06-warpspec.md)), pipelining ([Ch3](../../tutorial/ch03-pipeline.md)), TMA swizzle ([Ch4](../../tutorial/ch04-tma-swizzle.md)), and MMA ([Ch5](../../tutorial/ch05-mma.md)) supply the vocabulary; this case study shows how they compose when a **metadata plane** sits beside the operand plane.

**Why two dtypes in one study**

FP16 makes **metadata and layout** mistakes expensive in wall time because the math ceiling is lower. E4M3 raises the math ceiling so **sync and pipeline bubbles** surface sooner. Seeing both curves helps you decide whether the next experiment should target **loads**, **stages**, or **barriers**.
