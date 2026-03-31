# Optimization Patterns: Block-Scaled GEMM FP8

This document walks the **patterns** that moved **blockscale GEMM E4M3** from the **baseline** (`blockscale_gemm_dyn_sm90.co`) to the **best shipped kernel** (**iter066**), using the **2026-03-22** AI-tune log as the source of truth. All kernels target **SM90a**; headline **peak** remains **3026 TFLOPS** on **H800 PCIe**.

## Results ladder (2048³ and 4096³)

| Iter | TFLOPS @2048³ | TFLOPS @4096³ | Δ vs baseline @4k | Primary lever |
|------|---------------|---------------|-------------------|---------------|
| baseline | 314.2 | 397.9 | — | M64N128K32 reference |
| iter049 | **380** | — | — | **TMA overlap** with **scale accumulation** |
| iter051 | 372 | 602 | +51% | **N256 WGMMA** (M64N256K32) |
| iter053 | — | 610 | +53% | **N256** + **L2 256B promotion** on **RHS TMA** |
| **iter066** | — | **621** | **+56%** | **N256** + **L2** + **prefetch `scale_a`** (before WGMMA loop) |

**Best @2048³**: **iter049** (**+21%** vs baseline). **N256** shifts can **trade small-cube grid size** for **large-cube throughput**, so **iter051** is slightly **below** baseline at **2048³** while winning at **4096³**.

## Pattern 1: TMA overlap after WGMMA (iter049)

**Symptom**: The consumer waits on **WGMMA**, then performs **scale-related accumulation** work, while the **next** **K-tile** operands are not yet **in flight**—**TMA** sits **idle** at the handoff.

**Change**: **Issue the next K-block’s TMA loads** as soon as the **WGMMA wait** completes, **before** or **in parallel with** **scale_accumulator** logic, so **memory latency** overlaps **non-WGMMA** math.

**Outcome**: **380 TFLOPS** at **2048³** (**+21%** over **314.2**), without changing the **M64N128K32** tile class.

**Reading**: This is a **software pipeline** refinement in the spirit of [Chapter 3: Pipelining](../../tutorial/ch03-pipeline.md): reorder **independent** work so the **longest-latency** piece (**TMA**) starts **earlier**. Blockscale adds **scale_accumulator** as a **third** phase beside **load** and **MMA**; **iter049** shows that phase can **share** the bubble with **TMA**.

## Pattern 2: N256 WGMMA — double math per tile (iter051)

**Symptom**: **N128** tiles **finish** **K-pipeline** steps quickly but **launch** **many** CTAs along **N**; on **large** **N**, **wave quantization** and **per-CTA overhead** dominate.

**Change**: Move to **M64N256K32** — **double** the **N** dimension of the **WGMMA** tile per CTA. README notes **~40 KB** **shared memory** for the operand staging footprint.

**Outcome**: **602 TFLOPS** at **4096³** (**iter051**). **2048³** drops to **372 TFLOPS** (vs **380** on **iter049**) because **fewer** blocks cover the **N** dimension; the **grid** is **coarser** and **occupancy**/**parallelism** trade differently.

**Reading**: **Wider N** is the same **GEMM knob** as in dense FP16 tuning: **more math per block**, **fewer blocks**, **heavier SMEM**. For **blockscale**, the **RHS** and **scale_rhs** **footprint** scales with **N**; **SMEM** limits must still allow **multi-stage TMA** if the pipeline uses them.

## Pattern 3: L2 promotion on RHS TMA (iter053)

**Symptom**: At **4096³**, **RHS** panels are **large**; **TMA** traffic is **repeatedly** pulled through the **memory hierarchy** without **sticking** in **L2** as effectively as desired.

**Change**: Set **`CU_TENSOR_MAP_L2_PROMOTION_L2_256B`** on the **RHS** **tensor map** so hardware **promotes** lines into **L2** with a **256B** granularity policy (NVIDIA **Hopper** TMA / **tensor map** option).

**Outcome**: **610 TFLOPS** at **4096³** (**+8** over **iter051**).

**Reading**: This is a **hint**, not a new algorithm: it **biases** the **cache** toward **reuse** of **RHS** data across **K** iterations and **CTAs** that share **spatial locality**. It pairs naturally with **wider N** (**iter051**), which **increases** **per-CTA** **RHS** volume.

## Pattern 4: Prefetch per-row `scale_a` before WGMMA (iter066)

**Symptom**: **Scale** loads (**`__ldg`** or equivalent) can **stall** the **consumer** if they sit **inside** the **tight** **WGMMA** loop with **short** **II**.

**Change**: **Prefetch** **per-row** **`scale_a`** values into **registers** **before** the **inner WGMMA** loop body so **load latency** hides **behind** **independent** setup or **prior** **WGMMA** work.

**Outcome**: **621 TFLOPS** at **4096³** (**best**); **+56.1%** vs **baseline 397.9**.

**Reading**: Blockscale makes **scales** **first-class operands**. Treat them like **any other** **latency-bound** input: **software prefetch**, **double-buffering**, or **DMA-to-SMEM** (below) are all **valid** **design** axes.

## Choreo source variants (exploration, not the iter table)

The **`blockscale_gemm_v2/`** directory holds **`.co`** experiments that **factor** the **scale path** differently from the **register-immediate** style in **`blockscale_gemm_dyn_sm90.co`**:

| File (suffix / name) | Idea |
|----------------------|------|
| **`rhs_scale_dma_smem`** | Bring **RHS-related scales** via **TMA into shared memory** instead of **register** paths only. |
| **`scale_dma_smem`** | General **scale DMA to SMEM** staging. |
| **`transposed_scale`** | Change **scale layout** for **coalescing** / **TMA** vs **index cost**. |
| **`tileN`** | **Tile** along **N** explicitly in the **Choreo** structure (compare to **fixed** **WARP_N** macros). |

These align with the **README** theme: **scale DMA** is an alternative to **keeping scales in registers** when **register pressure** or **load scheduling** hurts **WGMMA** **II**.

**Warp-specialized** baselines:

- `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co` — **1p1c** template.
- `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c_m2048_n2048_k2048.co` — adds **`__cpp__`** / launch tuning (**`setmaxnreg`**-class constraints) for **specific** **M,N,K**.

**Persistent** variant: `blockscale_gemm_e4m3_dyn_sm90_warpspec_persis_1p1c.co` ties into [Chapter 7: Persistent kernels](../../tutorial/ch07-persistent.md) for **grid** behavior across **tiles**.

## Environment variables (reproducibility)

When running generated harnesses or **`bash /tmp/bs.cute.result`** wrappers:

| Variable | Role |
|----------|------|
| `CHOREO_TIMING_WARMUP` | Warmup iterations (default **100**) |
| `CHOREO_TIMING_REPEAT` | Timed iterations (default **1000**) |
| `CHOREO_DISABLE_TIMING` | Set **`1`** to skip timing |
| `CHOREO_SKIP_VERIFY` | Set **`1`** to skip numerical check |

Use **`--disable-timing`** on **`run.sh`** when you only need **correctness** or **compile** validation.

## Compile flags (Cute + warp specialization)

The tutorial index gives a **representative** **choreo** command:

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  -o /tmp/bs.cute.result && bash /tmp/bs.cute.result --execute
```

- **`-arch=sm_90a`** selects **Hopper** features (**WGMMA**, **TMA**).
- **`--use-warpspec`** matches **1p1c** (and related) **producer/consumer** lowering.
- **`--stmatrix`** enables **store-matrix** paths for **accumulator** writeback where the compiler pipeline expects it.

Shipped **iter049 / iter051 / iter053 / iter066** artifacts are **pre-generated `.cu`** folders with **`run.sh`**; use those for **bit-identical** reproduction of the **README** numbers.

## Takeaways

1. **Blockscale** adds a **scale critical path**; **iter049** proves **scheduling** (**TMA** vs **scale_accumulator**) matters as much as **tile size**.
2. **N256** (**iter051**) is the **large-cube** win: **more FLOP per CTA**, **cost** at **small** cubes.
3. **L2 promotion** (**iter053**) and **scale prefetch** (**iter066**) are **late** **percentage** gains on an already **strong** kernel—exactly where **memory hierarchy** and **operand latency** dominate.
4. **Choreo `.co` files** in **`blockscale_gemm/`** and **`blockscale_gemm_v2/`** document **alternative scale movement** (**DMA SMEM**, **transposed**) for **future** tuning when **register** or **layout** limits bind.

Full iteration history: **`ai-tune/2026-03-22/blockscale_gemm_v2`** (71 iterations). Summary: `choreo/benchmark/performance/blockscale_gemm_v2/README_blockscale_gemm_e4m3_aitune_2026-03-22.md`.
