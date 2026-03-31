# Baseline and Profiling: Dense GEMM FP16

This section establishes the **baseline kernel** and the **signals** that justified later changes. All numbers come from the 2026-03-23 AI-tune log unless noted; problem size is **8192³** for the headline baseline unless a smaller cube is explicitly called out.

## Baseline definition

The “main branch” reference is **208.7 TFLOPS** at **8192³** with:

- **1p1c** warp specialization (one TMA producer warpgroup, one WGMMA consumer warpgroup), as introduced in [Chapter 6: Warp specialization](../../tutorial/ch06-warpspec.md).
- **WN = 128** — warpgroup tile width along **N** (`MATMUL_WARP_N`).
- **STAGES = 4** — four shared-memory slots for the K pipeline, the same *ring of stages* idea as [Chapter 3: Pipelining](../../tutorial/ch03-pipeline.md), but with producer/consumer roles split across warpgroups.

The canonical Choreo source for this shape is the dynamic SM90 benchmark **`matmul_f16_dyn_sm90.co`** and the warpspec teaching kernel **`matmul_f16_dyn_sm90_warpspec_1p1c.co`**: operand staging is multi-buffered, the accumulator tile is not.

Operand **layout** (swizzle, `MATMUL_TILE_K`, `MATMUL_SWIZ`) is the same family as [Chapter 4](../../tutorial/ch04-tma-swizzle.md); the baseline is not “wrong” on addressing—it is **under-scheduled** relative to what the SM can accept once **roles**, **stages**, and **output staging** line up.

## TFLOPS accounting (sanity check)

For a square **GEMM** of side **S**, useful work is **2S³** multiply-adds (one FMA ≈ two flops). Reported **TFLOPS** are:

\[
\text{TFLOPS} \approx \frac{2 S^3}{t_{\text{sec}}} \times 10^{-12}.
\]

Holding **S** fixed (here **8192**) makes **TFLOPS** a direct proxy for **inverse runtime**. That is why the study can move between **2048³**, **4096³**, and **8192³**: each size changes **grid waves**, **L2 footprint**, and **tail behavior**, not the definition of a flop.

## What 208.7 TFLOPS implies

**208.7 TFLOPS** is a strong kernel by generic standards but **far below** both the **~1513 TFLOPS** theoretical FP16 tensor peak and the **~380 TFLOPS** that **cuBLAS** achieves on the same machine. The gap is not mysterious: GEMM is **memory-latency-, occupancy-, and pipeline-depth-sensitive** on Hopper. When any of those dimensions is mis-tuned, math throughput collapses even though WGMMA is “on.”

Expressed as a fraction of the practical ceiling (**~380 TFLOPS**), the baseline is roughly **55% of cuBLAS**. That single ratio is enough to justify a structured search: the kernel is not compute-bound in the naive sense; it is **schedule-bound**.

## Profiling narrative (conceptual)

We did not rely on a single counter. The optimization thread combined:

1. **Throughput vs. problem size** — run the same kernel at **2048³**, **4096³**, and **8192³**. A kernel that looks good on a small cube can be **SMEM- or wave-quantization-limited** on the largest.
2. **Occupancy vs. shared memory** — each extra **pipeline stage** multiplies operand staging in shared memory. Hopper exposes a **228 KB** shared memory budget per SM (typical configuration); crossing it changes **CTAs per SM** discontinuously.
3. **Role balance in 1p1c** — if the consumer sits on **`wait full`** while the producer is not far enough ahead, or the producer stalls on **`wait empty`**, the pipeline is **bubble-limited**. Warp specialization ([Ch6](../../tutorial/ch06-warpspec.md)) fixes *who* runs TMA vs. WGMMA, not *whether* the stages are sized correctly.

Early AI-tune phases at **2048³** (Phase 1) isolate **tile and stage** effects without the full **grid wave** story. That is deliberate: a smaller cube stresses **inner-loop efficiency** first.

## Waves, tails, and why size matters

With a **data-parallel** grid (one CTA per output tile), the number of blocks is roughly:

\[
\Big\lceil \frac{M}{\text{WM}} \Big\rceil \times \Big\lceil \frac{N}{\text{WN}} \Big\rceil,
\]

where **WM/WN** are warpgroup tile extents. For large **M,N**, this count is huge; the hardware schedules it in **waves** of width **O(number of SMs)**. If **per-block** runtime is uneven—or if **occupancy** drops so fewer CTAs run concurrently—the **last wave** leaves SMs idle. Persistent kernels ([Ch7](../../tutorial/ch07-persistent.md)) attack that **grid-level** tail. This baseline used a **conventional** grid; part of the later story is that **non-persistent** still won at **8192³** once **inner-block** throughput dominated.

## Phase 1 snapshot at 2048³ (1p1c)

At **2048³**, the documented 1p1c baseline was **204 TFLOPS** (WN=128, STAGES=4)—essentially the same family as the 8192³ baseline, scaled to a friendlier development size.

Two moves immediately illustrate **SMEM footprint vs. occupancy**:

| Iteration | Change | TFLOPS (2048³) | Reading |
|-----------|--------|----------------|---------|
| iter004 | WN=256, STAGES=2 | 208.9 | Fewer stages shrink the operand ring; slightly better residency / scheduling. |
| iter023 | +`ptx-barrier`, +`stmatrix`, +subspan | 214.3 | Compiler/hardware path improvements; **~+5%** over the Phase-1 baseline. |

The takeaway for readers of [Ch3](../../tutorial/ch03-pipeline.md): **more stages** are not automatically better—each stage is real bytes in **`lhs_load_s` / `rhs_load_s`**, and those bytes compete with **how many blocks** can live on an SM.

## Phase 2 preview: when numbers jump

Phase 2 (iter043–057) moved from **1p1c** refinements into **1p2c split-output** and **multi-size** validation. The important **profiling** transition is qualitative: once **TFLOPS** at **2048³** crossed **350+** (iter048, **354.1**), the kernel was no longer “missing WGMMA utilization” in a generic sense—it was **ready** to be stressed at **4096³** and **8192³**, where **output staging** and **occupancy** dominate.

| Step | TFLOPS | Size | Note |
|------|--------|------|------|
| iter046 | 242 | 2048³ | WN=176, STAGES=2 |
| iter048 | 354.1 | 2048³ | Same WN, STAGES=3 |
| iter050 | ~375 | 4096³ | 1p2c split-output |
| iter057 | 382.5 | 8192³ | Best headline |

The jump **242 → 354.1** on the **same** **WN** by changing **STAGES alone** is the clearest **profile-driven** lesson in the log: **pipeline depth** must be tuned **after** tile width settles, and **again** when the problem size changes.

## Baseline vs. best (8192³)

| Version | TFLOPS | Notes |
|---------|--------|-------|
| Baseline | 208.7 | 1p1c, WN=128, STAGES=4 |
| Best (iter057) | **382.5** | 1p2c split-output, WN=152, **non-persistent** |

The **+83%** delta is entirely “software schedule”: same ISA family, same broad algorithm (TMA + WGMMA + swizzled staging), different **tile geometry**, **pipeline depth**, **output staging model**, and **launch persistence** ([Ch7](../../tutorial/ch07-persistent.md)).

## Checklist: what we looked at before changing code

1. **End-to-end TFLOPS** at **2048³**, **4096³**, **8192³** with the same harness defaults (`CHOREO_TIMING_WARMUP`, `CHOREO_TIMING_REPEAT`).
2. **Shared memory per block** vs. **228 KB** SM limit—explicit **table** of **`STAGES × tile bytes`** plus **output** (including **split** slices when 1p2c).
3. **CTAs/SM** implied by that footprint—watch for **1 vs 2** transitions.
4. **Role balance** in 1p1c: are stalls visible as **bubbles** (producer not far enough ahead) or as **empty slots** (consumer starved)?
5. **Compiler flags** held **fixed** across comparisons when testing **tileflow**; only after a win was stable did we layer **`stmatrix` / `ptx-barrier` / cluster-aware TMA**.

## Baseline kernel vs. teaching kernel

Students often read **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** in [Ch6](../../tutorial/ch06-warpspec.md) and assume it is “the fastest” kernel in the repo. In this study it is the **clearest** expression of **1p1c**—not the **throughput** champion. The **dynamic** benchmark **`matmul_f16_dyn_sm90.co`** matches the **208.7 TFLOPS** baseline configuration on **main** and is the correct **apples-to-apples** reference when comparing against **AI-tune** artifacts.

If your **measured** TFLOPS on **H800** disagree with **208.7**, check:

- **Clocks** (boost vs. sustained)
- **Verification** path still disabled for timing (should be **off** only when explicitly debugging)
- **Problem size** actually **8192³** (off-by-one in **M/N/K** changes waves dramatically)

## Connecting bubbles to hardware counters (optional)

When **NSight Compute** is available, correlate **warp issue** stalls on the **consumer** with **`wait full`** sites, and **TMA** stalls on the **producer** with **`wait empty`**. This case study does not paste counter names because the **public** evidence is **TFLOPS**—but the **mental model** from [Ch6](../../tutorial/ch06-warpspec.md) maps cleanly: **full/empty** are not syntax sugar, they are the **schedule** you are optimizing.

## Why start at 2048³ in Phase 1

A **2048³** cube cuts **K** iterations by **64×** versus **8192³** for the same **`MATMUL_TILE_K`**, which shortens **edit-compile-measure** cycles. Phase 1’s **204 → 214.3** band is numerically smaller than Phase 2’s jumps, but it **de-risked** compiler flags and **SMEM** direction before paying the **long** runtimes of **8192³** on every iteration.

## Correctness as a profiling gate

Every row in the README assumes **bitwise** or **tolerance-checked** agreement with a reference on the same **M,N,K**. When **TFLOPS** jump **+50%** in one iteration, the **first** question is not “is Hopper happy?” but “did we **silently** change the **math**?” The harness’s **`CHOREO_SKIP_VERIFY`** flag exists for **inner-loop** timing experiments, but the **published** numbers were taken with **verification on** unless explicitly noted otherwise.

If you replicate this study, treat **correctness** failures as **hard stops**: a **fast wrong** kernel is **negative** progress because it corrupts the **search** direction.

## Baseline constants in plain language

Matching the **208.7 TFLOPS** baseline means matching its **tile vocabulary**, not just “warp spec on”:

- **`MATMUL_WARP_M` / `MATMUL_WARP_N`** — the **M×N** **WGMMA** tile owned by the **consumer** warpgroup before **`mma.store`** into **`output_s`**.
- **`MATMUL_TILE_K`** — thickness of each **K** slab staged through **shared** memory per **stage**.
- **`MATMUL_STAGES`** — how many **K** slabs are **in flight** as a **ring** for **operands** only.
- **`MATMUL_SWIZ`** — **swizzle** metadata tying **TMA** and **WGMMA** loads together ([Ch4](../../tutorial/ch04-tma-swizzle.md)).

When any constant changes, **recompute** **SMEM** before interpreting **TFLOPS**: a **+8** tweak to **WN** is harmless on paper and **catastrophic** in **KB**.

## Interpreting “close to cuBLAS” at baseline

**208.7 TFLOPS** vs. **~380 TFLOPS** **cuBLAS** is **not** “within tuning noise.” It is a **different operating point**: **library** kernels routinely combine **software pipelining**, **workspace**, **autotuning**, and **microkernels** per **arch**. The Choreo baseline is **intentionally readable**—its job is to teach **tileflow**. The AI-tune branch shows how far **the same vocabulary** can be pushed when **readability** trades off against **schedule**.

## One-sentence takeaway

The baseline was **already** a **Hopper-style** **TMA+WGMMA** **warp-specialized** pipeline; it lost **~45%** of **cuBLAS** throughput because **tile width**, **stage count**, **output staging**, and **launch** mode did not match **8192³** **occupancy** and **contention** realities—**not** because the kernel was “unoptimized” in the textbook sense.

## Summary

Profiling here is **TFLOPS plus resource arithmetic**: shared memory per block dictates occupancy; occupancy and pipeline depth dictate how often WGMMA hides TMA. The baseline was already warp-specialized and multi-staged—it was **not** naive—but it sat at **208.7 TFLOPS** at **8192³** because those choices were **suboptimal for the largest problem**. The next document walks the patterns that moved the needle into **cuBLAS territory**.

---

Next: [Optimization patterns](pattern-optimizations.md).
