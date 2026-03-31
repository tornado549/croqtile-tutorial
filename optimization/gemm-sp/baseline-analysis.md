# Baseline Analysis: Structured 2:4 Sparse GEMM

## What “2:4” means in this benchmark

**Structured 2:4 sparsity** is a fixed pattern: for every group of four values along the **sparse dimension** (here, along K in the weight-like operand), exactly two are retained and two are zero. The hardware encodes which two via **metadata** so the MMA path can fetch **packed** nonzeros without treating the matrix as dense.

Compared to unstructured sparsity, 2:4 is **predictable**: tile sizes, swizzle modes, and TMA transfers can be scheduled knowing the compression ratio is **2×** along K for the sparse side. The cost is **metadata traffic** and **instruction overhead**: every MMA slice must be paired with the right metadata words, and those loads can become a first-class bottleneck once the math path is well fed.

## Operand roles and dtypes

For the benchmarks summarized here:

- **FP16 sparse GEMM**: operands and accumulation stay in **FP16**; the story is classic Hopper **WGMMA** throughput versus memory and metadata.
- **E4M3 sparse GEMM**: **FP8 E4M3** inputs with **FP16** accumulation—higher nominal tensor throughput (peak **3026 TFLOPS** FP8 on H800 PCIe) but the same 2:4 layout and metadata semantics.

In both cases, profiling should separate **(a)** tensor math utilization, **(b)** TMA / shared staging, **(c)** metadata path (latency, bank conflicts, cache behavior), and **(d)** synchronization between producer and consumer warpgroups.

## FP16 baseline: 368 TFLOPS

The FP16 baseline is intentionally conservative but already uses important Hopper ingredients:

| Attribute | Baseline choice |
|-----------|-----------------|
| Warp specialization | **1 producer / 1 consumer** (1p1c) |
| Swizzle | **64** (lhs packing / shared layout) |
| Tile K (TK) | **64** |
| Pipeline | **2-stage** |

At **368 TFLOPS** on **4096 × 8192 × 8192**, the kernel is not “wrong”—it is **under-pipelined** and **metadata-conservative** relative to what the SM can sustain when stages, warp mix, and vectorized metadata loads align.

**How to read the number:** compare against dense FP16 peak (**1513 TFLOPS**) only as order-of-magnitude context; sparse kernels are not expected to hit dense peak because effective FLOPs per stored element differ and metadata is extra work. The actionable question is whether time goes to **MMA**, **TMA**, or **metadata + barriers**.

## E4M3 baseline: 671 TFLOPS

The E4M3 baseline is already much closer to hardware spirit:

| Attribute | Baseline choice |
|-----------|-----------------|
| Warp specialization | **1p1c** |
| Swizzle | **128 / 128** (lhs and rhs TMA swizzle) |
| Layout | **Prepacked** sparse operand |
| Pipeline | **2-stage** |

**671 TFLOPS** reflects a kernel that has absorbed lessons from dense FP8 work: good swizzle pairing, serious TMA setup, and a metadata path that is not yet fully software-pipelined or warp-mixed for hide latency.

Efficiency vs **3026 TFLOPS** FP8 peak is roughly **22%** at baseline—reasonable for a first structured-sparse implementation before **deep staging** and **producer–consumer overlap** are pushed.

## Profiling lens: where time goes first

Across both dtypes, early profiles tend to show one or more of:

1. **Consumer-bound WGMMA** with **idle cycles waiting on data or metadata** — often fixed by **more stages**, **better prefetch**, or **splitting producer work** (1p2c).
2. **Metadata loads not hiding under TMA** — shows up as front-end or memory latency on small, scattered accesses; mitigations include **vectorized loads (`uint2`)**, **`__ldg`**, **hoisting**, and **TMA-side metadata staging**.
3. **Barrier / empty-full latency** — producer signals and consumer waits can dominate when the pipeline is short; **early empty**, **merged barriers**, and **`warpgroup_wait<1>`**-style fine-grained sync reduce bubble time.

The FP16 baseline sits lower partly because **TK64** and **2-stage** depth leave less slack to hide metadata; the E4M3 baseline already exploits **wider swizzle** and **prepack**, so the next wins come from **micro-architectural overlap** rather than “turn on TMA.”

## Baseline comparison table

| Metric | FP16 baseline | E4M3 baseline |
|--------|---------------|---------------|
| TFLOPS | 368 | 671 |
| 1p1c | yes | yes |
| Stages | 2 | 2 |
| Notable layout | swizzle **64**, TK **64** | swizzle **128/128**, prepack |

## Takeaway for the next sections

Sparse GEMM optimization is not a single knob—it is **simultaneous** scheduling of **packed values**, **metadata**, and **accumulator lifetimes**. The following document walks patterns in the order they tended to appear in measured iterations: tighten **warpgroup timing**, improve **metadata delivery**, then **scale pipeline depth** and **warp specialization** until returns diminish or you hit the **`.co` compiler ceiling** and need **hand-authored `.cu`** for the last jumps (covered in [AI-tune last mile](aitune-last-mile.md)).

## Effective work and FLOP accounting (2:4)

For a dense GEMM of size \(M \times N \times K\), textbooks use \(2MNK\) multiply-adds. With **2:4** along the sparse axis, each logical K step touches **half** as many stored nonzero weights in that operand, but the hardware still performs **dense-rate MMA** on **packed** fragments—the metadata selects active lanes. Benchmarks in this tree report **TFLOPS** consistent with the harness definition in [setup-profiling](../setup-profiling.md): use the same formula the benchmark prints so **before/after** iterations stay comparable.

When you see **655 TFLOPS** (FP16) vs **1127 TFLOPS** (E4M3), part of the gap is **wider tensor throughput** for FP8, and part is **kernel schedule**: the E4M3 line reached **aggressive 3-stage** and **refined barriers** earlier in the measured sweep.

## Metadata path: why it shows up in profiles

Metadata is not “free sideband data.” It is on the **critical path** from global memory (or a staging buffer) to the sparse MMA interface. Symptoms of a hot metadata path include:

- **Low IPC** on consumer warps despite high `wgmma` issue rate—consumers stall between fragments.
- **Extra L1/L2 traffic** from scalar or poorly coalesced loads relative to operand TMA.
- **Serial dependence** where metadata is read **inside** the K loop without prefetch into registers or shared.

The FP16 AI-tune chain explicitly attacks this: `__ldg`, **`uint2` vectorization**, **hoisted loads**, **L2-promoted** patterns, and **TMA metadata** staging. The E4M3 chain does the same at higher baseline TFLOPS, with **TMA metadata staging** appearing as early as **iter001 (759 TFLOPS)**.

## Milestone table: FP16 (selected iterations)

These anchor points come from `README_gemm_sp_f16_aitune_2026-03-25.md`:

| Stage | Label | TFLOPS | Notes |
|-------|-------|--------|-------|
| Start | Baseline | **368** | 1p1c, swizzle64, TK64, 2-stage |
| `.co` high point | iter120 | **434** | 1p2c + 3-stage |
| `.cu` (organic) | iter137 | **543** | unroll 24 + FTZ |
| Best | iter143 | **655** | TK128, TMA metadata, split RHS TMA |

The jump from **434** to **655** is the **`.co` vs `.cu` story** in miniature: compiler-generated schedules plateau; hand scheduling exposes more ILP and finer TMA/MMA overlap (detailed in [aitune-last-mile](aitune-last-mile.md)).

## Milestone table: E4M3 (selected iterations)

From `README_e4m3_aitune_2026-03-21.md`:

| Iteration | TFLOPS | Change theme |
|-----------|--------|--------------|
| Baseline | **671** | 1p1c, swizzle 128/128, prepack, 2-stage |
| iter001 | **759** | TMA metadata staging |
| iter016 | **772** | early empty + merged barrier |
| iter023 | **811** | software pipeline + `warpgroup_wait<1>` |
| iter036 | **897** | 1p2c warp specialization |
| iter040 | **1090** | **3-stage pipeline** (large step; ~+62% vs baseline in headline docs) |
| iter068 | **1127** | early empty **arrive** (best; ~+68% vs baseline) |

The **iter040** discontinuity is worth emphasizing in profiles: before it, improvements are incremental tens of TFLOPS; after **3-stage** lands correctly, the kernel crosses into **>1000 TFLOPS** territory—classic **pipeline-depth** leverage once operands **and** metadata prefetch far enough ahead.

## Shared bottleneck themes (cross-dtype)

| Theme | FP16 evidence | E4M3 evidence |
|-------|---------------|--------------|
| Fine-grained warpgroup sync | `warpgroup_wait<1>` early in F16 chain (+4% class) | iter023 at **811** combines SW pipe + `warpgroup_wait<1>` |
| Metadata delivery | `__ldg`, `uint2`, hoisting, TMA meta | iter001 **TMA metadata staging** |
| Producer/consumer mix | 1p2c + 3-stage (iter120, iter143 path) | iter036 **897** for 1p2c; iter040 for 3-stage |
| Barrier / empty-full | (implicit in later F16 stages) | iter016, iter068 **early empty** / **arrive** |

## When to deepen the pipeline vs widen tiles

**More stages** hide **TMA latency** and **metadata latency** but increase **shared memory** and **register pressure**. If occupancy collapses, extra stages can **hurt**. The measured **E4M3** result that **3-stage** (iter040) bought **~+62%** says that, for this shape, the SM had enough **air cover** to absorb another stage—likely because **metadata staging** and **warp spec** had already reduced bubble time.

**Wider TK** (e.g. FP16 **TK128** in iter143) increases **per-iteration** work and can amortize **launch and epilogue** overhead, but requires **consistent swizzle/TMA** and sometimes **split RHS TMA** to keep bandwidth balanced.

## Summary

Baselines are **not naive kernels**—they already use Hopper features—but they leave **latency holes** on the metadata and synchronization path. FP16 at **368 TFLOPS** and E4M3 at **671 TFLOPS** are the **before** pictures; the **after** pictures (**655** and **1127**) come from applying the same **catalog** of patterns with different **order** and **ceiling**. The next chapter is that catalog, tied to measured deltas.

## Instrumentation reminders (Hopper)

When you replicate this analysis locally, pair **TFLOPS** with at least one **counter-backed** view:

- **Warp issue stall reasons** — barrier and **not enough eligible warps** often track **pipeline bubbles**.
- **Tensor Core activity** vs **DRAM throughput** — if TC is high but TFLOPS is low, suspect **short K** or **metadata** overhead distorting useful work per cycle.
- **L2 hit rate** on **metadata buffers** — improvements from **`uint2`**, **128B** alignment, and **hoisting** should **raise** hits or **lower** transactions per tile.

Nsight Compute section names vary by version; the **interpretation** above stays stable.

## Shape-specific notes (4096 × 8192 × 8192)

This **M × N × K** is **wide** along **N** and **K** with **moderate** M. Wave quantization on **114 SMs** can make **grid** launch efficiency **sensitive** to **tile M/N** choices. When comparing iterations, ensure **block tiling** did not change **CTA count** in ways that **fake** a TFLOPS win (fewer blocks can **inflate** per-SM work while **hurting** tail waves).

If an iteration changes **only** inner-loop sync, **grid** should be unchanged—those are the **cleanest** A/B tests.

## Relationship to dense GEMM case study

The [dense FP16 matmul](../matmul-f16/index.md) case tops out near **cuBLAS-class ~380 TFLOPS** at **8192³**. Sparse FP16 here reaches **655 TFLOPS** at **4096×8192×8192**—**not** directly comparable problem sizes or FLOP definitions, but the **same engineering habits** apply: **stage depth**, **1p2c**, **swizzle**, and **TMA** first; **micro** last.

## Glossary (quick)

| Term | Meaning here |
|------|----------------|
| **1p1c / 1p2c** | One TMA producer warpgroup; one vs two consumer warpgroups (warp specialization). |
| **TK** | Tile size along **K** for the inner steady-state loop. |
| **2-stage / 3-stage** | Number of **buffered** operand slots in the async pipeline. |
| **Prepack** | Sparse operand already in **hardware-packed** layout for 2:4. |
| **TMA metadata** | Using **TMA** (or TMA-like paths) to **fetch** metadata, not only scalars. |

## Files to diff when learning

Under `benchmark/performance/gemm_sp/` (Choreo tree), diff **baseline** `.co` against **iter120-class** `.co` to see **compiler-level** 1p2c+stage changes; diff **iter120** against **iter143 `.cu`** to see **manual** TMA/metadata/TK moves. For E4M3, trace **iter036 → iter040 → iter068** for **sync** and **depth** evolution.

## Noise and repeat counts

Sparse GEMM timings can show **variance** when **L2** state or **GPU boost** drifts. Prefer **median** of many repeats and, when possible, **lock** clocks for **comparative** runs. A **±5 TFLOPS** swing at **1100+ TFLOPS** is **~0.5%**—smaller than **iter068**’s polish over **iter040**, so **tight** methodology matters for **last-mile** claims.

## Correctness beyond TFLOPS

Throughput is meaningless if **metadata** and **packed** indices disagree. The benchmarks include **host-side** checks; when you fork kernels, keep those checks **enabled** until **iter143-class** changes stabilize. **TK** and **unroll** are the highest-risk edits for **silent** misalignment.
