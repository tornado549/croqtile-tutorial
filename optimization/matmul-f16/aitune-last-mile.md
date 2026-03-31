# AI-Tune: Last Mile and Shipped Kernels

This section records **how** the final percentages were extracted—build flags, benchmark harness knobs, and the **iteration** that shipped—without duplicating the full 65-iteration branch history (see `origin/ai-tune/2026-03-23/matmul_f16` in the README).

## Hardware and ceiling discipline

Device: **H800 PCIe**, **SM90a**, **114 SMs** (matching the persistent-kernel discussion in [Chapter 7](../../tutorial/ch07-persistent.md)).

Two different “tops” appear in documentation:

- **~1513 TFLOPS** — theoretical FP16 tensor **peak** headline for the part class.
- **~380 TFLOPS** — what **cuBLAS** achieves in practice on this stack.

The tuned kernels are best judged against **cuBLAS**, not the marketing peak. **iter061** is quoted in the README as **80.7% of cuBLAS** at **8192³** and **100.5% of cuBLAS** at **2048³**—the smaller cube is **friendlier** to the chosen tile and stage mix, so **short-cube efficiency can exceed** the library on that point while **8192³** remains the **stress** case.

## Shipped kernels (summary table)

| Artifact | TFLOPS | Problem | Role |
|----------|--------|---------|------|
| **iter048** | **354.1** | 2048³ | **1p1c**, **WN=176**, **STAGES=3** — demonstrates the **3-stage sweet spot** on a mid-size cube ([pipelining](../../tutorial/ch03-pipeline.md)). |
| **iter050** | **~375** | 4096³ | **1p2c split-output**, **WN=128**, **STAGES=2** — validates split-output before the final **8192³** push ([warp specialization](../../tutorial/ch06-warpspec.md)). |
| **iter057** | **382.5** | 8192³ | **1p2c split-output**, **WN=152**, **non-persistent** — **best** headline vs. baseline (**+83%** over **208.7**). |
| **iter061** | **380.6** | 8192³ | **1p2c split-output**, **WN=160**, **K-unrolled** — **occupancy-optimal** point in the sweep; **2 CTAs/SM**, **~114.7 KB** SMEM. |

Baseline reference (main): **208.7 TFLOPS** at **8192³** (**1p1c**, **WN=128**, **STAGES=4**), from the dynamic SM90 path (`matmul_f16_dyn_sm90.co` family).

## Build and run (iter061 template)

From the Choreo repo root (after `make build`):

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co \
  -o /tmp/iter061.cute.result && bash /tmp/iter061.cute.result --execute
```

Swap the input `.co` for **`*_iter057_*`**, **`*_iter050_*`**, or **`*_iter048_*`** to reproduce the other shipped points; the flag line stays the same in the README.

### iter057 (best 8192³ TFLOPS)

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter057_1p2c_so_wn152_nonpersis.co \
  -o /tmp/iter057.cute.result && bash /tmp/iter057.cute.result --execute
```

### iter050 (4096³ split-output)

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter050_1p2c_splitout.co \
  -o /tmp/iter050.cute.result && bash /tmp/iter050.cute.result --execute
```

### iter048 (2048³, 3-stage sweet spot)

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter048_s3_wn176_best.co \
  -o /tmp/iter048.cute.result && bash /tmp/iter048.cute.result --execute
```

### Harness environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CHOREO_TIMING_WARMUP` | 10 | Warmup iterations before timing |
| `CHOREO_TIMING_REPEAT` | 500 | Timed iterations |
| `CHOREO_DISABLE_TIMING` | 0 | Set `1` to skip timing |
| `CHOREO_SKIP_VERIFY` | 0 | Set `1` to skip correctness checks |

For apples-to-apples TFLOPS, keep verify **on** during development and only disable when you are certain the kernel is correct.

## iter057 vs. iter061 (two “wins,” different goals)

**iter057** (**382.5 TFLOPS** @ **8192³**) is the **absolute peak** in this README against the **208.7** baseline—use it when **8192³** is the only metric that matters (strong scaling studies, peak headline).

**iter061** (**380.6 TFLOPS** @ **8192³**) trades **1.9 TFLOPS** for a **more robust** cross-size story: **100.5% cuBLAS** @ **2048³** and **80.7% cuBLAS** @ **8192³**. Prefer **iter061** when the same binary must perform well on **both** **small** and **large** cubes without re-tuning **`MATMUL_*`** constants per job.

Both share **1p2c split-output** and the same **compiler** flag bundle; the difference is **WN** (**152** vs. **160**), **K-unroll**, and the **wait-depth** tuning that landed with **Phase 3**.

## Phase 3: WN sweep and the occupancy cliff

At **8192³**, Phase 3 focused on **WN** after split-output **1p2c** had already unlocked **cuBLAS-class** throughput.

- **iter061 (WN=160, K-unroll)** → **380.6 TFLOPS** — slightly below **iter057**’s **382.5**, but **preferred** in documentation for **cross-size** behavior (see **100.5% cuBLAS** at **2048³**).
- **WN=168** — **failed** the sweep: **SMEM > 228 KB** → **1 CTA/SM**. This is a **step-function** loss, not a gentle slope—exactly the kind of threshold occupancy models warn about, but **easy to miss** without measurement.

The compiler gained **`--wgmma-wait-depth=N`** during this phase so **WGMMA pipeline depth** could track **stage strategy** without hand-editing PTX.

**Interpreting the cliff:** At **WN=168**, **SMEM** crossed **228 KB**. On this GPU configuration that collapses residency to **one block per SM**. Throughput does not fall by **2%**—it can fall by **tens of percent** because **latency hiding** across **CTAs** disappears. The **fix** is not “more clever math,” it is **smaller tiles** or **fewer stages** until **two CTAs/SM** return.

## Phase timeline (condensed)

| Phase | Iterations | Focus | Outcome |
|-------|------------|-------|---------|
| 1 | 001–038 | 1p1c @ **2048³** | **214.3 TFLOPS** after **SMEM** + **lowering** tweaks |
| 2 | 043–057 | Split-output, multi-size | **382.5 TFLOPS** @ **8192³** (**iter057**) |
| 3 | 061–065 | **WN** sweep @ **8192³** | **380.6 TFLOPS** (**iter061**), **`wgmma-wait-depth`**, **WN=168** cliff |

Full **65**-iteration detail lives on branch **`origin/ai-tune/2026-03-23/matmul_f16`**; this tutorial captures the **shipped** checkpoints only.

## What “last mile” means here

The **first ~30 TFLOPS** came from **correct roles** and **reasonable** tiles. The **next ~150+ TFLOPS** came from **joint** tuning of **WN**, **STAGES**, and **1p2c split-output**, plus **non-persistent** launch at **8192³**. The **final few TFLOPS** are **WN sweeps**, **K-unroll**, and **wait-depth** alignment—changes that only make sense once **occupancy** is already **on the right side** of the **228 KB** cliff.

## Reproducibility checklist

1. Build **`./choreo`** from the same **Choreo** revision as the **`.co`** file (or expect codegen drift).
2. Use **`-arch=sm_90a`** for **H800**-class **Hopper**.
3. Keep **warmup** and **repeat** at README defaults unless you are diagnosing **noise**; for noisy hosts, increase **`CHOREO_TIMING_REPEAT`**, not **warmup**, first.
4. Record **GPU clock** state if comparing against external **cuBLAS** numbers—thermal or power limits can narrow **~380 TFLOPS** references by a few percent.

## How this relates to the optimization index

The parent page [Performance optimization](../index.md) lists this case alongside **sparse** and **block-scaled** GEMMs. **Dense FP16** is the **simplest** data path: no **metadata** tiles, no **scale** vectors—only **operand** staging, **accumulator** layout, and **launch** geometry. That simplicity makes the **208.7 → 382.5** story a good **second** tutorial after [Profiling setup](../setup-profiling.md).

## cuBLAS as the efficiency denominator

Quoting **“80.7% cuBLAS”** or **“100.5% cuBLAS”** anchors expectations: **cuBLAS** is **highly** optimized and **vendor-tuned**. Exceeding it on **2048³** while **matching** it within **~1%** on **8192³** (**iter061** vs. **iter057**) is a **useful** product decision—**libraries** also shift with **driver** and **BLAS** version, so always record the **reference** build when publishing **percent** metrics.

## File names and where to look in Choreo

The README embeds **dated** filenames (`matmul_f16_aitune_2026-03-23_*`) so **artifacts** stay **immutable** in git history. The **conceptual** families live beside them:

- **Dynamic baseline:** `benchmark/performance/matmul/matmul_f16_dyn_sm90.co`
- **1p1c teaching / reference:** `matmul_f16_dyn_sm90_warpspec_1p1c.co`
- **1p2c reference:** `matmul_f16_dyn_sm90_warpspec_1p2c.co`
- **Shipped tune points:** `*_iter048_*`, `*_iter050_*`, `*_iter057_*`, `*_iter061_*` under the same directory

When bisecting a regression, compare **`.co`** files with **`diff -u`** focusing on **`MATMUL_*`** macros and **`parallel`** structure—those dominate **SMEM** and **roles**.

## Closing numbers (copy/paste friendly)

- **Baseline:** **208.7 TFLOPS** @ **8192³** (**1p1c**, **WN=128**, **STAGES=4**)
- **Best:** **382.5 TFLOPS** @ **8192³** (**1p2c split-output**, **WN=152**, **non-persistent**) — **+83%** vs. baseline
- **Practical ceiling reference:** **~380 TFLOPS** **cuBLAS**, **~1513 TFLOPS** theoretical **FP16** peak headline (**H800 PCIe** class)

## Note on `--wgmma-wait-depth=N`

The README records this flag as a **Phase 3** addition. Treat **`N`** as **coupled** to **STAGES** and **consumer** issue rate: there is **no** universal **N** across **iter048** vs. **iter061** because **operand** **ring** depth and **WN** change **how** early **WGMMA** can **drain**. When reproducing **iter061**, use the **exact** **choreo** invocation from the README unless you are **explicitly** sweeping **`N`**.

## Further reading

- Full tables: `choreo/benchmark/performance/matmul/README_matmul_f16_aitune_2026-03-23.md`
- Concepts: [matmul-f16 index](index.md), [baseline analysis](baseline-analysis.md), [patterns](pattern-optimizations.md)
