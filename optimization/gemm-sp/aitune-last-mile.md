# AI-Tune Last Mile: `.co` vs `.cu` and Automated Exploration

AI-tune on sparse GEMM is not “replace the engineer”—it is **high-volume compile-measure** over a **structured neighborhood** of kernels Choreo already understands. This page explains the **boundary** where **generated `.co`** stops scaling and **hand-edited `.cu`** (or exported CUDA) picks up, using **FP16** numbers from `README_gemm_sp_f16_aitune_2026-03-25.md` and the **E4M3** ladder from `README_e4m3_aitune_2026-03-21.md`.

## What AI-tune automates

Each iteration typically:

1. **Mutates** one or a small set of parameters: stages, warp split, swizzle, metadata load style, flags, unroll factors, TK, TMA descriptors.
2. **Builds** the benchmark target (`benchmark/performance/gemm_sp/` tree—many **`.co`** variants and dated **`.cu`** subfolders).
3. **Runs** the standard harness (warmup, timed repeats, TFLOPS print).
4. **Records** the result in a README or `results.tsv` for diffing.

The **search** is cheap relative to human **hypothesis latency**; the **hard part** remains **correctness** (2:4 metadata must stay **aligned** with packed operands) and **interpretation** (did TFLOPS move because of **sync**, **TMA**, or **occupancy**?).

## FP16: the `.co` plateau (iter120)

The **best `.co`** outcome in the documented FP16 sweep is **iter120 at 434 TFLOPS**, from **1p2c + 3-stage**—already a **major** rewrite of the schedule vs the **368 TFLOPS** baseline, but still **short** of what hand code eventually reaches.

| Artifact class | Representative | TFLOPS | Role |
|----------------|--------------|--------|------|
| Baseline `.co` | baseline config | **368** | 1p1c, swizzle64, TK64, 2-stage |
| Best `.co` | iter120 | **434** | 1p2c + 3-stage |
| Strong `.cu` | iter137 | **543** | unroll 24 + FTZ |
| Best overall | iter143 | **655** | TK128 + TMA metadata + split RHS TMA |

**Reading:** **~+18%** from baseline to best `.co` (368→434), then **~+25%** more from iter120 to iter143 (434→655) once **CUDA-side** control appears. The **second leg** is not “small tuning”—it is **different expressiveness**.

## Why `.co` hits a ceiling

Choreo’s `.co` path optimizes within **compiler-chosen** loop nests, **register** allocation, and **automatic** async proxy placement. Sparse GEMM stresses **three coupled** concerns:

- **Operand TMA** (lhs/rhs, possibly split),
- **Metadata** movement (scalar, vector, or TMA),
- **WGMMA** batching and **warpgroup** barriers.

When the compiler **serializes** metadata consumption with MMA in a way no **single pragma** fixes, **manual unroll**, **explicit prefetch**, and **TMA descriptor** tricks require **`.cu`** surface area. That shows up empirically as **iter120** vs **iter137/iter143**.

## FP16: organic `.cu` vs peak `.cu`

**iter137 (543 TFLOPS)** is described as the best **“organic” `.cu`**—meaning incremental edits that **expose ILP** (unroll 24) and **numerical fast paths** (FTZ) without yet changing the **full** memory hierarchy story.

**iter143 (655 TFLOPS)** adds **TK128**, **TMA-backed metadata**, and **split RHS TMA**: these are **structural** memory-system changes, not just loop tweaks.

**Profile narrative:** if **iter137** improves **inner-loop** issue but **iter143** shifts **DRAM/L2** balance, Nsight should show **lower** `dram_throughput` bubbles **or** better **L2 hit** behavior on the metadata stream after iter143.

## E4M3: AI-tune without a sharp `.co`/`.cu` story in the headline table

The E4M3 README emphasizes **iter001 … iter068** on a **strong baseline (671 TFLOPS)**. The **largest discrete win** is **iter040 (1090 TFLOPS)** from **3-stage pipeline**—about **+62%** vs baseline in the documented summary—followed by **iter068 (1127 TFLOPS)** with **early empty arrive** for **+68%** overall.

AI-tune’s role here is the same **machinery**, but the **starting point** already includes **128/128 swizzle** and **prepack**, so the search spends more iterations on **pipeline** and **barrier** polish than on “enable basic TMA.”

| Iteration | TFLOPS | Interpretation for the sweep |
|-----------|--------|------------------------------|
| iter001 | **759** | Metadata **on** the TMA plane |
| iter016 | **772** | **Barrier** simplification |
| iter023 | **811** | **SW pipe** + **fine WG wait** |
| iter036 | **897** | **1p2c** |
| iter040 | **1090** | **3-stage** depth breakthrough |
| iter068 | **1127** | **Arrive** timing (best) |

## Cross-dtype: what transferred in the logs

| Pattern | FP16 milestone | E4M3 milestone |
|---------|----------------|----------------|
| Metadata staging / TMA meta | iter143 | iter001 |
| Fine warpgroup sync | chain: `warpgroup_wait<1>` | iter023 |
| 1p2c | iter120 (with 3-stage) | iter036 |
| 3-stage | iter120 / iter143 context | iter040 |
| Barrier micro-optimization | (secondary in F16 table) | iter016, iter068 |

The **transfer** is **causal structure** (what to try next), not **parameter equality**.

## Practical workflow for readers

1. **Freeze** problem size (**4096×8192×8192**) and **build flags**; only change **one family** of edits per run.
2. **Establish** baseline TFLOPS and **save** the artifact path under `benchmark/performance/gemm_sp/`.
3. **Run** AI-tune or a manual grid on **`.co`** until TFLOPS **flatten** (FP16: near **iter120** class).
4. **Export** hot `.co` to `.cu` (or work in the iter subfolders) for **unroll**, **explicit meta prefetch**, **TK** changes, **split TMA**.
5. **Re-verify** **numerical** checks—sparse **metadata** bugs are **silent** until a bitwise compare fails.

## Efficiency vs peak (sanity)

- **FP16** best **655 TFLOPS** vs **1513 TFLOPS** dense peak ≈ **43%**—respectable for **sparse** with **metadata tax**.
- **E4M3** best **1127 TFLOPS** vs **3026 TFLOPS** FP8 peak ≈ **37%**—similar **band**, different **dominant bottleneck** (sync at the end).

## Summary

AI-tune **compresses calendar time** on the **long tail** of kernel tuning: it finds **iter120**, **iter040**, and **iter068** faster than ad-hoc guessing. The **last mile** still belongs to engineers when **compiler schedules** cap **TFLOPS**—that is the **iter120 → iter143** gap on FP16. Keep **`.co` for breadth** and **`.cu` for depth**; measure every step in **TFLOPS** on the **same harness** so the tables stay comparable.

## Regression hazards specific to sparse GEMM

AI-tune can propose **legal** schedules that **violate** 2:4 invariants subtly—e.g. **metadata chunk** mis-aligned to **K** tile boundaries, or **double consumption** of a **packed** fragment under **unroll**. Mitigations:

- Keep a **small** self-check (deterministic seed) in CI.
- When TFLOPS **jumps** unexpectedly, **mistrust** until **bitwise** or tight **tolerance** checks pass.
- **Diff** metadata load **offsets** when **TK** changes.

## How to cite iterations in bug reports

Include: **dtype** (FP16 vs E4M3), **problem size**, **iterNNN** label, **TFLOPS**, **git hash** or **file path** under `benchmark/performance/gemm_sp/`, and **build flags**. That makes **repro** feasible without re-running the entire sweep.

## When not to use AI-tune

If the **baseline** does not **match** the tutorial’s **event** model (producer/consumer roles), **automated** edits will **amplify** bugs. Fix **correctness** and **single-GPU determinism** first; then enable **wide** search.

## Closing table: who owns which leap

| Leap | Primary owner |
|------|----------------|
| 368 → 434 | AI-tune on **`.co`** (structure: 1p2c, 3-stage) |
| 434 → 543 | Engineer-guided **`.cu`** (unroll, FTZ) |
| 543 → 655 | Engineer + TMA/meta/TK (**iter143**) |
| 671 → 1090 | AI-tune + **manual** validation (E4M3 staging/sync/depth) |
| 1090 → 1127 | **Barrier micro** (iter068) |

This division is **empirical** from the README timelines, not a rigid rule—your tree may move some **iter143** ideas back into **`.co`** as the compiler improves.

## README filenames as contracts

Treat `README_gemm_sp_f16_aitune_2026-03-25.md` and `README_e4m3_aitune_2026-03-21.md` as **contracts**: they tie **iter** labels to **TFLOPS** and **short** change blurbs. When the Choreo repo advances, **new** READMEs may supersede dates—update this tutorial’s **links** if filenames change, but keep the **method** (profile → pattern → measure).

## Suggested sweep ordering (automation-friendly)

1. **Metadata** path variants (`__ldg`, `uint2`, hoisting) — cheap codegen edits.
2. **`warpgroup_wait`** depth — low risk if **tests** pass.
3. **Stages** 2 → 3 — watch **SMEM**.
4. **1p1c → 1p2c** — watch **sync** complexity.
5. **TK** and **TMA meta** — high reward, **high** validation cost.
6. **`.cu` unroll** — after **`.co`** plateau.

E4M3 sweeps may **reorder** steps 1–4 because **baseline** already includes **strong** TMA.

## One-line outcomes (for slides)

- **FP16:** *368 → 655 TFLOPS; best `.co` 434 (iter120); best `.cu` 655 (iter143).*
- **E4M3:** *671 → 1127 TFLOPS; 3-stage breakthrough at iter040 (1090); best iter068.*

These lines are **accurate** to the READMEs cited in [index](index.md) and anchor **executive** summaries without losing the **engineering** story above.

## Version skew

If your local Choreo checkout **predates** the AI-tune READMEs, you will still find **gemm_sp** baselines under `benchmark/performance/gemm_sp/`; iteration-prefixed files may be **missing**. The **patterns** in [pattern-optimizations](pattern-optimizations.md) remain valid—only the **exact** TFLOPS rows require **regeneration** on your **GPU stepping** and **driver**.
