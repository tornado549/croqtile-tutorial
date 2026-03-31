# Optimization Patterns: Sparse GEMM (FP16 and E4M3)

This page is a **pattern catalog** with **measured anchors** from the 2026-03 AI-tune logs. The narrative is always: **profile → hypothesize bottleneck → apply one cluster of changes → re-measure**. Patterns that help **FP16** usually help **E4M3** as well; the difference is which pattern is **binding** first.

## Pattern map at a glance

| Pattern cluster | Typical symptom | FP16 touchpoint | E4M3 touchpoint |
|-----------------|-----------------|-----------------|----------------|
| `warpgroup_wait<1>` / fine sync | Bubbles at WG boundaries | Early chain (~+4% class) | iter023 **811** |
| `--wgmma-split-batch` | MMA batching inefficiency | F16 sweep | Same idea for WG math |
| `__ldg` metadata | Uncached / scalar metadata path | +0.5% and later hoists | TMA path reduces scalar pressure |
| 1p2c + deeper stages | Producer cannot run far enough ahead | iter120 **434** (3-stage) | iter036 **897**, iter040 **1090** |
| `stmatrix` / store path | Epilogue or spill traffic | +2% in F16 chain | Less dominant when math-bound |
| L2 / 128B promotion | Metadata not resident where needed | +0.7% in F16 chain | Combines with staging |
| `uint2` metadata vectors | Instruction overhead on meta | +8% in F16 chain | uint-style vectorization in related sweeps |
| Hoisted metadata `__ldg` | Late consume of metadata | +7% in F16 chain | Aligns with SW pipelining |
| Inner unroll (e.g. 24) | ILP inside K tile | iter137 **543** | Same philosophy |
| FTZ / denorm mode | Rare edge slow paths | +0.4% in F16 chain | dtype-dependent |
| TMA metadata staging | Metadata not overlapped with operands | iter143 ingredients | iter001 **759** |
| TK128 + split RHS TMA | Tile geometry vs bandwidth | iter143 **655** | (E4M3 uses own TK/swizzle) |
| Early empty / merged barrier | Full/empty latency | — | iter016 **772** |
| Early empty **arrive** | Finer signal timing | — | iter068 **1127** (best) |

The rest of this document walks clusters in **rough chronological** order as they appeared in the **FP16** story, then calls out **E4M3** milestones that mirror the same ideas.

---

## 1. Fine-grained warpgroup synchronization (`warpgroup_wait<1>`)

**Problem.** Producer and consumer warpgroups coordinate through **async proxies** and **barriers**. Coarse waits can leave **lanes idle** while data is actually ready.

**Pattern.** Use the **smallest sufficient** warpgroup wait depth—here summarized as `warpgroup_wait<1>` in the FP16 optimization chain—so consumers **resume** as soon as the **minimum** dependent async slice completes.

**FP16.** Documented as an early improvement on the order of **+4%** relative to the then-current kernel in the F16 chain.

**E4M3.** **iter023** at **811 TFLOPS** combines **software pipelining** with `warpgroup_wait<1>`: the profile suggested **sync-induced bubbles**; tightening wait granularity **recovers cycles** without changing tile shape.

**Transfer lesson.** Before widening tiles or adding stages, ensure **WG-level waits** are not **overserialized**.

---

## 2. MMA batch configuration (`--wgmma-split-batch`)

**Problem.** Hopper **WGMMA** can split work across **batches** of K fragments. A poor split leaves **tensor cores underfed** relative to operand delivery.

**Pattern.** Compiler/driver flags such as **`--wgmma-split-batch`** (as logged in the FP16 chain) adjust how **batches** map to **instructions**.

**FP16.** Reported near **+5%** in the documented sequence.

**E4M3.** The same **batching** concern applies; exact flag names may differ by build, but the **optimization intent** is identical: **match MMA batch size** to **stage depth** and **fragment layout**.

**Transfer lesson.** If Nsight shows **WGMMA issue slots** gapping while **shared** is ready, revisit **batching** before blaming TMA.

---

## 3. Metadata through read-only cache (`__ldg`)

**Problem.** Metadata often lives in **global** and is read **every K tile**. Scalar loads that miss L1/L2 behave like **pointer chasing** next to **wide TMA**.

**Pattern.** Force **read-only texture/L2-friendly** paths with **`__ldg`-style** loads (or equivalent intrinsics) so metadata streams **predictably**.

**FP16.** Small but real: about **+0.5%** in the chain—meaning the baseline already cached somewhat, but **consistency** matters across tiles.

**E4M3.** **iter001** jumps to **759 TFLOPS** with **TMA metadata staging**, which is a **stronger** statement: move metadata toward **the same async machinery** as operands where possible.

**Transfer lesson.** **`__ldg`** is the **scalar** version of “treat metadata like bandwidth.” **TMA staging** is the **vector/async** version.

---

## 4. Warp specialization 1p2c and multi-stage pipelines

**Problem.** In **1p1c**, one producer warpgroup must **issue all TMA** and often **assist** with setup that **steals** issue slots from where the consumer needs **steady** `wgmma` streams.

**Pattern.** **1p2c**: **one producer**, **two consumer** warpgroups (or split consumer roles—exact split follows the Choreo schedule), paired with **3-stage** (or more) **operand rings** so producers run **ahead** of consumers.

**FP16.** **iter120** reaches **434 TFLOPS** on **`.co`** with **1p2c + 3-stage**—a large structural win (~**+9%** class in the documented chain vs the prior step).

**E4M3.** **iter036** at **897 TFLOPS** marks **1p2c**; **iter040** at **1090 TFLOPS** is the **3-stage breakthrough**, roughly **+62%** vs the **671** baseline in the README narrative.

**Transfer lesson.** **Depth** (stages) without **producer throughput** (1p2c) often fails; **1p2c** without **enough stages** still bubbles.

---

## 5. Store matrix (`stmatrix`) and epilogue efficiency

**Problem.** Sparse GEMM is not only **MMA**; **accumulators** must **write back** through shared or registers with **bank-safe** patterns.

**Pattern.** Enable **`stmatrix`** (where the toolchain supports it) so **stores** match **Hopper’s** preferred **matrix store** paths.

**FP16.** About **+2%** in the documented F16 chain.

**E4M3.** When the kernel is **math- and sync-bound** above **1000 TFLOPS**, epilogue tricks matter **less** unless profiles show **store** as hot.

**Transfer lesson.** Apply **`stmatrix`** when **epilogue** or **spills** show up; skip if **metadata** is still the long pole.

---

## 6. L2-friendly metadata promotion (128B lines)

**Problem.** Metadata arrays are small per tile but **random-accessed** across **CTAs**; without attention to **line size**, you **thrash** L2.

**Pattern.** Align and pad so **metadata** touches **128B** granularity where possible—**coalesce** meta loads with **`uint2`** vectors and **stable** base pointers.

**FP16.** **L2 128B promotion** ~**+0.7%**; **`uint2` metadata** ~**+8%**; **hoisted `__ldg` metadata** ~**+7%**—these three are **one story**: **how** metadata arrives in registers **before** MMA consumes it.

**E4M3.** **Software pipelining** (iter023) and **TMA metadata** (iter001) achieve analogous **latency hiding**.

**Transfer lesson.** Treat **`uint2` + hoisting** as **software pipelining the metadata plane**.

---

## 7. Inner unroll and FTZ (FP16 iter137)

**Problem.** Compiler-generated **`.co`** schedules may not **unroll** the inner K loop enough to **overlap** address math, metadata **prefetch**, and **`wgmma`**.

**Pattern.** Hand **`.cu`**: **unroll 24** (documented) plus **flush-to-zero** (**FTZ**) to avoid **denorm** slowdowns on rare data.

**FP16.** **iter137** reaches **543 TFLOPS**—the best **“organic” `.cu`** before **TK128 / TMA-metadata / split-RHS** land in **iter143**.

**E4M3.** Inner unroll is still valid; FTZ is **less central** if inputs are **E4M3** and the FP16 accum path is already **well-scoped**.

**Transfer lesson.** **Unroll** when the profile shows **short dependency chains** on **loop indices**; pair with **FTZ** when host data allows.

---

## 8. TK128, TMA metadata, split RHS TMA (FP16 iter655)

**Problem.** **TK64** keeps **K tiles small**, which can **increase** trip count and **metadata** traffic per unit work.

**Pattern.** Move to **TK128**, drive **metadata** through **TMA** like operands where possible, and **split RHS TMA** so **bandwidth** tracks **consumer** demand.

**FP16.** **iter143** at **655 TFLOPS** is the **best overall** in the log—**+78%** vs **368** baseline.

**E4M3.** Uses different **TK/swizzle** defaults (**128/128** from the start); the **parallel** is **iter001** metadata staging + **iter040** depth, not necessarily identical TK.

**Transfer lesson.** **Tile K** jumps require **re-validating swizzle**, **metadata**, and **stage counts** together—never **TK** alone.

---

## 9. Early empty, merged barriers, early arrive (E4M3)

**Problem.** **Async pipelines** use **empty/full** phases. If **signals** are late or **barriers** are **over-synchronized**, producers and consumers **lose overlap**.

**Pattern.**

- **Early empty** and **merged barrier** reduce **round trips** on proxy state (**iter016**, **772 TFLOPS**).
- **Early empty arrive** refines **who** signals **when** (**iter068**, **1127 TFLOPS**, best).

**FP16.** The F16 README emphasizes **metadata** and **TMA** more than this exact barrier vocabulary, but **warpgroup_wait** and **stage** tuning play the **same role**.

**Transfer lesson.** Above **~900 TFLOPS** on E4M3, **sync polish** is worth **double-digit** TFLOPS—measure with **Nsight** **warp stalls** on **barrier**.

---

## FP16 optimization chain (ordered narrative)

The README summarizes this **sequence** (each step relative to its immediate predecessor in the campaign):

1. `warpgroup_wait<1>` — ~**+4%**
2. `--wgmma-split-batch` — ~**+5%**
3. `__ldg` metadata — ~**+0.5%**
4. **1p2c + 3-stage** — ~**+9%**
5. `stmatrix` — ~**+2%**
6. **L2 128B** metadata promotion — ~**+0.7%**
7. **`uint2` vector metadata** — ~**+8%**
8. **Hoisted metadata `__ldg`** — ~**+7%**
9. **Unroll 24** — ~**+3%**
10. **FTZ** — ~**+0.4%**

Then **TK128 + TMA metadata + split RHS TMA** (iter143) sits at the **end** as the **largest structural** jump beyond incremental percents.

## E4M3 ladder (key TFLOPS only)

| TFLOPS | Note |
|--------|------|
| **671** | baseline |
| **759** | TMA metadata staging |
| **772** | early empty + merged barrier |
| **811** | SW pipeline + `warpgroup_wait<1>` |
| **897** | 1p2c |
| **1090** | 3-stage |
| **1127** | early empty arrive (best) |

## When a pattern does not transfer one-to-one

- **Swizzle**: FP16 baseline **64** vs E4M3 **128/128**—do not copy numbers blindly; **validate bank conflicts**.
- **Accum dtype**: E4M3 → **FP16** accum changes **register** pressure vs pure FP16 sparse.
- **Compiler**: **`.co`** may **refuse** some unroll/TMA-meta combos—see [aitune-last-mile](aitune-last-mile.md).

## Closing checklist

Before closing a sparse GEMM session, verify:

1. **Metadata** loads are **vectorized**, **hoisted**, and ideally **TMA-staged**.
2. **Stages** match **producer rate**; **1p2c** is justified by **profiled** producer slack.
3. **Warpgroup waits** and **empty/full** are **minimal** without **race**.
4. **TK** changes forced **swizzle/TMA/meta** retuning.
5. **`.co` plateau** identified before spending days on **micro** tweaks—hand **`.cu`** may be cheaper.

This checklist is **dtype-agnostic**; the benchmarks show it working for both **FP16** and **E4M3** paths.

---

## Appendix A: Percent chain math (FP16)

The README lists **per-step** uplifts (e.g. **+4%**, **+5%**). These are **local** improvements vs the **kernel version immediately before** that edit in the campaign. They do **not** multiply cleanly to the **global** **368 → 655** ratio because **interactions** exist (e.g. **hoisting** matters more after **vectorization**). When writing your own notes, always record **absolute TFLOPS** alongside **percent** so **regressions** are obvious.

**Global ratios (documented):**

- Baseline **368** → best **655** → **+78%** overall for FP16 sparse GEMM on this shape.
- E4M3 baseline **671** → best **1127** → **+68%** overall.

## Appendix B: Risk list when applying patterns

| Risk | Mitigation |
|------|------------|
| Wrong **metadata** for repacked **A** | Run host **numerical** check every build; compare against **dense reference** on small M,N,K. |
| **3-stage** SMEM overflow | Print **shared** usage; watch **occupancy** cliff in profiler. |
| **1p2c** desync | Verify **event** order matches [Ch6](../../tutorial/ch06-warpspec.md) teaching kernels. |
| **FTZ** changes accuracy | Gate FTZ behind **benchmark** flag; document **ULP** impact for production. |
| **TK128** mis-swizzled | Diff **TMA descriptor** setup vs TK64; re-run **bank** conflict checks. |

## Appendix C: E4M3 vs FP16 — which pattern to try first

If your **measured** sparse FP16 kernel is **below ~450 TFLOPS** on this class of GPU, the logs suggest **metadata vectorization**, **`__ldg`**, **1p2c+3-stage**, and **`warpgroup_wait`** before **exotic** TMA-meta.

If your **E4M3** kernel is **already ~850+ TFLOPS**, the logs suggest **barrier / early empty / arrive** and **stage** tuning before more **operand** widening—your **math** is already **fed** enough that **sync** dominates.

## Appendix D: Choreo vocabulary mapping

Choreo **`.co`** sources express **TMA**, **swizzle**, **parallel** warps, and **pipelined** loops; the **AI-tune** branches apply **flags** and **parameter** macros that map to the same **CUDA** concepts this document names. When a README cites **`--wgmma-split-batch`**, treat it as **“ask the toolchain to reshape WGMMA batching”**—the exact spelling in your tree may differ slightly by revision.

## Appendix E: Duplicate experiment guardrails

Automation can **repeat** successful mutations until **noise** dominates. Keep:

1. **Fixed** GPU clock (where policy allows) or **wide** repeat count.
2. **Pinned** host memory if the harness uses **async** copies.
3. **One** change per experiment when **debugging**; **bundled** changes when **sweeping** known-good neighborhoods.

The **iter040** E4M3 result (**1090 TFLOPS**) is an example where a **bundled** **3-stage** change is justified—**stage depth** rarely works without **compatible** producer staging.

## Appendix F: Further reading in this repo

- [Setup profiling](../setup-profiling.md) — TFLOPS and efficiency definitions.
- [Ch4: TMA swizzle](../../tutorial/ch04-tma-swizzle.md) — swizzle modes referenced by both baselines.
- [Ch10: C++ inline macros](../../tutorial/ch10-cpp-inline-macros.md) — mentions **gemm_sp**-style producer/consumer asymmetry.

Together with the milestone tables on this page, you can reconstruct **why** each iteration is a plausible **next step** from the one before.

## Appendix G: Worked comparison (FP16 milestones)

| Step | TFLOPS | Delta | Cumulative vs 368 |
|------|--------|-------|-------------------|
| Baseline | 368 | — | — |
| iter120 (best `.co`) | 434 | +66 | +18% |
| iter137 (`.cu` organic) | 543 | +109 | +48% |
| iter143 (best) | 655 | +112 | +78% |

The **largest single jump** in this table is **iter120 → iter137** if measured linearly—but recall **iter137** stacks **many** prior micro-optimizations from the percent chain; the table is **milestone** sampling, not **atomic** edits.

## Appendix H: Worked comparison (E4M3 milestones)

| Step | TFLOPS | Delta | Notes |
|------|--------|-------|-------|
| Baseline | 671 | — | strong starting schedule |
| iter001 | 759 | +88 | metadata → TMA |
| iter040 | 1090 | +293 | 3-stage (vs iter036 897) |
| iter068 | 1127 | +37 | barrier polish after **>1000** |

After **iter040**, each **+37 TFLOPS** is **hard won**—typical of **sync-limited** regimes.

## Appendix I: Pattern rejection criteria

Stop pursuing a pattern when:

1. **Occupancy** drops enough to **erase** math gains (watch **warps active**).
2. **SMEM** exceeds **Hopper** limits for your **cluster** configuration.
3. **Correctness** checks **fail** intermittently (**race**).
4. **TFLOPS** flatlines across **three** independent mutations of the same **family** (diminishing returns).

Then **rotate** to the other dtype’s **milestone** for ideas—for example, if FP16 is **metadata-stuck**, study **iter001** on E4M3 for **TMA meta** layout hints.

## Appendix J: Dense vs sparse pattern overlap

Patterns from [dense matmul](../matmul-f16/pattern-optimizations.md)—**1p2c**, **multi-stage**, **split output**, **WN/tile** search—still apply, but **sparse** adds **metadata** as a **first-class** operand. In practice, **dense** work teaches **TMA + WGMMA rhythm**; **sparse** work teaches **never starve MMA while waiting on meta**. Readers optimizing **both** should keep **two** profiling presets: one that highlights **DRAM** and one that highlights **L1/L2** traffic on **small** meta arrays.
