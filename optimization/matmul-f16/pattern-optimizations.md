# Optimization Patterns: Dense GEMM FP16

Each pattern below ties to a **measurable** jump in the AI-tune history and maps back to Lv0 concepts. The through-line: **first** align shared-memory footprint and occupancy with the target problem size; **then** specialize roles and output staging so producer and consumer rarely wait; **last** tune compiler lowering details (`stmatrix`, barriers, TMA cluster awareness) for instruction-level overlap.

## 0. Pattern map (quick reference)

| Symptom (measured) | Pattern | Representative jump |
|--------------------|---------|---------------------|
| Good roles, high SMEM, low occupancy | Reduce **STAGES** or **WN** | iter004: **204 → 208.9** @ **2048³** |
| Under-latency-hiding on K | Increase **STAGES** *if* SMEM allows | iter048: **242 → 354.1** @ **2048³** |
| Output tile contention (large WM/WN) | **1p2c split-output** | iter050: **~375** @ **4096³** |
| Persistent overhead / inner tile wins | **Non-persistent** at large **S** | iter057: **382.5** @ **8192³** |
| Near cuBLAS, cliff on next WN | Stop before **SMEM > 228KB** | **WN=168** bad, **WN=160** good |

## 1. Warp specialization (1p1c) as the foundation

**Concept:** [Chapter 6](../../tutorial/ch06-warpspec.md) — split **TMA producer** and **WGMMA consumer** warpgroups with **`inthreads.async`**, **`full` / `empty`** events, and **`mma.commit`** between K stages.

**Why it matters:** Without role split, a single warpgroup interleaves copy and math on one timeline; Hopper’s **async TMA** and **WGMMA** are designed for **concurrent** progress. The baseline already used 1p1c; later gains came from **geometry around** that split, not from abandoning it.

**Evidence:** Phase 1 stayed inside 1p1c while moving from **204** to **214.3 TFLOPS** at **2048³** (iter023), proving that **micro-architectural** and **compiler** flags still matter once roles are fixed.

**Measure:** Compare **TFLOPS** before and after enabling **`--use-warpspec`** on the *same* tile constants; if the delta is small, your limiter is **not** role assignment—it is **footprint** or **stage depth**.

## 2. Pipeline depth (STAGES): the Chapter 3 trade in SMEM

**Concept:** [Chapter 3](../../tutorial/ch03-pipeline.md) — **multi-buffer** operand tiles along K; each stage is another ring slot in **`lhs_load_s` / `rhs_load_s`**.

**Pattern:** Increasing **STAGES** deepens **memory latency hiding** but **linearly increases** operand SMEM. On Hopper, that interacts violently with **CTAs/SM**.

**Data:**

- **iter004** moved to **STAGES=2** with a wider tile (**WN=256**), landing **208.9 TFLOPS** vs. the **204** TFLOPS baseline at **2048³** — a **SMEM pressure** story.
- **iter048** found a **three-stage sweet spot** for **1p1c WN=176**: **354.1 TFLOPS** at **2048³**, a massive jump from **242 TFLOPS** at **STAGES=2** (iter046, same WN).

**Key insight (documented in the tune log):** **Three stages help at 2048³ but hurt at 8192³** because the larger grid amplifies **occupancy** and **wave** effects—extra stages are not “free latency hiding”; they are **bytes that evict concurrent blocks**.

When you read [Ch3](../../tutorial/ch03-pipeline.md), mentally multiply **`MATMUL_STAGES`** by **`MATMUL_TILE_K`** and tile dimensions; that product must fit under the **per-SM shared budget** *after* accounting for **output** and **metadata**.

### Anti-pattern: “always four stages”

The baseline used **STAGES=4** with **WN=128**. That combination is a **reasonable default** for teaching ([Ch6](../../tutorial/ch06-warpspec.md) uses **4** in the excerpt), but it is **not** a universal optimum. Treat **STAGES** exactly like any other knob in [Ch3](../../tutorial/ch03-pipeline.md): **double buffering** is the minimal idea, **triple** or **quad** buffering only pay if **latency hiding** gains exceed **occupancy** losses. The log’s **2048³** result (**354.1** with **3** stages vs. **242** with **2**) is the quantitative proof.

### Size-dependent reversal

The same mechanism explains why **3-stage helps at 2048³ but hurts at 8192³**: at large **S**, **grid parallelism** and **cache behavior** change the **critical path**. Extra stages increase **SMEM** and can **reduce** concurrent blocks enough that **WGMMA** sits idle waiting for **slots**, wiping out the **latency** win. Always **re-sweep** **STAGES** when you change **S** by **4×**.

### TMA layout stays on the critical path

**Concept:** [Chapter 4](../../tutorial/ch04-tma-swizzle.md) — **`tma.copy`** and **`mma.load.swiz`** must agree on **swizzle** and **tile_K** slicing.

This case study did not chase **swizzle** as the primary variable; the big wins were **tile geometry** and **1p2c**. Still, **iter023** included **subspan** refinements alongside **`stmatrix`**, which is a reminder: when **TFLOPS** move **+5%** without changing **WN** or **STAGES**, you are often fixing **addressing** or **operand setup**—the space between **TMA** and **WGMMA**, not the **GEMM** graph itself.

### WGMMA warpgroup shape (Chapter 5 context)

**Concept:** [Chapter 5](../../tutorial/ch05-mma.md) — **WGMMA** runs in **group-4** warpgroups; **WM**, **WN**, **`MATMUL_WARP_K`**, and **`MATMUL_TILE_K`** must tile **K** without remainder in the inner slice loop.

**Pattern:** Once **WN** widens, verify **inner** **`cdiv(MATMUL_TILE_K, MATMUL_WARP_K)`** and **accumulator** layout still match hardware expectations. Phase 3’s **K-unroll** on **iter061** is an explicit **instruction-scheduling** lever on that inner structure—it is not a separate algorithm, it is **how** the **same** math is **issued**.

## 3. Tile width (WN) and K scheduling

**WN** (`MATMUL_WARP_N`) sets the **N** extent of the WGMMA tile. Wider **N** increases **arithmetic intensity** per staged K-slab but grows **operand and sometimes accumulator** footprints.

**Observations:**

- **iter046:** **WN=176**, **STAGES=2** → **242 TFLOPS** at **2048³** (**+13%** over the early baseline family).
- **iter048:** same **WN=176**, **STAGES=3** → **354.1 TFLOPS** at **2048³** — the **stage count** and **WN** **interact**; the optimum is **joint**, not separable.

Later, at **8192³**, **WN** was swept again (Phase 3). **WN=160** with **K-unroll** (iter061) gave **380.6 TFLOPS**, while **WN=168** fell off a cliff: **SMEM > 228 KB** forces **one CTA/SM**, a discontinuous **occupancy** loss. **WN=160** landed near **114.7 KB** SMEM with **two CTAs/SM**—a concrete example of **discrete scheduling thresholds** on real hardware.

**Practical sweep discipline:** When exploring **WN**, step by **multiples of 8** (or the alignment your **`mma`** slice requires) and **recompute SMEM** after **every** candidate. One **bad** step (**168**) cost more than a few TFLOPS—it changed **occupancy class**.

### Joint tuning (WN, STAGES) is non-separable

If you fix **WN** from a **2048³** experiment and only later discover **STAGES** is wrong at **8192³**, you will **revisit** **WN** anyway—operand ring size and **output** slices move together under **1p2c**. The optimization log’s ordering (**iter046 → iter048 → iter050 → iter057 → iter061**) reflects that **dependency chain**: **WN=176** mattered **with** **STAGES=3** at **2048³**, but the **8192³** winners (**152**, then **160**) are **different** because **split-output** and **non-persistent** launch changed the **binding constraint**.

## 4. Split-output 1p2c vs. shared output

**Concept extension:** Still [warp specialization](../../tutorial/ch06-warpspec.md), but with **two consumer warpgroups** (**1p2c**) and **separate output staging** (**split-output**) instead of a **single** **`output_s`** tile contended by multiple writers.

**Pattern:** For **large** tiles, **shared output** creates **SMEM traffic and synchronization** on the accumulator tile. **Split-output** gives each consumer a **private** slice of the output staging space, trading **slightly higher SMEM** for **less contention** and often **better instruction mix**.

**Data:**

- **iter050:** **1p2c split-output**, **WN=128**, **STAGES=2** → about **375 TFLOPS** at **4096³**.
- **iter057:** **1p2c split-output**, **WN=152**, **non-persistent** → **382.5 TFLOPS** at **8192³** — the **best** end-to-end result in the study.

The headline lesson from the README: **1p2c split-output beats shared output for large tiles** because it removes a **serialization point** on the result tile that shows up once **N** and **M** extents are warpgroup-scaled.

Reference kernels: **`matmul_f16_dyn_sm90_warpspec_1p2c.co`** in the benchmark tree and the shipped **`*_iter050_*` / `*_iter057_*`** variants.

### How to recognize output contention in data

You rarely see a smoking gun in a single counter without **NSight**-class detail. Heuristics that *did* correlate here:

- **TFLOPS** rises when moving from **1p1c** to **1p2c** at **larger** **WN**, but **only** when **split-output** is enabled—implying the **consumer** side was **serialized** on **`output_s`** traffic.
- **4096³** (**iter050**, **~375 TFLOPS**) validated the pattern **before** **8192³**, de-risking a **large** rewrite at the most expensive problem size.

## 5. Persistent vs. non-persistent launch

**Concept:** [Chapter 7](../../tutorial/ch07-persistent.md) — fixed **CTA** count (e.g., one per SM) with a **software tile loop** to cover the output grid; reduces **tail wave** waste when **`total_tiles >> SMs`**.

**Pattern in this study:** **Non-persistent** won at **8192³** for the best kernel. The README states plainly: **non-persistent > persistent** here because **wave quantization** at that size was **acceptable**, while persistent’s extra **control-flow and striping** did not pay for itself relative to the **occupancy-optimal** split-output tile.

This is not a verdict against persistence globally—it is **problem- and kernel-specific**. The right mental model from [Ch7](../../tutorial/ch07-persistent.md): persistent fixes **grid-level** underuse; if **inner-block** SMEM or **pipeline** choices already cap throughput, persistence cannot recover what **occupancy** lost.

### When to try persistent anyway

If **measured** **TFLOPS** at **8192³** is high but **smaller** **S** shows **tail** loss (many idle SMs on the final wave), **`NUM_SMS`** striping from [Ch7](../../tutorial/ch07-persistent.md) is the next experiment. In *this* tune, **inner-block** improvements were so large that **non-persistent** **382.5 TFLOPS** won—**wave quantization** did not dominate anymore.

## 6. Compiler flags and lowering quality

The shipped builds share a common flag bundle:

- **`--use-warpspec`** — enable the warp-specialized lowering path aligned with [Ch6](../../tutorial/ch06-warpspec.md).
- **`--stmatrix`** — use **STSM**-style shared-memory matrix setup where legal, improving the path from staged **`f16`** to **WGMMA** operands.
- **`--hoist-offset`**, **`--hoist-scale`** — hoist address arithmetic and scale factors so inner loops stay **thin**.
- **`--ptx-barrier`** — emit **barrier** instructions compatible with **async** producer/consumer synchronization.
- **`--tma-cluster-aware`** — bias TMA lowering for **cluster** and **multicast** realities on SM90 systems.
- **`--wgmma-wait-depth=N`** (added during Phase 3) — expose **pipeline wait depth** for WGMMA as a **tunable** to match **stage count and issue rate**.

**iter023** (+**ptx-barrier**, +**stmatrix**, +subspan) showed **+5%** at **2048³** on top of tile tweaks—evidence that **lowering** and **addressing** matter after the **tileflow** is sane.

### Flag interaction notes

- **`--tma-cluster-aware`** is most relevant when building for **SM90** systems that may use **TMA multicast** or **cluster** scheduling; it should stay **on** for apples-to-apples **Hopper** runs in this study.
- **`--wgmma-wait-depth=N`** appeared in **Phase 3** specifically because **WN sweeps** changed **issue** and **stage** timing; treat **`N`** as part of the **same** sweep matrix as **STAGES**, not as a polish pass.

### Experiment hygiene

When comparing **iterations**, freeze **everything** except the **one** knob under test: same **`CHOREO_TIMING_*`**, same **verification**, same **arch** (`sm_90a`). The README’s tables are trustworthy because the harness defaults stayed stable across **iter001–065**.

## 7. Narrative arc: bottleneck to pattern

1. **Baseline (208.7 TFLOPS @ 8192³)** — correct **roles** (1p1c) but **stage count / WN** mismatched to **occupancy** and **problem scale**.
2. **Phase 1** — reduce **SMEM** or improve **lowering** → modest but real gains (**214.3 TFLOPS** @ **2048³**).
3. **Phase 2** — **jointly** tune **WN** and **STAGES**, then move to **1p2c split-output** → **large** jumps (**354.1** @ **2048³**, **~375** @ **4096³**, **382.5** @ **8192³**).
4. **Phase 3** — **WN sweep** at **8192³** with **K-unroll** and **`wgmma-wait-depth`** → **380.6 TFLOPS** (iter061) and discovery of the **WN=168** **occupancy cliff**.

## 8. What we did *not* need to change

The study stayed inside **dense FP16 GEMM** with **TMA**-staged operands and **WGMMA** accumulation—no **mixed precision**, no **split-K** across CTAs, no **CUDA Graph** capture. Those are valid next levers, but they were **out of scope** relative to the **+83%** already available from **tileflow** and **compiler** flags documented here.

## 9. Worked example: reading iter046 → iter048 as profiling

Start from **iter046**: **242 TFLOPS** @ **2048³**, **WN=176**, **STAGES=2**. Occupancy is **healthy enough** that **WGMMA** is fed, but **latency** along **K** is not fully hidden—**TMA** completion and **consumer** **commit** points still leave **gaps**.

**Hypothesis:** add **one** more **operand** stage to lengthen the **producer** run-ahead without blowing **228 KB**.

**Change:** **iter048** sets **STAGES=3** with the **same** **WN=176**.

**Result:** **354.1 TFLOPS**—far more than **linear** scaling from **stage count** alone. That is the signature of a **bubble-limited** schedule: the extra **stage** buys **concurrency** between **TMA** and **WGMMA**, not “more FLOPs.”

**Follow-up:** at **8192³**, the **same** **3-stage** idea **hurts** because **SMEM** competes with **grid** parallelism. The **profiling** conclusion flips: **latency hiding** is no longer the **binding** constraint—**occupancy** is.

## 10. Worked example: split-output at 4096³ before 8192³

**iter050** (**~375 TFLOPS** @ **4096³**) is a **deliberate midpoint**. **4096³** has **64×** the work of **2048³** but **only half** the **K** depth of **8192³** relative to cache and wave structure. If **split-output** had regressed here, we would **not** have trusted **1p2c** at **8192³**.

The measured **~375 TFLOPS** validated **separate** **SMEM** output slices for **two** consumer warpgroups before committing to **iter057**’s **non-persistent** **8192³** configuration.

## 11. Numeric summary (all quoted TFLOPS)

| Kernel / phase | TFLOPS | Cube | Configuration (high level) |
|----------------|--------|------|----------------------------|
| Phase 1 baseline | 204 | 2048³ | 1p1c WN=128 STAGES=4 |
| iter004 | 208.9 | 2048³ | WN=256 STAGES=2 |
| iter023 | 214.3 | 2048³ | +ptx-barrier +stmatrix +subspan |
| iter046 | 242 | 2048³ | WN=176 STAGES=2 |
| iter048 | 354.1 | 2048³ | WN=176 STAGES=3 |
| iter050 | ~375 | 4096³ | 1p2c split WN=128 STAGES=2 |
| Main baseline | 208.7 | 8192³ | 1p1c WN=128 STAGES=4 |
| iter057 | 382.5 | 8192³ | 1p2c split WN=152 non-persistent |
| iter061 | 380.6 | 8192³ | 1p2c split WN=160 K-unroll |

Use this table as a **single-page** answer to “what changed between milestones?”

## 12. Multi-warpgroup preview (Chapter 8)

[Chapter 8: Multi-warpgroup](../../tutorial/ch08-multi-warpgroup.md) broadens the **parallel** story beyond **1p1c** / **1p2c** splits inside one **GEMM** block. This case study’s **1p2c** is **not** a full tour of that chapter—it is the **minimal** extension of [Ch6](../../tutorial/ch06-warpspec.md) needed to **remove output contention**. If you find yourself adding **third** roles (**1p2c** + **separate epilogue**, etc.), read **Ch8** next so **barrier** and **masking** rules stay coherent.

## 13. View and slicing vocabulary (Chapter 9)

[Chapter 9](../../tutorial/ch09-view-from.md) documents **views**, **subspan**, and **chunkat**—the same vocabulary **iter023** exercised when **`subspan`** changes landed alongside **`stmatrix`**. When **TFLOPS** move without **WN/STAGES** edits, audit **views** first: an off-by-one **subspan** can still **verify** on small **S** yet **mis-align** **TMA** for large **S**.

## 14. Decision tree (apply in order)

1. **Establish** **TFLOPS** at **2048³** and **8192³** for the **same** `.co` header.
2. If **8192³** is **much lower** than **2048³** in **efficiency** (TFLOPS per SM as a rough proxy), suspect **waves** or **occupancy**—print **SMEM** and **CTAs/SM**.
3. If **occupancy ≥ 2** but **TFLOPS** still low, profile **pipeline** depth: try **±1** **STAGES** at fixed **WN**.
4. If **STAGES** optimum differs across **2048³** vs. **8192³**, accept **size-specific** tuning or adopt **split-output** / **different** **WN** for the large cube only.
5. If **1p1c** plateaus at **large** **WN**, prototype **1p2c split-output** at **4096³** before **8192³**.
6. Only then sweep **compiler** flags and **`wgmma-wait-depth`**—these are **last-mile** levers once **footprint** is sane.

## 15. Risk register (what went wrong in the log)

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **SMEM cliff** | **TFLOPS** collapse after small **WN** bump | Precompute **bytes/block** before build |
| **Stage mismatch across sizes** | **2048³** great, **8192³** bad | Re-sweep **STAGES** per **S** |
| **Output contention** | **1p1c** caps out as **WN** grows | **1p2c split-output** |
| **Persistent overhead** | **Inner** wins negated by **striping** | Try **non-persistent** at target **S** |

## 16. Glossary (this case study)

- **1p1c** — one **TMA** producer warpgroup, one **WGMMA** consumer warpgroup ([Ch6](../../tutorial/ch06-warpspec.md)).
- **1p2c** — one producer, **two** consumers; here with **split** **output** buffers.
- **WN** — `MATMUL_WARP_N`, **N** extent of the **WGMMA** tile.
- **STAGES** — operand **ring** depth for **K** (**pipeline** slots).
- **Split-output** — **separate** **shared** **accumulator** tiles per consumer to avoid **writer** contention.
- **Non-persistent** — classic **grid** over **`block_m, block_n`** rather than [Ch7](../../tutorial/ch07-persistent.md) **striping** (in the winning **8192³** kernel).

## 17. If you only change one thing

For **8192³** on **H800**-class **Hopper**, the **largest** single structural win in this log was **not** a flag—it was **1p2c split-output** moving **TFLOPS** into the **370–382** band before the **WN** sweep fine-tuned **occupancy**. Flags like **`--stmatrix`** matter, but they **cannot** recover **serialization** on **`output_s`** if **two** consumers must **share** one **accumulator** tile.

## 18. Sibling tutorials (optimization index)

After this case, read **[Sparse GEMM](../gemm-sp/index.md)** and **[Block-scaled GEMM](../blockscale-gemm/index.md)** on the same [optimization index](../index.md). Those add **metadata** and **scale** tensors; the **SMEM** and **stage** arithmetic become **strictly harder**, so the **dense** story here is the **foundation** for **why** every **extra** buffer must **pay** for itself in **measured** **TFLOPS**.

## 19. Compiler vs. tileflow ownership

**Tileflow** owns **`MATMUL_*`**, **`parallel`**, **`inthreads.async`**, and **shared** **layouts**—these decide **correctness** and **occupancy class**. The **compiler** owns **instruction selection** (**`stmatrix`**, **barriers**, **TMA** lowering, **`wgmma-wait-depth`**). A common failure mode is **over-tuning** flags while **SMEM** is still on the **wrong** side of **228 KB**; the **log** avoids that by **freezing** flags while sweeping **WN/STAGES**, then **unfreezing** only after **split-output** lands.

## 20. Afterword: what “Choreo-specific” means here

Nothing in the **208.7 → 382.5** story **requires** Choreo magic—**TMA**, **WGMMA**, **warps**, and **shared** **rings** are **Hopper** facts. Choreo’s value is **tileflow clarity**: you can **see** **full/empty**, **stages**, and **roles** in one file and **diff** iterations meaningfully. When you port ideas to raw **CUDA** or **CUTLASS**, keep the **same** **measurement** discipline: **TFLOPS** at **fixed** **S**, **SMEM** tables, **occupancy** checks.

## 21. FAQ-style pitfalls

*Should I copy iter048’s STAGES=3 into my 8192³ kernel?*  
Not automatically—this log says three stages help **2048³** but hurt **8192³**. Re-measure on your **S**.

*Is WN=160 always optimal?*  
It is occupancy-optimal in the Phase 3 sweep for split-output **1p2c** on this SKU; other GPUs or cluster configs can move the cliff.

*Are compiler flags optional?*  
For a fair comparison to the README, keep the flag bundle identical; for research, toggle one flag at a time after tileflow is stable.

## Suggested reading order

- [Ch3: Pipelining](../../tutorial/ch03-pipeline.md) for **stage** intuition.
- [Ch6: Warp specialization](../../tutorial/ch06-warpspec.md) for **1p1c / 1p2c** roles.
- [Ch7: Persistent kernels](../../tutorial/ch07-persistent.md) for when **grid-level** launch should change.

Next: [AI-tune last mile](aitune-last-mile.md) for **repro commands** and **shipped kernel** pointers.
