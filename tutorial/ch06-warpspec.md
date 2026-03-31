# Async Pipelining: inthreads, Events, and Warp Roles

Chapter 3 showed how to **overlap** memory traffic with compute using **double buffering**: two named slots in shared memory, a prologue, a steady-state loop with **`swap`**, and an epilogue. That pattern hides a lot of latency, and it is still the right mental model for **which buffers exist** and **when it is safe to overwrite** them.

What it does *not* do is split **who** does the work. In the pipelined matmul from that chapter, the **same threads** that eventually multiply are also responsible for issuing DMA copies, waiting on futures, and swapping buffers. The schedule interleaves producer and consumer steps on one timeline. That is good engineering — but on **Hopper-class** hardware you can often do better.

This chapter introduces **warp specialization**: different **warpgroups** inside the same CUDA block adopt **different roles**. One group becomes a **producer** that keeps **TMA** loads in flight; another becomes a **consumer** that runs **WGMMA** on data that is already landing in shared memory. Because those groups execute **concurrently**, you get **true overlap**: loads for stage *s+1* can advance while math still consumes stage *s*, instead of forcing a single thread set to alternate.

You already know **TMA** and **swizzle** from Chapter 4, and **WGMMA** with **`: group-4`** from Chapter 5. Here we wire them together with **shared events**, **`inthreads.async`**, **async TMA completion signals**, and **`mma.commit`** — the vocabulary Choreo uses for a **1 producer + 1 consumer (1P1C)** pipeline. The running example is the tileflow portion of **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** in the Choreo benchmark tree, with the usual header constants:

- **`MATMUL_WARP_M = 64`**, **`MATMUL_WARP_N = 128`** — warpgroup output tile (WGMMA footprint along M and N).
- **`MATMUL_TILE_K = 64`** — K depth of each staged slab in shared memory per pipeline stage.
- **`MATMUL_WARP_K = 16`** — K step per inner **`mma.row.row`** (hardware slice along K).
- **`MATMUL_SWIZ = 128`** — swizzle metadata for **TMA** and **`mma.load.swiz`** (here **`2 * MATMUL_TILE_K`**, as the benchmark asserts).
- **`MATMUL_STAGES = 4`** — number of **ring-buffer** slots in shared memory (producer fills, consumer drains).

If any of those numbers feel arbitrary, treat them as **tuning knobs** you would sweep in a real project; the **pattern** — events, stages, and split roles — is what transfers to other kernels.

## From swap-based double buffering to split roles

In Chapter 3, **double buffering** meant: while you compute on **`lf0` / `rf0`**, you prefetch into **`lf1` / `rf1`**, then **swap** names so the “current” and “next” buffers exchange meaning. Correctness came from never reading a slot that was simultaneously being overwritten.

**Warp specialization** keeps the same **safety story** — each **stage** is either **empty** (safe for the producer to write) or **full** (safe for the consumer to read) — but it **assigns** those actions to **different warpgroups**. The producer never executes **`mma.row.row`**; the consumer never issues **`tma.copy`**. They coordinate only through **events** and the **shared** arrays they agree on.

On **NVIDIA Hopper**, a cooperative thread array can organize **multiple warpgroups**; **WGMMA** already required **four warps** (**`: group-4`**) in Chapter 5. The 1P1C kernel goes one step further: it launches **two** such groups inside the block (**`parallel p1 by 2 : group-4`**) and uses **`p1`** as a **role selector**. That is not a runtime “if” scattered through unrelated code — it is **`inthreads.async (condition)`**, which tells Choreo which warpgroup runs which choreographed region.

## New syntax you will see

The kernel below uses several constructs together. Skim this list now; we will unpack each in context.

1. **`inthreads.async (condition) { ... }`** — Within a **`parallel`** group, only threads (or warpgroups) for which **`condition`** holds execute the block. Here **`condition`** is **`p1 == 0`** or **`p1 == 1`**: **role assignment** for warp specialization.
2. **`shared event full[N], empty[N]`** — Arrays of **named synchronization events** in **shared** scope, one **`full`** and one **`empty`** per pipeline **stage**.
3. **`wait event_name`** — Block until another participant **signals** that event (consumer waits on **`full`**, producer waits on **`empty`**).
4. **`trigger event_name`** — Signal an event (release a waiter).
5. **`tma.copy.async<event>.swiz<N>`** — Issue **asynchronous TMA**; when the transfer completes, the hardware **signals** the tagged **`event`** (here **`full[stage]`**). The **`.swiz<N>`** piece matches Chapter 4’s layout contract.
6. **`mma.commit`** — **Commit** accumulated WGMMA results for the current **pipeline step**, establishing a fence between **stages** so shared memory can be reused safely and accumulator state stays well-defined.
7. **`stage = iv_k % MATMUL_STAGES`** — **Circular buffer indexing**: map the linear K-tile index **`iv_k`** to a physical slot **`0 .. STAGES-1`**.
8. **Predicates like `p1 == 0`** — These appear **inside** the **`inthreads.async (...)`** header, not as a separate **`if`** around the whole warpgroup body: Choreo uses the predicate to **specialize** which warpgroup owns which loop nest.

## Shared memory and the event arrays

Before the parallel roles split, the block declares **staging** for **all** pipeline slots at once:

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
shared f16 [MATMUL_STAGES * MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
shared f16 [MATMUL_STAGES * MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
```

**`lhs_load_s`** and **`rhs_load_s`** are not single tiles anymore. They are **`MATMUL_STAGES`** copies of the **same logical shape** — a **ring** of **A** and **B** slabs. Index **`stage`** (derived from **`iv_k % MATMUL_STAGES`**) picks **which row-of-slots** in that ring is active for this iteration.

**`full[s]`** means “stage **`s`** has been **filled** by TMA and is ready for WGMMA.” **`empty[s]`** means “the consumer has **finished** with stage **`s`**; the producer may **overwrite** it.” Those names are conventional: you could rename them in a scratch kernel, but **full/empty** reads well in documentation and matches many production pipelines.

**`output_s`** stays a **single** warpgroup-sized tile — the consumer accumulates the full **K** sweep into **`mc`**, then **`mma.store`** once into **`output_s`**, then **`tma.copy`** to global. Only the **operand** staging is multi-buffered; the **result** tile does not need a ring unless you split writers (not this kernel).

## Full tileflow: 1P1C matmul

The excerpt below is the **`__co__ void matmul`** body from **`matmul_f16_dyn_sm90_warpspec_1p1c.co`**, trimmed to the tileflow (no host harness). Constants are as in the benchmark header.

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
    shared f16 [MATMUL_STAGES * MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_STAGES * MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait empty[stage];
          tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
          tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
          trigger full[stage];
        }
      }

      inthreads.async (p1 == 1) {
        mc = mma.fill.f16 0.0f;
        foreach {s} in [MATMUL_STAGES] {
          trigger empty[s];
        }
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait full[stage];
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
          mma.commit;
          trigger empty[stage];
        }
        mma.store mc, output_s;
        tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

Read the outer **`parallel {block_m, block_n} … : block`** as before: one CUDA block per **`MATMUL_WARP_M × MATMUL_WARP_N`** output tile. Inside, **`parallel p1 by 2 : group-4`** creates **two** warpgroups distinguished by **`p1 ∈ {0, 1}`**. Each **`inthreads.async`** body is a **different program** for a **different** warpgroup.

## Producer warpgroup: wait empty, async TMA, trigger full

The **`p1 == 0`** region is the **producer**:

```choreo
inthreads.async (p1 == 0) {
  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    stage = iv_k % MATMUL_STAGES;
    wait empty[stage];
    tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
    tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
    trigger full[stage];
  }
}
```

For each K-tile index **`iv_k`**:

1. **`stage = iv_k % MATMUL_STAGES`** picks a **ring slot**. After **`MATMUL_STAGES`** iterations along **`iv_k`**, you wrap around and **reuse** the same shared-memory slice — exactly like double buffering generalized to **N** slots.
2. **`wait empty[stage]`** ensures the **consumer** has **released** this slot. Until **`empty[stage]`** fires, the producer must not issue TMA into **`lhs_load_s` / `rhs_load_s`** at that **`stage`** — otherwise you would corrupt data the consumer still treats as valid.
3. The two **`tma.copy.async<full[stage]>.swiz<…>`** lines start **asynchronous** loads from **global** into the **`stage`** slice of shared memory. The **`<full[stage]>`** annotation ties **completion** of those copies to the **`full[stage]`** event: when TMA finishes, **`full[stage]`** becomes eligible to wake waiters (the exact micro-model is target-specific, but the **Choreo contract** is “signal **`full`** when this async transfer is done”).
4. **`trigger full[stage]`** completes the **handshake** from the producer’s point of view for this iteration — in this kernel it appears **after** the async copies are issued; paired with the **`<full[stage]>`** completion semantics, it yields a clear **“stage is ready for math”** story for the consumer.

Notice the producer **never** touches **`mc`**, **`mma.load`**, or **`mma.row.row`**. Its entire job is to **keep the ring full ahead** of the consumer, modulo **`MATMUL_STAGES`**.

## Consumer warpgroup: bootstrap empty, wait full, compute, commit, release

The **`p1 == 1`** region is the **consumer**:

```choreo
inthreads.async (p1 == 1) {
  mc = mma.fill.f16 0.0f;
  foreach {s} in [MATMUL_STAGES] {
    trigger empty[s];
  }
  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    stage = iv_k % MATMUL_STAGES;
    wait full[stage];
    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
      mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
      mma.row.row mc, ma, mb;
    }
    mma.commit;
    trigger empty[stage];
  }
  mma.store mc, output_s;
  tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

Three segments matter.

**Bootstrap.** Before the main K loop, the consumer runs:

```choreo
foreach {s} in [MATMUL_STAGES] {
  trigger empty[s];
}
```

At the start, **every** slot is logically **empty** — no valid operand data yet — but the **producer** is blocked on **`wait empty[stage]`** until those **`empty`** events exist. The consumer **pre-triggers** all **`empty[s]`** so the **first** wave of TMA loads can enter the ring without deadlock. Think of it as **initializing the credit counter** for each buffer slot.

**Steady state.** For each **`iv_k`**:

- **`wait full[stage]`** stalls until the producer’s TMA (and associated signaling) indicates that **`stage`** holds a consistent **A/B** slab for this K tile.
- The inner **`foreach {iv_warp}`** walks **`MATMUL_TILE_K / MATMUL_WARP_K`** steps (here **64/16 = 4**) of **WGMMA**, loading swizzled tiles with **`mma.load.swiz<MATMUL_SWIZ>`** and accumulating with **`mma.row.row`** — same **Chapter 5** story, but reading from **`at(stage, 0)`** instead of a single buffer.
- **`mma.commit`** ends the **pipeline stage** from the **accumulator’s** point of view. **WGMMA** pipelines deeply; **`commit`** is the Choreo-level **fence** that says “the partial products from this **stage’s** operand loads are **integrated** into **`mc`** before we advance to reusing shared memory for another **logical** K step.” Without that boundary, you risk **semantic overlap** between what the hardware still considers “in flight” for the previous stage and what the next **`wait full`** / **`mma.load`** assumes about **shared** contents.
- **`trigger empty[stage]`** hands the slot **back** to the producer: “I am done reading **`stage`**; you may overwrite it on a future **`iv_k`** when the ring wraps.”

**Epilogue.** After all K tiles, **`mma.store`** writes **`mc`** to **`output_s`**, and **`tma.copy`** pushes the result tile to **global** — analogous to the non-specialized Hopper kernel, but the **K** loop lived entirely inside the **specialized consumer**.

### Inner K: four WGMMA steps per staged slab

With **`MATMUL_TILE_K = 64`** and **`MATMUL_WARP_K = 16`**, the inner loop

```choreo
foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
```

iterates **four** times per **`stage`**. Each pass loads a **K = 16** slice via **`chunkat(_, iv_warp)`** on both staged tensors, then **`mma.row.row mc, ma, mb`**. All four passes **accumulate into the same** **`mc`** for this **iv_k** before **`mma.commit`** ends the **stage** from the hardware’s point of view.

If you change **`MATMUL_TILE_K`**, keep **`cdiv(MATMUL_TILE_K, MATMUL_WARP_K)`** integral and respect the benchmark’s swizzle rule (**`MATMUL_SWIZ == 2 * MATMUL_TILE_K`** for this FP16 path) so **TMA** and **`mma.load.swiz`** stay matched.

## Why four stages?

**`MATMUL_STAGES = 4`** is a **latency hiding** knob. Each **stage** is one **A** slab and one **B** slab in shared memory. With **more** stages, the producer can **run ahead** further while the consumer chews through older slots — up to the point where **shared memory** or **event bookkeeping** stops paying off.

The **`%`** operator is what makes a **fixed** number of physical slots behave like an **unbounded** logical timeline of K tiles. If **`cdiv(K, MATMUL_TILE_K)`** is larger than **`MATMUL_STAGES`**, you **reuse** slot 0 only after it has been **emptied** again — the **`wait empty` / `trigger empty`** pair is what makes that reuse **safe**, not the modulo arithmetic alone.

**Fewer stages** (for example **2**) shrink **shared** footprint and simplify bookkeeping but give the producer **less** slack to hide **TMA** latency; **more stages** increase **run-ahead** until **capacity** or **returns** flatten out. **`MATMUL_STAGES = 4`** is a typical hand-tuned compromise; an auto-tuner might treat **stage count** as a discrete search axis next to **tile K**.

## Handshake timeline for one logical K tile

Walking **one** **`iv_k`** as a dialogue between roles clarifies the **credit** story:

1. **Consumer** has already signaled **`empty[stage]`** (bootstrap or previous lap), so the **producer** may **`wait empty[stage]`** and proceed.
2. **Producer** launches **async TMA** into **`lhs_load_s.at(stage, …)`** and **`rhs_load_s.at(stage, …)`**, tied to **`full[stage]`**, and **`trigger full[stage]`** participates in waking the consumer.
3. **Consumer** passes **`wait full[stage]`**, runs **four** **`mma.row.row`** steps on that **stage**, **`mma.commit`**, then **`trigger empty[stage]`**.
4. When **`iv_k`** advances enough that **`iv_k % MATMUL_STAGES`** equals this **stage** again, the cycle repeats — the **ring** has wrapped.

This mirrors **producer–consumer queues** on CPUs: **`empty`** credits are **spent** when the producer claims a slot and **replenished** when the consumer finishes reading; **`full`** credits mean **data is ready**. The modulo index only picks **which physical slot** participates in that exchange.

## `mma.commit` and pipeline boundaries

On **WGMMA** paths the hardware **overlaps** operand fetch, issue, and accumulation across **time** more aggressively than a textbook **“one `mma` at a time”** picture suggests. **`mma.commit`** is Choreo’s way to mark a **logical boundary** after you have finished the **`mma.row.row`** sequence for **one** **staged K slab** (**one** **`iv_k`** worth of operand data in **shared** at **`stage`**).

Intuitively, you want two properties before **`trigger empty[stage]`** releases shared memory back to **TMA**:

1. **Accumulator coherence:** Partial sums from **this** **stage’s** operand loads are **fully folded** into **`mc`** so the **next** **`wait full`** / **`mma.load`** round does not race with **in-flight** **WGMMA** micro-operations from the **previous** use of overlapping register state.
2. **Shared reuse safety:** The **consumer** asserts it will not **read** **`lhs_load_s` / `rhs_load_s`** at **`stage`** again for the **completed** **iv_k** once **`empty`** fires — the **producer** may then **overwrite** that slice.

Exact **ISA** mapping is **target-dependent**; as a **reader** of **`.co`** files, treat **`mma.commit`** as **mandatory glue** between **“done with this stage’s math”** and **“signal empty.”** Removing it in experiments is a **stress test**, not a **performance knob**, until you **prove** otherwise on your **chip** and **toolchain** revision.

## Single-role Hopper matmul vs 1P1C (same primitives, different schedule)

| Aspect | **`matmul_f16_dyn_sm90.co`** (Chapter 5 excerpt) | **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** (this chapter) |
|--------|---------------------------------------------------|-----------------------------------------------------------|
| **Warpgroups in the block** | One **`parallel … : group-4`** body for load + MMA | **`parallel p1 by 2 : group-4`** — **two** warpgroups, **`p1`** selects **role** |
| **Who issues TMA** | Same warpgroup that **WGMMA**-computes | Only **`p1 == 0`** |
| **Who runs `mma.row.row`** | Same warpgroup after **sync** TMA | Only **`p1 == 1`** |
| **Operand buffering** | One **lhs** / **rhs** slab per **block** (per K iter in excerpt) | **`MATMUL_STAGES`** **ring** of slabs in **shared** |
| **Synchronization** | **Implicit** / **wait**-style pairing with copies (in full benchmarks) | Explicit **`shared event`** arrays **`full[]`**, **`empty[]`** |
| **Overlap model** | **Interleaved** steps on **one** warpgroup | **Concurrent** **producer** and **consumer** warpgroups |

Neither layout changes the **math**; both still implement **blocked GEMM** with **TMA** + **WGMMA**. Warp specialization changes **throughput** by changing **who** can be **busy** at the same **clock** cycle.

## Role predicates: `inthreads.async (p1 == 0)` vs a plain `if`

You might wonder whether **`if (p1 == 0) { … }`** inside a shared warpgroup body could **specialize** roles. The tutorial kernel uses **`inthreads.async (condition)`** so **specialization** is part of the **parallel** **structure**: Choreo **partitions** the **warpgroups** so each **body** describes a **complete** **program** for **one** **role**, not a **divergent** branch **every** instruction. That keeps **tileflow** **readable** — **two** **straight-line** **loops** instead of **one** **loop** **full** of **predication**.

Different **compiler** versions may accept **related** spellings; when **reading** **repositories**, treat **`inthreads.async (…)`** as the **idiomatic** **warp-specialization** **wrapper** for **conditional** **async** **regions** inside **`parallel`**.

## How this relates to Chapter 3’s `swap`

In the **`swap(lf0, lf1)`** world, **two** names alias **two** buffers, and you **exchange** them each iteration. In the **event** world, **names are fixed** (**`lhs_load_s.at(stage, …)`**) and you **cycle `stage`**. **`full` / `empty`** play the role of **futures** or **barriers** that tell you **which side** of the analogy is “current” for each participant.

The big difference is **concurrency**: **swap** pipelining on one thread set is **interleaved**; **1P1C** pipelining is **parallel** warpgroups. Your profiler may show **TMA** and **WGMMA** activity **overlapping in time** inside the same block — that is the payoff.

## Practical reading order inside the file

When you open **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** in the repository, start at the **`__co__ void matmul`** comment block: it literally labels **producer** versus **consumer**. Then trace **one** value of **`iv_k`** through both **`inthreads.async`** bodies — **`stage`**, **`wait`**, **`tma.copy.async`**, **`wait full`**, inner **`mma`**, **`mma.commit`**, **`trigger empty`**. Once one lap is clear, imagine **`iv_k+1`** starting while the other warpgroup is still finishing **WGMMA** on the previous slot: that is the **pipeline** you built.

## Pitfalls that cost correctness

- **Deadlock:** If you remove the **initial `trigger empty[s]`** loop, the producer’s first **`wait empty`** may wait **forever**.
- **Undersynchronized reuse:** If you drop **`mma.commit`** or mis-order **`trigger empty`** relative to **loads**, you can read **stale** swizzled data or **lose** accumulator coherence — treat **`commit`** as part of the **contract** between **WGMMA stages**, not as optional noise.
- **Swizzle mismatch:** **`MATMUL_SWIZ`** must stay consistent with **`tma.copy…swiz`** and **`mma.load.swiz`** (Chapter 4 + 5). Warp specialization does not relax layout rules.
- **Role drift:** Both **`inthreads.async`** bodies must agree on **`stage`** indexing and buffer shapes. A typo in **`.at(stage, 0)`** on **one** side shows up as **silent** wrong answers or rare races.
- **Too few stages for your latencies:** If **TMA** often completes **after** the consumer needs the next **`full`**, you **stall**; raising **`MATMUL_STAGES`** (when shared memory allows) increases producer **run-ahead**.
- **Mismatched trip counts:** Producer and consumer both use **`foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)]`**. If one loop were shorter, you would **leak** or **orphan** events — keep **K** iteration **symmetric** across roles.

## Debugging and verification

The benchmark’s **`main`** prints timing, **TFLOPS**, and runs a **host** numerical spot-check on **C**. When a pipeline edit breaks correctness, re-check **event order** before you suspect **MMA** layout: specialization bugs are usually **synchronization** bugs.

The source file honors **`CHOREO_SKIP_VERIFY`** and related env vars for faster iteration; return to **full verify** before trusting results.

## Events versus copy futures

Chapter 3’s **DMA futures** and Chapter 5’s **`wait`** on async copy handles tie **completion** to **specific operations**. **Shared events** are **named barriers** any cooperating participant can **`wait`** or **`trigger`**. Here, **`full`** / **`empty`** describe **buffer lifecycle**, while **`tma.copy.async<full[stage]>`** still connects **hardware completion** to **`full[stage]`**. See **`documentation/events.md`** for more event idioms; this chapter stays tied to **matmul** so you can **`grep`** **`inthreads.async`** for working examples.

## Summary

Warp specialization lets **one warpgroup** feed **TMA** while another **consumes** with **WGMMA**, using **shared events** instead of a single thread set that **swaps** buffers. The **1P1C** pattern pairs **`wait empty` → async TMA → `trigger full`** with **`wait full` → MMA → `mma.commit` → `trigger empty`**, and **`iv_k % MATMUL_STAGES`** maps a long K sweep onto a **small ring** of shared-memory slots. **`mma.commit`** marks the end of a **consumer stage** so accumulators and shared reuse stay aligned with the hardware pipeline.

**Key syntax introduced**

- **`inthreads.async (condition) { … }`** — **Conditional** execution inside a **`parallel`** region; use **`p1 == 0`** / **`p1 == 1`** (or similar) for **warp/warpgroup roles**.
- **`shared event full[N], empty[N]`** — **Per-stage** synchronization handles in **shared** scope.
- **`wait event`** / **`trigger event`** — **Block** until signaled / **signal** waiters.
- **`tma.copy.async<event>.swiz<N>`** — **Asynchronous TMA** with **completion** tied to **`event`** and **swizzle** metadata **`N`**.
- **`mma.commit`** — **Fence** between **WGMMA pipeline stages** after a **stage’s** **`mma.row.row`** sequence.
- **`stage = iv_k % MATMUL_STAGES`** — **Ring index** for **multi-buffered** operand tiles.

**Further reading in-tree**

- `choreo/benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co` — full **1P1C** kernel, timing harness, and verification.
- `croktile-tutorial/tutorial/ch03-pipeline.md` — **swap**-based double buffering baseline.
- `croktile-tutorial/documentation/events.md` — additional event-oriented patterns and semantics.

### Where to go next

Chapter 7 (**persistent** kernels) and Chapter 8 (**multi-warpgroup** scaling) build on the same **tileflow** vocabulary: you already know how to **fill**, **compute**, and **store**; warp specialization adds **who** issues each phase and **how** **`full` / `empty`** credit flows around the ring. When you read more **`.co`** benchmarks, look for **`inthreads.async`** and **`shared event`** — they are the telltales for **async pipelining** beyond single-role warpgroups.

### Suggested experiments

- Compare **wall-clock** and **TFLOPS** between **`matmul_f16_dyn_sm90.co`** and **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** on the same **M, N, K** after normalizing timing flags — expect wins to depend on **problem size**, **memory bandwidth**, and **stage count**, not a universal speedup.
- In a **scratch copy** of the warpspec file, temporarily set **`MATMUL_STAGES`** to **2** and observe whether **latency hiding** improves or **stalls** increase; watch **shared memory** usage in the occupancy report if you have one.
- Trace **deadlock** deliberately (comment out the **`trigger empty`** prologue in a **throwaway** branch) to cement the **credit** intuition — then restore it.

### Closing thought

Choreo is still **orchestration**: which warpgroup **waits** on which **event**, which **async TMA** fills which **shared** slice, and where **`mma.commit`** closes a **WGMMA** stage. The **1P1C** kernel is a small, readable **Hopper** reference — **TMA** and **WGMMA** remain the primitives from Chapters 4 and 5; **roles** and **events** are the extra degrees of freedom that let **memory** and **math** overlap **across warpgroups** instead of **time-slicing** on one program counter.

When **`stage`**, **`full`**, and **`empty`** feel automatic, you are in good shape for Chapter 7 (**persistent** kernels) and Chapter 8 (**multi-warpgroup** accumulators), where the same vocabulary composes into larger launch strategies.
