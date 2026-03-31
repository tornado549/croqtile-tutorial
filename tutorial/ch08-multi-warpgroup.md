# Multi-Warpgroup Matmul: Scaling with Multiple Accumulators

Chapter 7 introduced **persistent** matmul kernels: one CTA can sweep many output tiles with `step` and guards, amortizing launch overhead and improving occupancy on large problems. This chapter stays inside a **single** CTA but asks a different scaling question: **how do you make one block responsible for a wider slice of M than a single warpgroup can natively cover?**

On Hopper-class hardware, **WGMMA** (warp-group matrix multiply-accumulate) typically cooperates across **four warps** — 128 threads — per **warpgroup**. One such group naturally “owns” a **64×64** FP16 result tile in the configuration we use here (`MATMUL_WARP_M = 64`, `MATMUL_WARP_N = 64`). That is a comfortable, hardware-aligned unit. If you stop there, each CUDA block’s footprint along M is also 64 rows unless you add more warpgroups.

**Multi-warpgroup** kernels raise the **block tile height** along M — for example to **128** rows — by running **two** warpgroups in parallel inside the same block. Each warpgroup keeps its **own** FP32 accumulator `mc`, loads **different** rows of the left-hand matrix **A** from shared memory, but reuses the **same** **B** tile already sitting in shared. After the K loop, each group stores **its** 64×64 piece of the result into a shared staging buffer; one TMA (or wide copy) then writes the full **128×64** tile to global memory.

The payoff is familiar from tiling theory: **larger per-block tiles** mean **fewer blocks** for the same matrix dimensions, which reduces **grid launch overhead**, improves **data reuse** of **B** within the block, and often helps **SM utilization** — provided the larger shared-memory and register footprint still fits your target.

You might ask: **why not** keep **one** warpgroup per block and simply use **more** blocks along M? That design is valid and often easier to reason about. Multi-warpgroup layouts earn their keep when **B** reuse across the **M** split is strong: **one** TMA brings **B** into shared, and **two** groups multiply **different** row bundles of **A** against **that same** **B** without a second global read of **B** for the second **M** stripe. You still pay for **A** at the **full** **128×K** footprint per **K** step, but you **halve** the **B** traffic **per result row bundle** in the sense that two **64×64** result patches share one **B** slab. Whether that wins over a simpler kernel is **empirical**; the Choreo pattern exists so you can express the layout cleanly and let the compiler target WGMMA.

## One warpgroup vs. two: the M axis

Fix these sizes in your head (they match `matmul_f16_dynamic_mwg_impl_1.co` in the Choreo benchmarks):

| Symbol | Value | Role |
|--------|-------|------|
| `MATMUL_WARP_M` | 64 | Rows of C one warpgroup contributes |
| `MATMUL_WARP_N` | 64 | Columns of C one warpgroup contributes |
| `MATMUL_TILE_M` | 128 | **Block** tile height along M (two warpgroups) |
| `MATMUL_TILE_N` | 64 | Matches `WARP_N`; block width along N |
| `MATMUL_TILE_K` | 64 | K chunk per TMA slice into shared |
| `MATMUL_WARP_K` | 16 | K step per inner `mma.load` / `mma.row.row` |
| `MATMUL_SWIZ` | 128 | Swizzle parameter for TMA + MMA (here `2 * TILE_K`) |

So **`TILE_M = 2 × WARP_M`**: the **block** owns a **128×64** output rectangle, and **two** warpgroups split it **along M** into two **64×64** panes. Along **N**, there is no split at the warpgroup level in this kernel — both groups work on the same **N** span, which is exactly **`WARP_N`**.

The **key insight** for reading the code: **the right-hand tile in shared memory is shared**. `rhs_load_s` has shape **`[MATMUL_WARP_N, MATMUL_TILE_K]`** — enough for **one** **B** slab for this block’s column stripe. **Both** warpgroups read that same buffer (via `rhs_load_s.chunkat(_, iv_warp)`), while **left-hand** data is **partitioned** by warpgroup index **`p1`**.

## `parallel p1 by 2 : group-4`

Choreo expresses multiple warpgroups with a **parallel** over a small index, here **`p1`**, with a **group** annotation that says how many warps cooperate per group:

```choreo
parallel p1 by 2 : group-4 {
  // body: two iterations (p1 = 0 and 1), each a warpgroup of 4 warps
}
```

Read **`by 2`** as “**two** warpgroup instances in this block.” Read **`: group-4`** as “**each** instance is a **warp group** of **four** warps” (128 threads), which is the usual WGMMA thread team on Hopper.

Outside this region, the block still has a single **K** loop and shared buffers. Inside, **`p1`** selects **which** of the two groups is executing — and that index drives **which slice of `lhs_load_s`** and **which slice of `output_s`** this group uses.

The **128 threads** per warpgroup matter for how the hardware executes **`mma.row.row`**: WGMMA is a **collective** on a **warp group**, not a single warp. Choreo’s **`: group-4`** states that requirement in the tileflow so the lowering path can target the right instruction shape. When you read vendor documentation that says **“four warps cooperate”**, map that directly to **`group-4`** here.

## Full kernel walkthrough

Below is the **tileflow** core of the benchmark kernel (eliding C++ harness). Compare it mentally to a single-warpgroup Hopper matmul: the **TMA loads** and **output tile** are sized for **`TILE_M`**, while **compute** is duplicated in shape but **split** across **`p1`**.

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_TILE_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_TILE_M, MATMUL_WARP_N] output_s;

    mc = mma.fill.f32 0.0f;
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      tma.copy.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
      tma.copy.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s;
      parallel p1 by 2 : group-4 {
        foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
          ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0).chunkat(_, iv_warp);
          mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
          mma.row.row mc, ma, mb;
        }
      }
    }
    parallel p1 by 2 : group-4 {
      mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
    }
    tma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_WARP_N).at(block_m, block_n);
  }
}
```

### Grid: one block per 128×64 output tile

The outer `parallel {block_m, block_n}` uses **`cdiv(M, MATMUL_TILE_M)`** along M — each block covers **128** rows — and **`cdiv(N, MATMUL_WARP_N)`** along N. Note the **asymmetry**: **N** is stepped in units of **`WARP_N` (64)**, not **`TILE_N`**, because the inner compute is already sized to **64** columns per group and this kernel does not stack multiple groups along N.

Name the **output** tile this block owns **`C[block_m, block_n]`** in logical coordinates: a **128×64** submatrix of the full **`output`**. Index **`block_m`** advances in steps of **128** along **M**; **`block_n`** advances in steps of **64** along **N**. That matches how **`lhs.subspan(MATMUL_TILE_M, …).at(block_m, …)`** strides down **M** and how **`rhs.chunkat(block_n, …)`** selects the **N** stripe of **B** (stored as **`[N, K]`** in this kernel) that multiplies into those **128** rows of **A**.

### Shared buffers sized for the **block** tile

- **`lhs_load_s`**: **`[128, 64]`** — the full **A** tile for this block and current **K** chunk.
- **`rhs_load_s`**: **`[64, 64]`** — one **B** tile; **both** warpgroups consume it.
- **`output_s`**: **`[128, 64]`** — staging for the **entire** block result before the final TMA to global.

In **bytes**, the dominant pieces scale with **`TILE_M`**: **`lhs_load_s`** and **`output_s`** are **FP16**, so each is **`128 × 64 × 2`** bytes per **K** slab for **A** (one slab resident at a time in this simple version) plus the **staged C** tile. **`rhs_load_s`** stays **`64 × 64 × 2`**. When you add warpgroups along **M**, **A** and **C** staging grow **linearly** with **`TILE_M`**; **B** staging does **not**, which is exactly the **reuse** story.

### K loop: TMA in, then both warpgroups accumulate

For each **`iv_k`**:

1. **Load A** — TMA fills **`lhs_load_s`** with the **`MATMUL_TILE_M × MATMUL_TILE_K`** slab from global **`lhs`** at **`(block_m, iv_k)`**.
2. **Load B** — TMA fills **`rhs_load_s`** from **`rhs`** at this block’s column **`block_n`** and the same **K** slice.

Then **`parallel p1 by 2 : group-4`** runs the **math**:

- **`ma`** — `lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0)` takes a **`64 × 64`** view of **`lhs_load_s`**: row offset **`p1 * 64`**, column offset **0**. So **`p1 = 0`** gets the **top** half of the **A** tile, **`p1 = 1`** the **bottom** half.
- **`mb`** — `rhs_load_s.chunkat(_, iv_warp)` does **not** depend on **`p1`**: every warpgroup loads the **same** **B** stripes as **`iv_warp`** advances along **K**.
- **`mma.row.row mc, ma, mb`** — each group has **its own** **`mc`** (conceptually per-group accumulator), accumulating the product of **its** **A** rows with the **shared** **B** data.

The inner **`foreach {iv_warp}`** walks **`TILE_K / WARP_K = 4`** steps along the **K** dimension inside shared memory, matching the usual blocked MMA pattern.

**Concrete walk-through for one `iv_k`.** Suppose **`block_m`**, **`block_n`**, and **`iv_k`** are fixed. TMA lands a **128×64** **A** panel and a **64×64** **B** panel in shared. For **`p1 = 0`**, **`ma`** sees rows **0…63** of **`lhs_load_s`**; for **`p1 = 1`**, rows **64…127**. For **both**, **`mb`** cycles through **four** **K**-shards of **`rhs_load_s`** as **`iv_warp`** runs **0…3**. After this **`iv_k`**, **`mc`** in each group holds the **partial** sum for **its** **64×64** **C** patch for **all** **K** processed **so far**. The outer **`foreach {iv_k}`** repeats that story until **K** is exhausted, so **`mc`** is the **full** dot-product along **K** for each patch.

**Producer–consumer ordering.** The text lists **TMA** copies before the **`parallel p1 by 2`** MMA nest. That is the **logical** order: **shared** must hold the new **A** and **B** slabs before **`mma.load`**. The implementation may use **asynchronous** TMA with explicit **barriers**; the tileflow you read in the benchmark still expresses **dependences** in the same way as a synchronous kernel — **consumers** start only after **producers** finish for that **K** iteration. If you later **pipeline** two **K** buffers in shared, the **multi-warpgroup** split is unchanged: each **phase** still ends with **both** groups having seen **consistent** **`lhs_load_s`** and **`rhs_load_s`** for the slice they multiply.

### Store: partition **`output_s`** like **`lhs_load_s`**

After all **K** iterations, a second **`parallel p1 by 2 : group-4`** stores each group’s **`mc`** into **its** subrectangle of **`output_s`**:

```choreo
mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
```

This mirrors the **load** geometry: **`subspan(64, 64).at(p1, 0)`** addresses a **64×64** window inside the **128×64** **`output_s`**, so the two groups **do not stomp** each other’s outputs.

### Epilog: one TMA for the whole tile

Finally,

```choreo
tma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_WARP_N).at(block_m, block_n);
```

writes the **complete** **128×64** staged tile to global **`output`**. From the host’s perspective, this block has computed the same result as a hypothetical kernel with **smaller** **M** tiles and **more** blocks — but with **half** as many blocks along **M** for this configuration.

### Swizzle and **`MATMUL_SWIZ`**

Chapter 4 tied **TMA swizzle** to **shared-memory** behavior and **MMA** operand layout. Here **`MATMUL_SWIZ = 128`** matches **`2 * MATMUL_TILE_K`**, enforced by **`#if`** checks in the benchmark source. **Swizzle** applies **uniformly** to **`lhs_load_s`** and **`rhs_load_s`**: **`p1`** only shifts the **origin** of **`ma`** inside **`lhs_load_s`** via **`.at(p1, 0)`**; it does **not** imply a different swizzle mode per group. If you change **`TILE_K`**, update **swizzle** and re-validate **TMA** and **`mma.load`** together — the **multi-warpgroup** pattern does not relax those constraints.

## Mental model checklist

When you see **`TILE_M`** and **`WARP_M`** in the same kernel:

- **`TILE_*`** — **shared memory and TMA** are usually sized for what **the whole block** owns in one shot.
- **`WARP_*`** (or **`group-4`** MMA geometry) — **each warpgroup’s** compute and accumulator are sized for **one** Tensor Core “team” worth of output rows/columns.
- **Replication** — **B** is **replicated in use** (one copy in shared, many readers), not **replicated in storage**; **A** is **split** so each group only **loads** the rows it **owns**.

### What to get wrong first (so you can fix it faster)

Two bugs show up often when people **first** sketch multi-warpgroup matmuls by analogy with **multi-warp** Ampere kernels:

1. **Indexing `rhs` with `p1`.** If **`mb`** accidentally depended on **`p1`**, you would be telling a story where **different** groups need **different** **B** tiles — at that point you usually want **different** **`block_n`** or an **N-side** split, not a second **M** group reading the same **`rhs_load_s`**.

2. **Forgetting to partition `output_s`.** If both groups **`mma.store`** to **`output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(0, 0)`**, the **second** group **overwrites** the **first**. The **`.at(p1, 0)`** on **store** is not decorative; it is the **same** **M-axis** addressing as on **`ma`**.

Also watch **occupancy**: **doubling** **`TILE_M`** grows **`lhs_load_s`** and **`output_s`** roughly **linearly** in that dimension. **Shared** limits can **push** you to **fewer** resident blocks per SM even as **arithmetic intensity** improves. Profile **both** **throughput** and **occupancy** when you change **`TILE_M`** or the **number** of groups.

## Extending to three or more warpgroups along M

Nothing in the pattern is special to **two**. If hardware resources allow, you can set **`MATMUL_TILE_M = 192`** with **`MATMUL_WARP_M = 64`**, use **`parallel p1 by 3 : group-4`**, and keep the same **shared** **`rhs_load_s`** while **`lhs_load_s`** and **`output_s`** grow to **192×64**. Each **`p1`** would select **`subspan(64, …).at(p1, 0)`** with **`p1 ∈ {0,1,2}`**. The same story applies for larger **multiples** as long as **`TILE_M`** remains divisible by **`WARP_M`** and you respect target-specific limits on warpgroups per block, shared memory, and registers.

**N-side** scaling is a **different** design fork. This chapter’s kernel keeps **`parallel p1`** strictly on the **M** split. If you also want **two** **64×64** patches **along N**, you typically introduce **another** parallel axis (or **another** **block** dimension) and **either** **replicate** or **partition** **B** accordingly — you **cannot** blindly keep a **single** **`rhs_load_s`** sized **`[64,64]`** if **each** group needs **disjoint** **N** data. The **M-only** split is the **sweet spot** for **one B tile** serving **many A row bundles**.

### Sketch: three warpgroups

If you generalize to **`MATMUL_TILE_M = 192`**, the tileflow changes only where **counts** and **buffer sizes** appear — the **pattern** is unchanged:

```choreo
shared f16 [192, MATMUL_TILE_K] lhs_load_s;
shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
shared f16 [192, MATMUL_WARP_N] output_s;
// ...
parallel p1 by 3 : group-4 {
  foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
    ma = mma.load.swiz<MATMUL_SWIZ>
      lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0).chunkat(_, iv_warp);
    mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
    mma.row.row mc, ma, mb;
  }
}
parallel p1 by 3 : group-4 {
  mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
}
```

The **`rhs_load_s`** declaration is **identical** to the two-group case: still **`[64, 64]`** in **K** for this **N** stripe. Only **`lhs_load_s`**, **`output_s`**, and **`by 3`** track the taller **M** tile. Your **grid** along **M** becomes **`cdiv(M, 192)`** instead of **`cdiv(M, 128)`**.

## Relation to Chapter 7

Chapter 7’s **persistent** kernels reuse **one** CTA across **many** tiles by **iterating** **`block_m` / `block_n`** in software. This chapter’s **multi-warpgroup** layout reuses **B** across **two** accumulators **within** one tile iteration. In a full production stack you might **combine** both: persistent scheduling **and** multiple warpgroups per tile, trading **even more** launch and DRAM traffic overhead for **higher** per-SM work — always subject to occupancy and memory capacity.

If you are reading the chapters in order: Chapter 5 set up **MMA** lifecycle and **warp** vs **block** tiles; Chapter 4’s **TMA + swizzle** is exactly what feeds **`lhs_load_s`** and **`rhs_load_s`** here; Chapter 7’s **persistence** is about **how many tiles** a CTA visits, not **how wide** each tile is along **M**. This chapter sits at the intersection — **wider tiles** without **wider** **B** storage.

## `subspan` and `.at(p1, 0)` in plain language

Choreo’s **view** operations compose the same way whether you have one warpgroup or four. **`subspan(MATMUL_WARP_M, MATMUL_TILE_K)`** declares a **window** whose **height** is **64** rows and **width** is the full **K** chunk (**64**) sitting in **`lhs_load_s`**. That window is **logically** a **64×64** matrix for each **`iv_warp`** step after **`chunkat(_, iv_warp)`** narrows along **K**.

**`.at(p1, 0)`** then **slides** that window **down** the **M** axis: **`p1 = 0`** anchors at row **0** of **`lhs_load_s`**, **`p1 = 1`** at row **64**. The second index **0** is the column anchor inside the **subspan** — here the **left** edge of the **K** chunk, because the **entire** **TILE_K** width is already addressed by **`subspan(..., MATMUL_TILE_K)`** and the **`chunkat`** picks **which 16-column stripe** (for **`WARP_K = 16`**) participates in this **`mma.row.row`**.

The **store** line repeats the **same** **subspan** shape and **same** **`.at(p1, 0)`**, but the buffer is **`output_s`** instead of **`lhs_load_s`**. That **symmetry** is deliberate: whatever **rows** of **A** a group used to **form** its **`mc`**, it writes **`mc`** back to the **matching** **rows** of the **staged** **C** tile. If **load** and **store** ever disagree on **`p1`**, you have a **silent** correctness bug — the kind **unit tests** catch quickly if **M** and **N** are small enough to **eyeball** a full matrix.

## Single warpgroup vs. this kernel (at a glance)

| Aspect | One **`group-4`** warpgroup per block | Two warpgroups (**this chapter**) |
|--------|----------------------------------------|-------------------------------------|
| **Output tile per block** | **64×64** (with these **`WARP_*`**) | **128×64** |
| **`lhs_load_s` height** | **64** | **128** |
| **`rhs_load_s` shape** | **64×64** | **unchanged** — still **one** **B** tile |
| **Blocks along M** (fixed **M**) | **`cdiv(M, 64)`** | **`cdiv(M, 128)`** |
| **Accumulators** | one **`mc`** per group | **two** **`mc`** instances (**one** per **`p1`**) |
| **Parallel pattern** | **`parallel p1 by 1 : group-4`** (or implicit single group) | **`parallel p1 by 2 : group-4`** |

The table is **schematic**: real single-warpgroup kernels might still use **`TILE_M = 128`** with **more** warps doing **non-MMA** work, or different **N** tiling. The point is to separate **“how tall is shared **A**?”** from **“how many WGMMA teams share one **B**?”** — this chapter’s answer is **128** and **two**.

## Check your understanding

**Why is `rhs_load_s` not shaped `[MATMUL_TILE_M, MATMUL_TILE_K]`?** Matrix **B** in the multiply only needs **as many N rows as the block’s N tile** and **as many K columns as the current slab**. Here the **N** tile width is **64**, matching **`WARP_N`**. Doubling **`TILE_M`** does not double the **N** dimension of **B** for this block — both **64×64** result patches sit in the **same** **N** column range. Hence **one** **`rhs_load_s`** suffices.

**Could both groups use the same `ma` slices?** No — that would compute the **same** **64×64** output twice instead of **two** **different** row bundles of **C**. The **hardware** wants **two** **independent** accumulators because **two** **different** slices of **A** participate.

**Does `p1` ever appear in the global TMA source coordinates?** Not in this kernel. **Global** **A** is loaded **once** per **K** iteration as a **128×64** tile into **`lhs_load_s`**. **Partitioning** happens **after** the data is **in shared**, via **`subspan` / `.at`**. That keeps **TMA** descriptors **simple** and avoids issuing **two** **A** loads that overlap in memory.

## Reading the benchmark file

The repository path **`choreo/benchmark/performance/matmul/matmul_f16_dynamic_mwg_impl_1.co`** contains the full **C++** harness (timing, verification, CLI flags) around the **`__co__ void matmul`** you saw above. The **preprocessor** block at the top is worth skimming even if you never change the constants: it encodes **WGMMA** shape rules (**`WARP_M`**, **`WARP_K`**), **swizzle** legality, and **`TILE_M % WARP_M`**. Treat those **`#error`** lines as **documentation** that happens to be executable — when a future you bumps **`TILE_K`**, the compiler will refuse combinations that break **FP16** Hopper assumptions instead of failing mysteriously at runtime.

## Summary

- A **single** warpgroup on this path naturally covers **64×64** of the result; **two** groups inside one block cover **128×64** by **splitting A and C along M** and **sharing B**.
- **`parallel p1 by 2 : group-4`** launches **two** **128-thread** warpgroups; **`p1`** indexes which **M** slice each group owns in **`lhs_load_s`** and **`output_s`**.
- **`rhs_load_s`** is **one** tile per **K** step; both groups read it, which is the **main reuse win** versus launching separate blocks for each **64-row** strip.
- **`mma.store`** uses the **same** **`subspan(…).at(p1, 0)`** idiom as **`mma.load`** so each accumulator lands in the correct **half** of **`output_s`**.

The benchmark source **`matmul_f16_dynamic_mwg_impl_1.co`** is a concrete reference implementation with preprocessor checks (for example **`TILE_M` divisible by `WARP_M`**, **`SWIZ` tied to `TILE_K`**) that guard the assumptions this chapter relied on. When you tune tile sizes, start from those constraints and measure — **larger blocks** are not automatically faster, but **multi-warpgroup** layout is the standard way to grow **M** without giving up **WGMMA**’s native **64-row** chunks.

---

**Next:** [Chapter 9 — Beyond `chunkat`: `view` and `from` for irregular access](ch09-view-from.md) continues the tutorial with more flexible indexing when rigid **`subspan`** / **`chunkat`** tiles are not enough.
