# Persistent Matmul: if-guards, step, and Tile Iteration

In Chapter 4 you saw a Hopper-style FP16 matrix multiply whose **grid** was sized directly from the problem: one CUDA **thread block** (CTA) per output tile, via `parallel {block_m, block_n} by [cdiv(M, ...), cdiv(N, ...)] : block`. That pattern is easy to reason about: each block owns one **`MATMUL_WARP_M × MATMUL_WARP_N`** region of the output, loads K-panels with TMA, accumulates with WGMMA, and stores the result. Chapter 6 added **warp specialization** inside that story—different warps playing different roles in a pipelined loop.

This chapter changes the **launch geometry**. Instead of spawning as many CTAs as there are output tiles, we launch a **fixed** number of blocks—here, one per streaming multiprocessor (SM)—and let each block **iterate** over many tiles in software. That is the **persistent kernel** pattern: a small, steady pool of CTAs that keeps working until the whole output is covered.

By the end of this chapter, you should understand:

- Why a fixed CTA count can use hardware more efficiently when the tile grid is large
- How to **linearize** output tiles and assign them to blocks with `tile_iter # block_id`
- Why an **`if (tile_id < total_tiles)`** guard is required when `total_tiles` is not divisible by the CTA count
- What **`.step(M, K)`** on **`subspan`** means when stride should be spelled explicitly
- How **integer division and remainder** recover **`(block_m, block_n)`** from a linear **`tile_id`**
- How **`NUM_SMS`** relates to the GPU you target and why **`cdiv(total_tiles, NUM_SMS)`** is the shared loop bound

---

## Why persistent kernels?

Suppose **`M`** and **`N`** are large enough that the number of output tiles **`tiles_m * tiles_n`** is **much larger** than the number of SMs on your GPU. With a **data-dependent** grid—one CTA per tile—the hardware schedules tiles in **waves**. After each wave finishes, a new wave starts. In the **last** wave, only **some** SMs have work; the rest sit idle. That **tail** underutilizes the chip.

A **persistent** layout flips the picture: you launch **`NUM_SMS`** blocks (or another fixed count you choose), and **every** block stays busy across **many** logical tiles. Work is **striped** across CTAs so that block **`b`** handles tiles **`b`, `b + NUM_SMS`, `b + 2 * NUM_SMS`, ...**. As long as there is another tile assigned to that block, it keeps going; you avoid the “almost empty final wave” problem at the **grid** level because you never asked for **`total_tiles`** separate CTAs in the first place.

The tradeoff is **software complexity**: you need a loop over tile indices, a mapping from a **linear** tile id back to **`(block_m, block_n)`**, and a guard for **padding** iterations when the striping does not divide evenly. Choreo’s tileflow makes that explicit and readable.

The reference for this chapter is the **static persistent** FP16 matmul on SM90 in Choreo’s benchmarks: `choreo/benchmark/performance/matmul/matmul_f16_dyn_persis_sta_sm90.co`. The **tileflow** body below matches that kernel’s structure (constants are discussed in the next section).

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  int total_tiles = cdiv(M, MATMUL_WARP_M) * cdiv(N, MATMUL_WARP_N);

  parallel block_id by NUM_SMS : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)] {
      tile_id = tile_iter # block_id;
      if (tile_id < total_tiles) {
        block_m = tile_id / cdiv(N, MATMUL_WARP_N);
        block_n = tile_id % cdiv(N, MATMUL_WARP_N);

        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          tma.copy.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).step(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          tma.copy.swiz<MATMUL_SWIZ> rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).step(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k) => rhs_load_s;
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            parallel p by 1 : group-4 {
              ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.chunkat(_, iv_warp);
              mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
              mma.row.row mc, ma, mb;
            }
          }
        }
        mma.store mc, output_s;
        tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**Typical constants** in that file (Hopper-class **H800 PCIe** in the benchmark) include:

- **`NUM_SMS = 114`** — persistent launch width chosen to track SM count for that SKU
- **`MATMUL_WARP_M = 64`**, **`MATMUL_WARP_N = 128`**, **`MATMUL_TILE_K = 64`**, **`MATMUL_WARP_K = 16`**, **`MATMUL_SWIZ = 128`**

The **inner** K-loop, TMA + WGMMA structure, and **`parallel ... : group-4`** warpgroup are the same ideas as Chapter 4 and Chapter 5; what is new is the **outer** `parallel block_id by NUM_SMS`, the **`foreach tile_iter`**, the **`#`** composition, **`if`**, **`.step`**, and the arithmetic that maps **`tile_id`** to **`block_m` / `block_n`**.

---

## Fixed launch count: `parallel block_id by NUM_SMS : block`

Earlier matmul chapters used a **two-dimensional** parallel over **`block_m`** and **`block_n`**, with bounds derived from **`M`** and **`N`**. Here we use a **one-dimensional** parallel over **`block_id`**:

```choreo
parallel block_id by NUM_SMS : block {
```

This means: launch exactly **`NUM_SMS`** thread blocks. The index **`block_id`** runs from **`0`** to **`NUM_SMS - 1`**. It is **not** tied to a single output tile anymore; it is simply **which** persistent worker you are in the pool.

Contrast that with:

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
```

In the data-dependent version, **grid size grows with problem size**. In the persistent version, **grid size is fixed** (here, **`NUM_SMS`**), and **problem size** shows up in **`total_tiles`** and the **inner** loop count instead.

---

## Linearizing tiles and striping them across CTAs

We treat the **`tiles_m × tiles_n`** output grid as a single sequence of **`total_tiles`** tiles, numbered **`0 .. total_tiles - 1`** in **row-major** order over **`(block_m, block_n)`**:

```choreo
int total_tiles = cdiv(M, MATMUL_WARP_M) * cdiv(N, MATMUL_WARP_N);
```

So **`tile_id = 0`** corresponds to **`(block_m, block_n) = (0, 0)`**, then **`(0, 1)`**, …, then **`(1, 0)`**, and so on—exactly the order implied by:

```choreo
block_m = tile_id / cdiv(N, MATMUL_WARP_N);
block_n = tile_id % cdiv(N, MATMUL_WARP_N);
```

Here **`cdiv(N, MATMUL_WARP_N)`** is the number of tiles along **N**; dividing by it recovers the row index **`block_m`**, and the remainder is the column index **`block_n`**. This is ordinary integer **tileflow** arithmetic: it tells TMA and the output store **which** logical output tile this iteration is responsible for.

Each persistent CTA does not own a contiguous chunk of that sequence. Instead, CTA **`b`** processes **`tile_id`** values **`b`, b + NUM_SMS, b + 2 * NUM_SMS, ...`**—a **stride-`NUM_SMS`** walk through the linearized list. That is how **`NUM_SMS`** workers share the load evenly in the steady state.

---

## `tile_iter`, `block_id`, and the `#` operator

The loop:

```choreo
foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)] {
  tile_id = tile_iter # block_id;
```

does two things at once.

First, **`tile_iter`** counts **which pass** this CTA is on: **`0`**, **`1`**, …, up to **`cdiv(total_tiles, NUM_SMS) - 1`**. The upper bound uses **`cdiv`** (ceiling division) so that if **`total_tiles`** is not a multiple of **`NUM_SMS`**, we still iterate enough times for the **last** few tiles to be reached by **some** block.

Second, **`tile_id = tile_iter # block_id`** **composes** the pass index with the **block index** to form the **global linear tile id** for this iteration. Read **`#`** here as “combine **`tile_iter`** and **`block_id`** into the **`tile_id`** that this striping scheme assigns to this block at this step.” In other words, for block **`block_id`**, iteration **`tile_iter`** handles the tile at position **`tile_iter * NUM_SMS + block_id`** in the linear ordering—matching the stride-**`NUM_SMS`** picture above.

If you are coming from CUDA C, this is the same algebra you would write by hand for persistent kernels; Choreo just makes the **iteration space** and **composition** part of the tileflow language.

### Why the loop bound is `cdiv(total_tiles, NUM_SMS)`

You might first guess that each block should iterate **`total_tiles / NUM_SMS`** times. That only works when **`total_tiles`** divides evenly by **`NUM_SMS`**. In general, **some** blocks must run **one more** pass than others so that **all** tiles **`0 .. total_tiles - 1`** appear in the table exactly once.

Using **`cdiv(total_tiles, NUM_SMS)`** as the **shared** bound is the standard fix: it is the **maximum**, over all blocks, of how many **`tile_iter`** steps that block needs before every valid **`tile_id`** for that block has been visited. Blocks that would otherwise “finish early” still execute the extra iterations, but those iterations hit **`tile_id >= total_tiles`** and **skip** the **`if`** body—cheap compared to launching **`total_tiles`** separate kernels or using divergent grid sizes per block (which CUDA does not offer at launch granularity anyway).

---

## The `if` guard: out-of-range `tile_id`

Because **`foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)]`** may run **one extra** iteration for some blocks when **`total_tiles`** is not divisible by **`NUM_SMS`**, some combinations **`(tile_iter, block_id)`** produce a **`tile_id`** that is **past the end** of the real tile list—**`tile_id >= total_tiles`**.

The guard:

```choreo
if (tile_id < total_tiles) {
  ...
}
```

skips **all** TMA, MMA, and store work for those **padding** iterations. Shared memory is still **declared** at block scope (outside the `if`), but only **valid** tiles execute the body. Without this guard, you would index **`block_m` / `block_n`** and global tensors **out of range** for bogus **`tile_id`** values.

This is a central pattern in persistent tileflow: **round up** the iteration count for simplicity, then **predicate** the real work.

### What stays outside the `if`

In the listing at the top of the chapter, **`shared`** buffers are declared **once per CTA**, **outside** the **`if`**. That matches how you would allocate SMEM in CUDA: **one** allocation per thread block, reused for **every** logical tile that block processes. Only the **contents** and the **global indices** change per iteration.

If you later add **warp-specialized** producers and consumers (Chapter 6), you will still want **persistent** outer loops to sit **outside** role-specific inner regions so that **all** warps agree on how many **tile** iterations the block performs—even when only some warps participate in a given stage.

---

## `.step(MATMUL_WARP_M, MATMUL_TILE_K)` on `subspan`

The TMA lines use **`subspan`** with an explicit **`.step`**:

```choreo
lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).step(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).step(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k)
```

Conceptually, **`subspan`** picks a **tile shape** (rows × columns in the tensor’s index space), **`.at(block_m, iv_k)`** (or **`block_n`**) anchors that tile in the **global** matrix, and **`.step`** states the **stride** between successive tiles along those dimensions when the stride might differ from the **visible** tile size or when you want the compiler and reader to see the layout spelled out.

In earlier chapters, **`subspan`** often appeared **without** **`.step`**, relying on the **default** that stride matches the tile extent. Here, **`.step(MATMUL_WARP_M, MATMUL_TILE_K)`** matches the **M × K** tile shape for LHS (and similarly for RHS along **N × K**). For this symmetric case the numeric stride equals the subspan size, but writing **`.step`** makes the **intent** obvious: “this is the hop between adjacent tiles in index space,” which matters when you start varying layouts or reading kernels that use non-default strides.

The **store** path still uses **`output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n)`** without an explicit **`.step`** in the snippet—consistent with Chapter 4’s style when the default is clear.

### When `.step` differs from the subspan shape

The persistent matmul uses **equal** step and tile size along each axis, which is the common case for **dense** GEMM tiles that abut without gaps. Other kernels use **`.step`** to express **non-contiguous** tiling: for example, a subspan might cover **64** elements along K while the **next** tile in the sequence starts **128** elements away because of **padding**, **interleaving**, or a **blocked storage** format. Treat **`.step`** as part of the **view** you hand to TMA: it tells the copy **how indices advance** when you move **`iv_k`** or **`block_m`** by one in the logical loop, not only how big one tile is.

---

## Putting the inner loop in context

Inside the **`if`**, the body is intentionally familiar:

1. **`mma.fill.f16`** seeds the accumulator **`mc`**.
2. **`foreach iv_k`** stages **`MATMUL_TILE_K`**-wide K-panels from **`lhs`** and **`rhs`** into **`lhs_load_s`** and **`rhs_load_s`** with **`tma.copy.swiz<MATMUL_SWIZ>`**.
3. **`foreach iv_warp`** with **`parallel ... : group-4`** runs **WGMMA** (`mma.load.swiz`, **`mma.row.row`**) over **`MATMUL_WARP_K`** chunks along K.
4. **`mma.store`** then **`tma.copy`** commits **`output_s`** to **`output`** at **`(block_m, block_n)`**.

So the **only** structural difference from the one-tile-per-CTA kernel is the **wrapper**: fixed **`parallel block_id`**, **`foreach tile_iter`**, **`tile_id = tile_iter # block_id`**, **`if (tile_id < total_tiles)`**, and the **linear-to-2D** mapping. The **tensor-core pipeline** itself is unchanged.

### Data-dependent grid versus persistent launch (recap)

| Aspect | One CTA per tile (Chapter 4 style) | Persistent (`NUM_SMS` CTAs) |
|--------|-----------------------------------|-----------------------------|
| Launch count | **`cdiv(M, …) * cdiv(N, …)`** — grows with problem size | **Fixed** — e.g. **`NUM_SMS`** |
| Which tile a block does | **`(block_m, block_n)`** from the parallel indices | **`(block_m, block_n)`** from **`tile_id`** after striping |
| Tail SM utilization | Last grid wave may leave SMs idle | CTAs keep pulling tiles until none remain |
| Extra tileflow | Minimal | **`total_tiles`**, **`foreach tile_iter`**, **`#`**, **`if`** |

Neither style changes the **correctness** of the multiply; both should match a reference GEMM modulo floating-point associativity. The persistent form is an **occupancy and scheduling** choice, especially attractive when **`total_tiles >> NUM_SMS`**.

---

## Tile scheduling variants in the repo

The benchmark directory also ships **persistent** matmuls that change **how** linear **`tile_id`** maps to **`(block_m, block_n)`**—or equivalently, how tiles are **ordered** in memory and time—for example:

- **`matmul_f16_dyn_persis_colmajor_sm90.co`** — column-major (or otherwise reordered) traversal
- **`matmul_f16_dyn_persis_swizzle_sm90.co`** — **swizzle**-style reordering of tile ids for locality or load balance
- **`matmul_f16_dyn_persis_hilbert_sm90.co`** — **Hilbert curve**–style space-filling traversal

The **launch** and **`if`-guard** ideas stay the same; what changes is the **formula** that turns **`tile_id`** into **`block_m`** and **`block_n`** (and sometimes the goal—e.g., spreading traffic across DRAM channels or reducing L2 thrash). When you read those files, compare their **`tile_id`** mapping to the simple row-major **`/`** and **`%`** shown here.

SM86 variants of the same names (e.g. **`matmul_f16_dyn_persis_colmajor_sm86.co`**) adapt the **same scheduling ideas** to an earlier architecture’s tensor and copy instructions; the **persistent loop skeleton** is still the right mental model.

---

## Choosing `NUM_SMS` and partial output tiles

The benchmark pins **`NUM_SMS`** to **114** for **H800 PCIe** because that matches the **SM count** of the SKU under test. In your own port, you might set **`NUM_SMS`** from a **runtime query** (driver API) or a **compile-time** constant for a **family** of GPUs. The persistent pattern still works if you launch **fewer** CTAs than SMs—some SMs simply stay idle—or **more** CTAs than SMs (multiple blocks per SM over time), though the sweet spot is usually **close to** the hardware SM count so you **fill** the machine without inventing gratuitous oversubscription.

**`total_tiles`** already accounts for **partial** tiles at the **bottom** and **right** of the **`M × N`** output because **`cdiv(M, MATMUL_WARP_M)`** and **`cdiv(N, MATMUL_WARP_N)`** count tiles the same way the data-dependent grid did. TMA and the store still target **`MATMUL_WARP_M × MATMUL_WARP_N`** logical tiles; the **global** tensor shapes **`M`**, **`N`**, **`K`** remain **symbolic** in the tileflow, and the generated code must respect **tail** elements exactly as in the non-persistent kernel. The **persistent** layer does not change **edge** behavior—it only changes **how many** CTAs iterate **which** **`(block_m, block_n)`** pairs.

---

## Checklist: evolving a matmul to persistent form

If you already have a **working** tile-per-CTA Hopper matmul, you can migrate it mechanically:

1. **Compute** **`total_tiles`** from **`M`**, **`N`**, and your output tile sizes (same **`cdiv`** products you used for the grid extents).
2. **Replace** **`parallel {block_m, block_n} by [...]`** with **`parallel block_id by NUM_SMS`** (tune **`NUM_SMS`** to your target GPU).
3. **Wrap** the former body in **`foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)]`**, set **`tile_id = tile_iter # block_id`**, and add **`if (tile_id < total_tiles)`**.
4. **Derive** **`block_m`** and **`block_n`** from **`tile_id`** with **`/`** and **`%`** (or swap in a **swizzle / Hilbert** mapping from the other benchmarks).
5. **Keep** **`shared`** declarations at **block** scope; **reuse** SMEM across tile iterations.
6. **Optional:** add **`.step(...)`** on **`subspan`** TMA sources if you want strides explicit or non-default.

After that, **re-run** numerical verification (same as any refactor): persistent kernels are easy to get **almost** right and still write **past the end** of **`output`** if the guard or mapping is off by one.

When debugging, print or log **`total_tiles`**, **`cdiv(total_tiles, NUM_SMS)`**, and—for a single block—**`block_id`** and the sequence of **`tile_id`** values across **`tile_iter`**. If any **`tile_id`** appears **twice** or **skips** a valid id, the **`#`** schedule or the loop bound is wrong; if invalid ids are **not** skipped by the guard, you will see **memory faults** or **garbage** in untouched regions of **`output`**.

---

## Summary

Persistent matmul kernels launch a **fixed** number of CTAs—**`parallel block_id by NUM_SMS : block`**—and **loop** over logical output tiles. **`total_tiles`** counts **`tiles_m * tiles_n`**; each CTA advances by **`NUM_SMS`** in the linearized order via **`tile_id = tile_iter # block_id`**. **`if (tile_id < total_tiles)`** drops **padding** iterations when **`total_tiles`** is not a multiple of **`NUM_SMS`**. **`.step(...)`** on **`subspan`** documents (and can generalize) the **stride** between tiles in global memory. Inside the guard, **integer division and remainder** recover **`block_m`** and **`block_n`**, and the **TMA + WGMMA + store** path matches the non-persistent Hopper matmul you already know from earlier chapters.

Next, Chapter 8 widens the story to **multiple warpgroups** and **multiple accumulators** so a single output tile can engage more SM throughput at once—building on the same TMA and WGMMA core you now know how to **schedule** across the machine with persistent CTAs.
