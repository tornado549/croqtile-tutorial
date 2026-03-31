# High-Performance Data Movement: TMA and Swizzle

In Chapters 2 and 3, you learned to move data with **DMA** — explicit copies that software orchestrates tile by tile. That model is clear and portable, but on modern NVIDIA GPUs it leaves performance on the table. Starting with the **Hopper** architecture (compute capability 9.0, often written **SM90**), NVIDIA added a dedicated hardware path for bulk, multi-dimensional loads and stores: the **Tensor Memory Accelerator**, or **TMA**.

This chapter introduces TMA in Choreo, together with **swizzle** — a layout trick that keeps shared-memory accesses from tripping over **bank conflicts** when tensor cores and wide loads expect a particular byte pattern in SMEM. We will read a real **FP16 matrix multiply** tileflow from the Choreo benchmark suite. The MMA pieces (`mma.load`, `mma.row.row`, and friends) appear here because they sit in the same loop as TMA; we will use them as motivation and name them, but **Chapter 5** is where we unpack tensor-core programming in depth.

By the end of this chapter, you should understand:

- Why TMA exists and how it differs from ordinary DMA in Choreo
- What swizzle means for shared memory and why the literal `128` shows up next to `tma.copy.swiz` and `mma.load.swiz`
- How **dynamic shapes**, **`global`**, **`void` + output parameters**, **`cdiv`**, **`subspan().at()`**, **`chunkat`**, **`shared` buffers**, **`: group-4`**, and the **`_`** wildcard fit together in a Hopper-style kernel

---

## From software copies to hardware tensor movement

On pre-Hopper GPUs, getting a tile from global memory into shared memory usually meant something like: many threads each load a few elements, or you use vectorized loads in a loop you wrote yourself. The **program** pays for every instruction that participates in that copy.

**TMA** moves that work into a **hardware unit**. You give it a description of a multi-dimensional tensor in memory (a *tensor map* or descriptor, in NVIDIA's terms), and it issues the right sequence of memory transactions to copy a **rectangular tile** in one logical operation. Warps can keep doing math (or wait on a lightweight barrier) while TMA works. That is the performance story: **fewer instructions spent on movement**, better overlap, and a path that lines up with what **tensor memory** and **WGMMA** expect on Hopper.

Choreo exposes this through **`tma.copy`** and variants like **`tma.copy.swiz<N>`**. The **swizzle** parameter tells the hardware (and the compiler) how to **remap** the destination layout in shared memory so that later **column-like** or **K-dimension** accesses do not hammer the same **shared memory bank** repeatedly.

Shared memory is split into **32 banks**. Successive 4-byte words (in the classic model) map to successive banks. If a warp reads a **row** and every lane reads a consecutive element, you get a nice broadcast or spread across banks. If a warp reads a **column** of a row-major `f16` matrix, many lanes can land on the same bank — **bank conflicts**, which serialize accesses and hurt throughput. **Swizzling** applies a fixed permutation of how addresses map to banks (often described in bytes, e.g. 32, 64, or **128**), so that the access pattern your MMA or TMA consumer uses sees fewer conflicts. For FP16 kernels, a common choice is **`swiz<128>`** when the tile width in **K** matches the recipe (in the benchmark, `MATMUL_TILE_K` is 64 elements of `f16` → 128 bytes per row fragment, and the build asserts `MATMUL_SWIZ == 2 * MATMUL_TILE_K`).

You do not need to memorize every hardware rule in one sitting. The important idea: **`tma.copy.swiz<N>`** is not just a copy; it is **copy + agreed-upon SMEM layout** so that **`mma.load.swiz<N>`** on the same buffer is consistent.

### How this relates to `dma.copy` from earlier chapters

In Chapters 2 and 3, **`dma.copy`** expressed the same *intent* — move a tile from one memory kind to another — but the generated path was oriented toward **software-driven** movement (threads participating in the copy, possibly pipelined with `inthreads` and events). That remains a valid mental model for portability and for hardware where TMA is not available.

**`tma.copy`** is the **Hopper-shaped** spelling: the Choreo compiler maps it to TMA descriptors and instructions appropriate for SM90-family targets. You still think in terms of **source fragment** and **destination buffer**, but the **mechanism** underneath is different. When a kernel uses **`mma.load.swiz`** and WGMMA-style **`parallel ... : group-4`**, pairing **TMA loads** with those intrinsics is the normal layout; trying to swap in generic DMA for the same SMEM buffers would break the assumptions the MMA side makes about **byte-level** placement unless you carefully matched layouts by hand.

In short: **DMA** teaches you **tiling and orchestration**; **TMA** is the **next gear** when you commit to Hopper tensor pipelines.

---

## Reference kernel: FP16 matmul tileflow (SM90)

The following is the **tileflow-only** core of the dynamic FP16 matrix multiply used in Choreo's Hopper benchmark (`matmul_f16_dyn_sm90.co`). Compile-time constants are:

- `MATMUL_WARP_M = 64`, `MATMUL_WARP_N = 128`, `MATMUL_TILE_K = 64`, `MATMUL_WARP_K = 16`
- `MATMUL_SWIZ = 128`
- `M`, `N`, and `K` are **symbolic** — the actual sizes come from the host at runtime.

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    mc = mma.fill.f16 0.0f;
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      tma.copy.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
      tma.copy.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s;
      foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
        parallel p by 1 : group-4 {
          ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.chunkat(_, iv_warp);
          mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
          mma.row.row mc, ma, mb;
        }
      }
    }
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
    mma.store mc, output_s;
    tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
  }
}
```

A **smaller, test-oriented** version lives in `choreo/tests/gpu/end2end/matmul_f16_dynamic.co`. It uses the same TMA + swizzle + warpgroup structure with simpler numbers (`WARP_M = 64`, `WARP_N = 64`, `TILE_K = 64`, `WARP_K = 16`, literal `128` for swizzle) and is easier to grep and run under the test harness. The **ideas** are identical; the benchmark kernel widens the **N** tile to 128 to tune throughput on real hardware.

---

## Walking the loops: what one thread block actually does

It helps to narrate a single **output tile** before you worry about the whole grid.

Pick indices **`block_m`** and **`block_n`**. Conceptually, this CUDA block is responsible for the **`MATMUL_WARP_M × MATMUL_WARP_N`** region of **`output`** at logical position **`(block_m, block_n)`** in the block grid — that is, rows roughly **`block_m * MATMUL_WARP_M`** through **`(block_m + 1) * MATMUL_WARP_M - 1`**, and columns **`block_n * MATMUL_WARP_N`** through **`(block_n + 1) * MATMUL_WARP_N - 1`**, modulo edge cases when **`M`** or **`N`** are not exact multiples (hence **`cdiv`** on the grid).

1. **Allocate SMEM** for this block's LHS and RHS **K-panels**: `lhs_load_s` holds an **`MATMUL_WARP_M × MATMUL_TILE_K`** tile; `rhs_load_s` holds **`MATMUL_WARP_N × MATMUL_TILE_K`**. These are the staging areas TMA will fill.

2. **Zero the accumulator** with **`mma.fill.f16`**. The variable **`mc`** is not ordinary SMEM; it is an **MMA accumulator state** that persists across the **`iv_k`** loop (Chapter 5 will say more about its type and scope).

3. **For each K-tile** **`iv_k`** from **`0`** to **`cdiv(K, MATMUL_TILE_K) - 1`**:
   - **TMA** pulls the corresponding **`MATMUL_WARP_M × MATMUL_TILE_K`** slice of **`lhs`** into **`lhs_load_s`**, using **`subspan(...).at(block_m, iv_k)`** so the global indexing matches the block's row stripe and the current K chunk.
   - **TMA** pulls the **`MATMUL_WARP_N × MATMUL_TILE_K`** slice of **`rhs`** for this **`block_n`** and **`iv_k`** into **`rhs_load_s`** via **`chunkat(block_n, iv_k)`**.
   - **Inside that loaded K-tile**, **`iv_warp`** runs from **`0`** to **`cdiv(MATMUL_TILE_K, MATMUL_WARP_K) - 1`**. With the numbers in the benchmark, **`MATMUL_TILE_K / MATMUL_WARP_K = 64 / 16 = 4`**: four **MMA slices** along K per outer K-tile.
   - For each **`iv_warp`**, the **warp group** loads **`ma`** and **`mb`** fragments from SMEM with **`mma.load.swiz`** and issues **`mma.row.row`** to fold that partial product into **`mc`**.

4. **After** all **`iv_k`** iterations, **`mc`** holds the full dot products for this output tile (again modulo Chapter 5 for numeric detail). **`mma.store`** writes **`mc`** to **`output_s`**, a dense **`MATMUL_WARP_M × MATMUL_WARP_N`** SMEM buffer without the input swizzle story (the consumer is just TMA to global).

5. **`tma.copy output_s => output.subspan(...).at(block_m, block_n)`** commits the tile to global memory.

Notice how **two nested notions of "K"** appear: the **outer** loop **`iv_k`** stages **`MATMUL_TILE_K`** elements at a time from global memory — a size chosen to match **memory** and **descriptor** efficiency — while the **inner** loop **`iv_warp`** feeds the **MMA** in thinner **`MATMUL_WARP_K`** slivers. The outer loop is about **what fits through TMA and SMEM**; the inner loop is about **what one WGMMA instruction eats per step**.

Chapter 3 taught you to **overlap** DMA with compute using software queues. On Hopper, producers still think in waves of **load → math → store**, but **TMA** and **async** barriers (topics for later chapters) let hardware participate in that overlap more aggressively than a naive thread loop.

---

## New syntax: dynamic shapes and `global`

### `global f16 [M, K]` and friends

Earlier chapters often used **fixed** shapes like `s32 [6, 17, 128]` in the function signature. Here, **LHS** is `global f16 [M, K]`:

- **`f16`** is **half precision** (16-bit floating point), the dtype for this FP16 matmul.
- **`M`, `K`, `N`** are **symbolic dimensions**. Their concrete values are supplied when the host builds views or launches the kernel — the tileflow is **generic** in those sizes.
- **`global`** is a **memory space qualifier**: this tensor lives in **device global memory** (the usual CUDA `__device__` heap), as opposed to `shared` or register-backed tiles you declare inside the kernel.

Using symbolic `M`, `N`, `K` lets one Choreo function describe **any** problem size that satisfies alignment and tiling constraints, without recompiling a different literal shape each time.

### `void` and the output parameter

Instead of returning a fresh `spanned_data` from the `__co__` function, this kernel is declared as:

```choreo
__co__ void matmul(..., global f16 [M, N] output)
```

**`void`** means the tileflow does not **return** a value type to the expression that invoked it. The result is written **in place** through **`output`**, which is a `global` view of the full `M × N` result matrix. That matches how many high-performance GPU entry points work: you pass destination pointers (or views) explicitly so the caller controls allocation and aliasing.

---

## Ceiling division: `cdiv`

Tiling requires: "how many **K** tiles do we need if each tile has height `MATMUL_TILE_K`?" If `K` is not a multiple of the tile size, you need one more partial tile — ordinary integer division truncates, so Choreo provides **ceiling division**:

```choreo
cdiv(K, MATMUL_TILE_K)
```

Same for the **grid** over output blocks:

```choreo
cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)
```

Any time you see `cdiv(a, b)` in tileflow code, read it as \(\lceil a / b \rceil\) for positive sizes.

When **`M`**, **`N`**, or **`K`** are not multiples of the tile sizes, the **last** tile along that dimension is **partial** in the mathematical sense. Real kernels often pair **`cdiv`** grid sizing with **predicate masks**, **zero-padding**, or **epilogue cleanup** so out-of-bounds threads do not read or write garbage. The tileflow here follows the same **grid** pattern you will see in tuned libraries; the exact guard strategy may be lowered in the device layer or rely on allocation padding from the host. If something looks "one block too wide" on paper, check how the host allocates buffers and whether the Choreo target inserts bounds checks for your configuration.

---

## Shared memory as named buffers

Inside the `parallel ... : block` region, we declare:

```choreo
shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
```

This is an **explicit shared-memory allocation** for that CUDA thread block: a 2D array of `f16` with fixed compile-time extents, living in **SMEM**. TMA's destination operand is exactly these buffers (`=> lhs_load_s`, `=> rhs_load_s`). Later, **`mma.load.swiz`** reads the **same** SMEM with the **same** swizzle tag so the layout contract is honored.

The accumulator path uses a separate SMEM tile for the result before scattering back to global:

```choreo
shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
mma.store mc, output_s;
tma.copy output_s => output.subspan(...).at(block_m, block_n);
```

So: **TMA in**, tensor cores **accumulate**, **mma.store** to SMEM, **TMA out** to global.

---

## `subspan().at()` versus `chunkat`

Both carve the global tensors into tiles, but they emphasize different things.

### `chunkat`

**`chunkat`** assumes a **regular decomposition**: divide each dimension into equal chunks according to the tile shape implied by the surrounding type and indexing. You have seen this in earlier chapters for DMA. Here:

```choreo
rhs.chunkat(block_n, iv_k)
```

indexes the **RHS** `global f16 [N, K]` by **block index along N** and **tile index along K**. The shape of the chunk is consistent with how the compiler and descriptor expect **N-major** vs **K-major** slicing for that operand.

### `subspan(M, K).at(i, j)`

**`subspan`** creates a **view** with explicit **tile extents** (here `MATMUL_WARP_M` by `MATMUL_TILE_K` for the LHS panel). **`.at(block_m, iv_k)`** then selects **which** tile in the global tensor: row of blocks `block_m`, K-tile `iv_k`.

Use **`subspan().at()`** when you want to be **explicit** about the **logical tile size** and **stride** of the view (especially when matching a fixed MMA or TMA tile), and **`chunkat`** when you are happy with the default **even tiling** along named indices.

For LHS:

```choreo
lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
```

For output:

```choreo
output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n)
```

Read it as: "take an `MATMUL_WARP_M × MATMUL_TILE_K` windowing pattern on `lhs`, then pick the window at `(block_m, iv_k)`," and similarly for the output tile.

| Situation | Often use | Rationale |
|-----------|-----------|-----------|
| LHS panel must match exact **`WARP_M × TILE_K`** for TMA/MMA | **`subspan(WARP_M, TILE_K).at(block_m, iv_k)`** | Makes the **view extents** literal in the source; easy to match **`shared`** sizes. |
| RHS / simpler regular tiling along block indices | **`chunkat(block_n, iv_k)`** | Shorter when the default chunking matches the tensor's role in the kernel. |
| Output tile writeback | **`subspan(WARP_M, WARP_N).at(block_m, block_n)`** | Same as LHS: explicit **tile footprint** in the global tensor. |

Both forms are **views** over the same underlying **`global`** data; the compiler still has to prove the access is coherent for the target. Prefer **`subspan().at()`** when you are **matching hardware recipes** from a spreadsheet or an existing CUTLASS/CUTE kernel, and **`chunkat`** when the decomposition is uniform and already obvious from context.

---

## TMA with swizzle: `tma.copy.swiz<N>`

The load stage for each K tile:

```choreo
tma.copy.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
tma.copy.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s;
```

- **`tma.copy`** is the Hopper bulk copy primitive in Choreo.
- **`.swiz<MATMUL_SWIZ>`** (here `128`) selects the **shared-memory swizzle pattern** for the destination.
- The left-hand side is a **global** tensor fragment; the right-hand side is **`shared`** SMEM.

The store back is an unswizzled **`tma.copy`** from `output_s` to the global `output` subspan — the swizzle that mattered was for **inputs** aligned with **`mma.load.swiz`**.

---

## Warpgroups: `: group-4`

Hopper **WGMMA** (warp-group matrix multiply accumulate) expects a **warp group** — here **4 warps** = **128 threads** — to cooperate on one MMA operation. Choreo annotates that with:

```choreo
parallel p by 1 : group-4 {
  ...
}
```

So even though the **parallel** count is `1` in the `p` dimension, the **team** that executes the body is a **group of 4 warps**. The `mma.row.row` intrinsic uses that collaboration. Chapter 5 goes deeper; for this chapter, treat **`: group-4`** as "schedule this body as a Hopper warp group, not a single warp."

---

## The `_` wildcard in `chunkat(_, iv_warp)`

Inside the K tile, the code steps **`iv_warp`** over sub-chunks of size **`MATMUL_WARP_K`** along K:

```choreo
foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
  parallel p by 1 : group-4 {
    ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.chunkat(_, iv_warp);
    mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
    mma.row.row mc, ma, mb;
  }
}
```

**`chunkat(_, iv_warp)`** means: tile along the **second** index (`iv_warp`) in the usual way, but **do not** introduce an extra outer chunk index on the first dimension — **`_`** is a **wildcard** that says "no tiling selector here; keep the full extent of that dimension for this operand." Intuitively, the **entire** `MATMUL_WARP_M` (or `MATMUL_WARP_N`) side is already in SMEM for this block; only K is subdivided for each MMA slice.

If you mistakenly wrote **`chunkat(some_m, iv_warp)`** on **`lhs_load_s`** here, you would be implying an extra partition along the **M** side of SMEM even though this block already owns a **fixed** **`MATMUL_WARP_M`** rows. The wildcard makes the **intent** obvious to readers and to tooling: **M (or N) is whole; K is the only chunked axis inside SMEM.**

---

## MMA lines (preview of Chapter 5)

A few lines tie the TMA loads to tensor cores:

```choreo
mc = mma.fill.f16 0.0f;
...
ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.chunkat(_, iv_warp);
mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
mma.row.row mc, ma, mb;
...
mma.store mc, output_s;
```

- **`mma.fill.f16`** seeds the accumulator (`mc`) to zero in FP16 semantics.
- **`mma.load.swiz`** brings a fragment from SMEM in the **same** swizzle layout TMA wrote.
- **`mma.row.row`** performs a multiply-accumulate in the **row-major** sense for A and B with accumulation into `mc` (details and shapes in Chapter 5).
- **`mma.store`** writes the accumulated tile back to SMEM for the final **`tma.copy`** to global.

Do not worry if the opcode names feel opaque on first read; the **data-movement** story you need from this chapter is: **TMA filled `lhs_load_s` / `rhs_load_s` with swizzle `N`; MMA consumed those buffers with the same `N`; then results left via SMEM and TMA.**

The end-to-end test **`matmul_f16_dynamic.co`** uses **`mma.fill.f32`** with **`mma.row.row`** and FP16 tensors — a slightly different accumulator story for the test harness than the **`mma.fill.f16`** benchmark variant. For **this** chapter, treat both as "initialize accumulator, accumulate, store"; Chapter 5 unifies the precision and layout rules.

---

## SM90, swizzle literals, and when things fail to compile

The **`// REQUIRES: TARGET-SM_90`** line on the test file is a hint: **TMA + WGMMA + `group-4`** is not a generic CUDA 7.0 feature. You need a **Hopper-class** target (and typically an **`sm_90a`** or similar arch flag in your Choreo invocation) for the pipeline in this chapter to lower cleanly.

The benchmark source **statically asserts** relationships between constants — for example that **`MATMUL_SWIZ`** equals **`2 * MATMUL_TILE_K`** for this FP16 recipe, and that **`MATMUL_SWIZ`** is one of **`32`**, **`64`**, **`128`**. Those checks mirror hardware constraints: **swizzle mode**, **element size**, and **tile width** must agree or the descriptor setup is invalid. If you change **`TILE_K`** without changing **`swiz`**, expect either a **compile-time error** from the C preprocessor or a **runtime failure** when building tensor maps, depending on how far the bad combination gets.

Treat **`MATMUL_WARP_K = 16`** for **`f16`** the same way: it is not an arbitrary tunable in isolation; it is tied to **the MMA instruction shape** on SM90. Chapter 7 and beyond revisit **autotuning**; for now, copy **working triples** `(TILE_K, SWIZ, WARP_K)` from known-good kernels before you experiment.

---

## Mental model checklist

When you read or write a Hopper-style Choreo kernel like this, ask:

1. **Where does data live?** `global` tensors vs `shared` buffers vs MMA accumulators.
2. **How big is each tile?** `WARP_M`, `WARP_N`, `TILE_K`, `WARP_K`, and how `cdiv` counts loops.
3. **How do global indices map to blocks?** `parallel {block_m, block_n} by [cdiv(M, ...), cdiv(N, ...)]`.
4. **Is the SMEM layout consistent?** The same numeric **`swiz<N>`** on **`tma.copy.swiz`** and **`mma.load.swiz`** for a given buffer.
5. **Who executes the MMA body?** **`: group-4`** for WGMMA-shaped work.

---

## Common pitfalls (movement and layout)

- **Mismatched swizzle tags.** If **`tma.copy.swiz<128>`** feeds a buffer but **`mma.load.swiz<64>`** reads it, you are asking tensor cores to interpret **the wrong byte layout**. Keep the template argument **identical** per SMEM allocation unless you **really** know a compatible remapping path exists (usually: it does not).

- **`subspan` extents disagree with `shared` sizes.** The global view's tile shape and the **`shared f16 [...]`** declaration must describe the **same** number of elements in each dimension. Off-by-one or transposition here shows up as silent wrong answers or illegal TMA shapes.

- **Forgetting that `void` means no return value.** Host code must pass a **writable** `output` view; you cannot assign the result of **`matmul(...)`** to a fresh **`spanned_data`** as in Chapter 1 unless you wrap a helper that allocates and passes the view for you.

- **Assuming `_` means "any index."** It means **omit** this **chunk** axis, not "infer a loop variable." The surrounding **`foreach`** and **`parallel`** still determine which threads run.

---

## Summary

- **TMA** on Hopper offloads multi-dimensional tile copies to dedicated hardware, reducing the instruction and thread burden compared to purely software DMA loops.
- **`tma.copy.swiz<N>`** performs such a copy into **shared memory** using a **swizzle** pattern (commonly **128** bytes for FP16 tiles sized to match), reducing **bank conflicts** for subsequent accesses.
- **`global f16 [M, K]`** and symbolic **`M`, `N`, `K`** express **dynamic** problem sizes; **`void`** plus an **`output`** parameter writes results **in place**.
- **`cdiv`** is **ceiling division** for safe tiling when dimensions are not multiples of tile size.
- **`subspan(...).at(i, j)`** builds an explicitly sized **view** and selects a tile; **`chunkat`** divides tensors along indices; **`_`** skips tiling on one dimension.
- **`shared f16 [...] name;`** declares **SMEM** arenas that TMA and MMA share.
- **`: group-4`** marks a **warp-group** parallel region, matching Hopper WGMMA execution width.

In [Chapter 5](ch05-mma.md), we focus on **MMA** itself: what the accumulators represent, how layouts connect to PTX, and how to reason about correctness and performance without drowning in hardware manuals.
