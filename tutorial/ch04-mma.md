# Tensor cores: the `mma` operations

Chapter 3’s tiled matmul still does the inner contraction the way a CPU would: a `foreach k` loop where each iteration multiplies scalars and accumulates into an output element. That loop is correct and readable, but it leaves the GPU’s **tensor cores** idle. Those units are built for something else entirely: a small matrix–matrix multiply on fixed tile sizes (commonly 16×16×16 for FP16) issued as one macro-operation, with throughput far beyond scalar FMA on the same chip.

In hardware, a tensor core is a dedicated matrix-multiply block: it takes operand tiles from registers, multiplies and accumulates in one shot, and hands results back through the same register path. A typical FP16 path operates on roughly 16×16×16 tiles per instruction family; exact geometry depends on datatype and architecture.

Raw CUDA exposes this through **WMMA-style APIs**: you declare `wmma::fragment` objects with specific shapes, call `load_matrix_sync`, `mma_sync`, and `store_matrix_sync`, and you babysit row- versus column-major variants and staging so every lane’s register mapping matches what the ISA expects. It works, but the ceremony is easy to get wrong.

Croktile wraps the same capability in a **four-step lifecycle** you can read at a glance: **fill** the accumulator, **load** operand tiles from shared memory, **multiply**–accumulate, then **store** the result back to shared memory. That lifecycle is the backbone of this chapter — first on SM86 (Ampere), where one warp drives one MMA, then on SM90 (Hopper), where four warps form a warpgroup.

![MMA lifecycle: fill, load, multiply, store — registers vs shared memory](../assets/images/ch04/fig1_mma_lifecycle_dark.png#only-dark)
![MMA lifecycle: fill, load, multiply, store — registers vs shared memory](../assets/images/ch04/fig1_mma_lifecycle_light.png#only-light)

*One pass through steps 2–3 consumes a K-slice; repeat for the full contraction, then step 4 writes the accumulated tile.*

## The MMA lifecycle

Every tensor-core matmul follows the same rhythm:

1. **`mma.fill 0.0`** — allocate an accumulator tile `mc` in registers and zero it.
2. **`mma.load`** — bring A and B operand tiles from shared memory into MMA operand registers `ma` and `mb`.
3. **`mma.row.row mc, ma, mb`** — issue the tensor-core instruction: **C += A × B** into `mc`.
4. **`mma.store mc, dst`** — write `mc` from registers into shared memory (and from there you usually `dma.copy` to global).

You loop steps 2–3 over K, reusing the same `mc`, then run step 4 once to flush the completed output tile. The names `mc`, `ma`, and `mb` are opaque register tiles — you do not declare per-lane layouts; the compiler derives them from the target architecture and your layout choice (`row.row` here).

## SM86 (Ampere): one warp, one MMA tile

On SM86, tensor-core MMA is scoped to a **single warp** (32 threads). In Croktile, that is `: group` — one cooperative thread group the size of one warp.

![Ampere vs Hopper MMA cooperation scope](../assets/images/ch04/fig2_sm86_vs_sm90_dark.png#only-dark)
![Ampere vs Hopper MMA cooperation scope](../assets/images/ch04/fig2_sm86_vs_sm90_light.png#only-light)

*SM86: one warp per MMA. SM90: four warps (`: group-4`) cooperate on WGMMA.*

Here is a complete FP16 matmul kernel for SM86. Tile sizes match the Croktile benchmark defaults: all `MATMUL_*` constants are 16, so one block tile equals one MMA tile along M and N.

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_TILE_N)] : block {
    shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s;

    parallel {warp_m, warp_n} by [cdiv(MATMUL_TILE_M, MATMUL_MMA_M), cdiv(MATMUL_TILE_N, MATMUL_MMA_N)] : group {
      mc = mma.fill 0.0;

      foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
        lhs_load_s = dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => shared;
        rhs_load_s = dma.copy rhs.subspan(MATMUL_TILE_N, MATMUL_TILE_K).at(block_n, iv_k) => shared;

        foreach iv_warp_k in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)] {
          ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
          mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
          mma.row.row mc, ma, mb;
        }
      }

      mma.store mc, output_s.subspan(MATMUL_MMA_M, MATMUL_MMA_N).at(warp_m, warp_n);
    }

    dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_TILE_N).at(block_m, block_n);
  }
}
```

**`__co__ void` and in-place output.** The kernel returns nothing; results go through `output`. That matches the usual GPU pattern of writing through a global pointer.

**Block grid.** `cdiv(M, MATMUL_TILE_M)` is ceiling division — how many tiles along M, including partial tiles. `block_m` and `block_n` pick which output tile this block owns.

**Shared staging.** `output_s` holds the block’s result tile before a single `dma.copy` pushes it to global memory.

**Warp grid and `mma.fill`.** `parallel {warp_m, warp_n} ... : group` maps MMA tiles to warps. With all dimensions 16, extents are 1×1 — one warp covers the whole block tile. Wider block tiles would add warps, each with its own `mc`.

**K loop and DMA.** Each `iv_k` stage pulls A and B strips into `lhs_load_s` / `rhs_load_s` via `dma.copy` with `subspan(...).at(...)`. Chapter 7 goes deeper on `subspan` versus `chunkat`.

**Operand loads.** `mma.load` moves the warp’s tile from shared memory into `ma` / `mb`. `chunkat(warp_m, iv_warp_k)` selects the M×K slice for this warp and inner-K step.

**The MMA opcode.** `mma.row.row mc, ma, mb` is the tensor-core multiply-accumulate. **`row.row` is a layout contract** — both operands are interpreted as row-major in the MMA register format. The wrong variant is a correctness bug, not a perf hint.

**Store and epilogue.** After K completes, `mma.store` writes `mc` into the warp’s sub-rectangle of `output_s`, then `dma.copy` sends the full block tile to global memory.

## SM90 (Hopper): WGMMA and warp groups

Hopper adds **Warpgroup Matrix Multiply-Accumulate (WGMMA)**: the same **C += A × B** idea, but issued cooperatively by **four warps** (128 threads). In Croktile that wider scope is `: group-4` instead of `: group`.

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;

  mc = mma.fill.f16 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      parallel p by 1 : group-4 {
        ma = mma.load lhs_load_s.chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
  mma.store mc, output_s;
  dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

**Same four steps.** Fill, load, multiply, store — the mental model does not change; the cooperation scope does.

**`mma.fill.f16`.** Hopper often spells accumulator precision explicitly — `.f16`, `.f32`, etc. FP16 operands with FP32 accumulation is a common pattern for long K. SM86 commonly uses the shorter `mma.fill 0.0` and relies on inference.

**`parallel p by 1 : group-4`.** One warpgroup (four warps) executes the inner loads and MMA. The mnemonic `mma.row.row` matches Ampere, but the hardware issue is wider.

**`chunkat(_, iv_warp)`.** `_` means “do not tile that dimension here” — keep the full M (or N) extent already resident in shared memory for this block; only K is subdivided per MMA slice.

| Aspect | SM86 (Ampere) | SM90 (Hopper) |
|--------|---------------|---------------|
| Thread scope | One warp — `: group` | Four warps — `: group-4` |
| Accumulator init | `mma.fill 0.0` | `mma.fill.f16 0.0f` (precision suffix) |
| Global → shared | `dma.copy` | Same (TMA appears in Chapter 7) |
| Core math | `mma.row.row mc, ma, mb` | Same mnemonic, wider hardware |
| Store | `mma.store` into per-warp tile | `mma.store` into warpgroup tile |

## Multi-warpgroup MMA

Chapter 3 introduced `parallel p1 by 2 : group-4` — two warpgroups in one block. With MMA, both groups can share the same B tile while loading different rows of A:

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_TILE_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
  shared f16 [MATMUL_TILE_M, MATMUL_WARP_N] output_s;

  mc = mma.fill.f32 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    parallel p1 by 2 : group-4 {
      foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
        ma = mma.load lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0).chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  parallel p1 by 2 : group-4 {
    mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
  }
  dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

**Splitting M with `p1`.** With `MATMUL_TILE_M = 128` and `MATMUL_WARP_M = 64`, the block spans 128 rows; `p1` selects the upper or lower 64-row strip. `lhs_load_s.subspan(MATMUL_WARP_M, ...).at(p1, 0)` gives each warpgroup its A rows; both use the same `rhs_load_s`.

**Mirrored store.** `mma.store` targets `output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0)` so each warpgroup writes its half of the staging buffer, then one `dma.copy` emits the combined tile.

## `mma.row.row.scale`: block dequantization

The `mma.row.row mc, ma, mb` form above covers FP16×FP16. For **FP8** (`f8_e4m3` or `f8_e5m2`), accumulation is often FP32 while operands need **block scaling**: one scale per operand tile, stored beside the matrix data.

```choreo
mma.row.row.scale mc, ma, mb, sc_a, sc_b;
```

**Fused dequantization.** The multiply-accumulate matches `mma.row.row`, then each result element is scaled by `sc_a` and `sc_b` — no separate dequant kernel.

**Scales as operands.** You typically load `sc_a` and `sc_b` from metadata tensors; the compiler broadcasts per-tile scales to the right lanes.

## New syntax

| Syntax | Meaning |
|--------|---------|
| `mc = mma.fill 0.0` | Initialize an MMA accumulator tile to zero |
| `ma = mma.load src.chunkat(...)` | Load operand tile from shared into MMA registers |
| `mma.row.row mc, ma, mb` | C += A × B on tensor cores (row-major operands) |
| `mma.store mc, dst` | Write accumulator tile from registers to shared |
| `mma.fill.f16 0.0f` / `mma.fill.f32 0.0f` | Accumulator with explicit precision |
| `mma.row.row.scale mc, ma, mb, sc_a, sc_b` | MMA with per-tile block dequantization (FP8 + scales) |
| `cdiv(a, b)` | Ceiling division: number of tiles, rounding up |
| `__co__ void fn(...)` | Kernel that writes results in-place (no return value) |
| `subspan(M, K).at(i, j)` | View with explicit tile extents, selected by index |
| `chunkat(_, iv_warp)` | `_` wildcard: no tiling on that dimension |

## Chapter summary

| Topic | Takeaway |
|-------|----------|
| Why tensor cores | Fixed small tiles per instruction; far higher throughput than scalar inner loops |
| Raw CUDA cost | Fragments, sync loads/stores, layout discipline — easy to misconfigure |
| Croktile lifecycle | **fill → load → multiply → store**; loop load/multiply on K, store once |
| SM86 | `: group` — 32-thread warp, one MMA scope |
| SM90 | `: group-4` — 128-thread warpgroup, WGMMA |
| Layout | `mma.row.row` must match actual storage order |
| FP8 / scales | `mma.row.row.scale` fuses block dequantization with MMA |

Croktile maps `mc`, `ma`, and `mb` to the right register layouts and barriers — the `wmma::fragment` / manual sync story stays below the surface. You still own layout contracts, tile sizes that match the hardware MMA shape, and the choice of `: group` versus `: group-4`. Tensor cores and memory still take turns: when multiply is hot, loads are not, and the other way around. The [next chapter](ch05-branch-control.md) is about **warp specialization and conditional control** — different threads playing different roles so you can overlap data movement with compute instead of one strictly serial loop.
