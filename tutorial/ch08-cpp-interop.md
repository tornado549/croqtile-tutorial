# C++ Interop: Device Functions, Inline Code, and the Preprocessor

Every high-level language needs an escape hatch to the layer below. Python has `ctypes` and C extensions. Rust has `unsafe` blocks and `extern "C"`. Java has JNI. The reason is always the same: no matter how expressive the abstraction, some hardware feature, some legacy library, some performance-critical intrinsic lives outside the language's domain.

Croqtile's interoperability story has three parts:

1. **`__device__` functions** — Standard CUDA device functions that coexist with `__co__` functions in the same `.co` file. The compiler passes them through unchanged.
2. **`__cpp__`** — Inject verbatim C++ or PTX into the generated code for hardware-specific instructions the DSL does not emit.
3. **The preprocessor** — `#define`, `#if`, `#ifdef`, `#error` for compile-time configuration, the same role `#define` plays in C/C++ codebases.

So far the tutorial has built a stack of abstractions: tiles, parallelism, MMA, pipelines, events, TMA. That is the Croqtile you want to live in. This chapter is about the times you need to step outside.

![Croqtile kernel body with an embedded __cpp__ island for verbatim PTX/C++](../assets/images/ch08/fig1_escape_hatch_dark.png#only-dark)
![Croqtile kernel body with an embedded __cpp__ island for verbatim PTX/C++](../assets/images/ch08/fig1_escape_hatch_light.png#only-light)

## How `.co` files compile

A `.co` file can contain three kinds of code: host C++ (ordinary functions, `main()`), `__co__` functions (Croqtile-managed kernels), and `__device__` functions (standard CUDA device code). The Croqtile compiler processes each differently:

![.co file compilation flow: source -> compiler -> host C++ and device CUDA -> nvcc -> binary](../assets/images/ch08/fig2_compilation_flow_dark.png#only-dark)
![.co file compilation flow: source -> compiler -> host C++ and device CUDA -> nvcc -> binary](../assets/images/ch08/fig2_compilation_flow_light.png#only-light)

- **`__co__` functions** are transformed into `__global__` CUDA kernels with generated launch configurations, shared memory declarations, and register allocation.
- **`__device__` functions** are passed through to the device compilation unit **unchanged**. The Croqtile compiler does not rewrite them — they appear in the generated CUDA exactly as you wrote them.
- **Host code** (everything else) becomes the host-side C++ that sets up buffers, launches kernels, and handles I/O.

## `__device__` functions: CUDA alongside Croqtile

When an algorithm needs a helper function that operates at the warp or thread level — a custom reduction, a sorting network, a special math function — writing it as a standard CUDA `__device__` function is often the natural choice. Croqtile does not need to manage these; they are plain CUDA.

A `__co__` function can call a `__device__` function using the `call` keyword:

```choreo
template <int K>
__device__ void warp_topk(f32* vals, s32* idxs);

template <typename T>
__device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
  return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
}

__co__ void moe_topk(f32 [N_TOKENS, N_EXPERTS] scores,
                     s32 [N_TOKENS, K]& topk_ids,
                     f32 [N_TOKENS, K]& topk_scores,
                     int N_BLOCK) {
  parallel n by N_BLOCK : block {
    foreach m in [ |scores.span| / N_THREAD / N_BLOCK ] {
      shared_scores = dma.copy scores.chunkat(n#m, _) => shared;

      parallel gid by [N_THREAD / 32] : group {
        parallel tid by 32 : thread {
          score = shared_scores.data.span_as(|shared_scores.data.span|).at(gid # tid);

          local s32 [K] frag_idx{-1};
          local f32 [K] frag_val{-1.0f};
          frag_idx.at(0) = tid;
          frag_val.at(0) = score;

          call warp_topk<8>(frag_val, frag_idx);

          inthreads (tid == 0) {
            foreach k in K {
              topk_ids.at(n#m, gid#k) = frag_idx.at(k);
              topk_scores.at(n#m, gid#k) = frag_val.at(k);
            }
          }
        }
      }
    }
  }
}
```

**`__device__ void warp_topk<K>`** — A templated device function implementing warp-level top-K selection using shuffle instructions. It operates on raw pointers (`f32*`, `s32*`), not Croqtile spans.

**`__device__ T SHFL_XOR`** — A warp shuffle intrinsic wrapper. The `__forceinline__` and `__shfl_xor_sync` are standard CUDA — Croqtile passes them through.

**`call warp_topk<8>(...)`** — The `call` keyword invokes a `__device__` function from within a `__co__` body. Arguments are passed by pointer; the compiler handles the address translation between Croqtile spans and raw pointers.

**`inthreads (tid == 0)`** — Only thread 0 in each warp writes back results. Note this uses `inthreads` without `.async` — a sequential thread filter, not a concurrent region.

This pattern — Croqtile handles the parallelism, tiling, and memory orchestration; `__device__` functions handle the per-warp algorithm — is common in production kernels like MoE (mixture-of-experts) top-K, custom reductions, and specialized math.

## `__cpp__`: verbatim C++ injection

`__cpp__` takes a string literal and pastes it, character for character, into the generated CUDA file. Whatever you place there must be valid at the splice point. The Croqtile compiler does not parse or rewrite the contents.

**Two forms:**

- **`__cpp__("...")`** — Ordinary string; best for short one-liners.
- **`__cpp__(R"(...)")`** — Raw string literal; use for `asm volatile` where escaping every `"` would be painful.

### Register hints: `setmaxnreg`

The canonical `__cpp__` use case is register redistribution in warp-specialized pipelines ([Chapter 5](ch05-branch-control.md)). The producer warpgroup is register-light (mostly TMA loads), while the consumer is register-heavy (MMA accumulators). PTX's `setmaxnreg` instruction moves the register budget:

```choreo
parallel p1 by 2 : group-4 {
  inthreads.async (p1 == 0) {
    __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      // ... TMA loads ...
    }
  }

  inthreads.async (p1 == 1) {
    __cpp__(R"(asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");)");
    mc = mma.fill.f16 0.0f;
    // ... WGMMA compute ...
  }
}
```

**Placement** — Put the hint at the top of each `inthreads.async` branch, before the heavy loop.

### Early returns and guards

MoE-style kernels often process variable-sized expert segments; some launches have zero width:

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

The identifiers must match what the surrounding generated code declares.

### Macros inside `__cpp__` strings

A common mistake: assuming the preprocessor expands macros inside string literals.

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

This fails — the preprocessor does not expand inside strings. Use numeric literals inside `__cpp__` and `#if` / `#error` outside to enforce consistency:

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## The preprocessor

Croqtile's preprocessor runs before the main compiler pass. Macros expand in both `__co__` regions and host C++ in the same `.co` file, so one definition keeps tile geometry and host-side checks aligned.

| Directive | Role |
|-----------|------|
| `#define NAME value` | Object-like macro: textual replacement |
| `#if` / `#elif` / `#else` / `#endif` | Conditional inclusion |
| `#ifdef` / `#ifndef` | Shorthand for whether a macro is defined |
| `#error message` | Force a compile-time failure with a message |

Function-like macros (`#define MAX(a, b) ...`) are not supported. Use `constexpr` helpers in ordinary C++.

### Tile geometry as macros

Production matmul sources centralize tile dimensions at the top:

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

The same names appear in `parallel`, shared-memory declarations, `foreach` bounds, and host-side verification.

### Compile-time assertions with `#error`

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

Treat these guards as documentation of hardware constraints.

### Conditional code paths

```choreo
#define PATH0

__co__ foo() {
  #ifdef PATH0
    // path 0 code
  #else
    // path 1 code
  #endif
}
```

The preprocessor keeps one branch and discards the other before Croqtile parses the `__co__` body. Command-line defines: `croqtile kernel.co -DMATMUL_TILE_K=128`.

## Reading a production `.co` file

When you open a benchmark kernel, read top down:

1. **Macros and `#error` guards** — The contract: allowed tile sizes, swizzle rules, architecture flags.
2. **`__device__` functions** — Helper algorithms (top-K, reductions, shuffle wrappers) that Croqtile passes through.
3. **Host setup** — Buffers, launch configuration, timing; ordinary C++.
4. **The `__co__` function** — Orchestration: `parallel`, `foreach`, TMA/MMA, `inthreads.async`, events. Map each region back to earlier chapters.
5. **`__cpp__` islands** — Usually a handful of lines. Pause on each and ask what the hardware receives that the DSL does not spell.

## Chapter summary

| Topic | Takeaway |
|-------|----------|
| `__device__` functions | Standard CUDA device code in `.co` files; passed through unchanged; called with `call` from `__co__` |
| `__cpp__` | Verbatim paste into generated CUDA; raw strings for `asm`; names must match generated C++ |
| `call` keyword | Invoke `__device__` functions from `__co__` bodies |
| `setmaxnreg` | Register redistribution: `dec` on producer, `inc` on consumer |
| Preprocessor | `#define` for tile geometry; `#if` / `#error` for constraints; `#ifdef` for variants; `-D` for sweeps |

**New syntax**

| Syntax | Meaning |
|--------|---------|
| `__device__ fn()` | Standard CUDA device function (pass-through) |
| `call fn(args)` | Invoke a `__device__` function from `__co__` |
| `__cpp__("...")` | Inject verbatim C++ (ordinary string) |
| `__cpp__(R"(...)")` | Inject verbatim C++ (raw string literal) |
| `#define NAME value` | Object-like macro |
| `#if expr` / `#elif` / `#else` / `#endif` | Conditional compilation |
| `#ifdef NAME` / `#ifndef NAME` | Test whether a macro is defined |
| `#error "message"` | Compile-time assertion failure |

The [next chapter](ch09-debug-verbose.md) turns to the other side of the workflow: what to do when the kernel compiles but the output is wrong — debugging, verbose modes, and systematic narrowing.
