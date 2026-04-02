# Debug and Verbose: Printing, Assertions, and GDB

GPU kernels are **opaque systems**. Thousands of threads execute concurrently, shared memory is invisible from the host, and when results are wrong, there is no stack trace pointing to the bad line. This opacity is not unique to GPUs — any system where the programmer cannot directly observe intermediate state faces the same challenge: distributed systems (where is the lost message?), embedded firmware (which interrupt handler corrupted the register?), optimizing compilers (which pass broke the semantics?).

The universal debugging discipline is the same everywhere: **systematically narrow the search space**, from cheap checks to expensive ones. Start with static analysis and compile-time assertions (free, catches whole classes of bugs). Move to targeted runtime probes (cheap, localizes the problem). Resort to interactive debuggers only when the cheaper tools have narrowed the suspect list.

Croqtile provides tools at each level: **compile-time shape printing** (`print!`, `println!`) to verify tile dimensions without launching a kernel, **runtime assertions** (`assert`) for invariant checks, **guarded device `println`** for runtime inspection of specific threads, **debug RTTI** for `cuda-gdb`, and a **verbose mode** (`-v`) to see what the compiler is actually doing under the hood.

## `print!` and `println!`: compile-time inspection

The bang variants (`print!`, `println!`) run at **compile time**, not at runtime. They print to the compiler's output and are useful for inspecting shapes, extents, and type information without launching a kernel:

```choreo
__co__ void check_shapes(f32 [3, 2] b) {
  print!("shape of b: ");
  println!("b.span = ", b.span);
}
```

When you compile this file, the compiler emits:

```
shape of b: b.span = {3, 2}
```

No GPU needed. String literals are concatenated (`"wor" "ld!"` becomes `"world!"`). Use `print!` / `println!` to verify that `chunkat` and `subspan` produce the tile sizes you expect before you spend time on a full kernel launch.

## `print` and `println`: runtime device output

Without the bang, `print` and `println` emit device-side `printf` calls in the generated CUDA:

```choreo
__co__ void inspect(s32 [4, 8] data) {
  foreach i in [data.span] {
    println("element ", i, " = ", data.at(i));
  }
}
```

Each call takes a comma-separated mix of string literals and expressions. The output order across threads is **nondeterministic** — GPU `printf` buffers are flushed asynchronously.

**Guarding output.** For a specific check, guard the print with a condition:

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
    // ... compute ...
    if (px == 0 && py == 0 && qx == 3 && qy == 5) {
      println("partial sum = ", accum);
    }
  }
```

Without the guard you get one line per thread — thousands of lines for a large grid.

## `assert`: runtime invariant checks

Croqtile's `assert` builtin checks an invariant at runtime and aborts the kernel with a message if it fails:

```choreo
assert(stage < MATMUL_STAGES, "stage index out of bounds");
```

On the device, this compiles to a `printf` of the message followed by an abort. Use assertions to catch index-out-of-range, null pointer, and other "this should never happen" conditions early.

## Verbose mode: `-v`

Pass `-v` (or `--verbose`) to the compiler to see the external commands it invokes — which preprocessor, which code generator, which `nvcc` invocation:

```bash
croqtile kernel.co -v -o kernel
```

This is useful when you suspect the compiler is passing wrong flags to `nvcc` or when you need to see the exact generated file paths for manual inspection.

## Runtime checks: `-rtc`

The compiler supports graduated runtime checking with `-rtc` (or `--runtime-check`):

```bash
croqtile kernel.co -rtc high -o kernel
```

Levels: `entry`, `low`, `medium`, `high`, `all`, `none`. Higher levels insert more bounds checks and validation at the cost of performance. Use `high` or `all` during development, `none` for production.

## Debugging MMA-heavy kernels

When the wrong answer comes from a tensor-core path, suspect **layout first** — row versus column major, and whether the RHS is `[N, K]` versus `[K, N]` in memory. Then check **indexing** (which `block_m`, `block_n`, and K slice you attached with `.at` / `chunkat`). Then check **async ordering** if you introduced asynchronous copies or split producer and consumer warps.

A common mistake is mislabeling `mma.row.row` when the staged data is actually column-major, or using `chunkat` indices that do not align with the MMA tile geometry. If you use swizzled loads (`tma.copy.swiz` / `mma.load.swiz`), a mismatch shows up as a regular pattern in the error (e.g., every sixteenth element correct, the rest shifted).

## Debug RTTI and `cuda-gdb`

When `print` / `println` is not enough, Croqtile supports debug RTTI (Runtime Type Information) that makes its types visible to `cuda-gdb`.

Compile with `-g` to enable debug symbols:

```bash
croqtile kernel.co -g -o kernel_debug
```

The generated code includes RTTI structs for Croqtile types:

| Croqtile type | GDB type | Fields |
|--------------|----------|--------|
| Shaped tensor (`s32 [M, N]`) | `choreo::rtti::spanned<int, 2>` | `.span.data[]` (dimensions), `.stride.data[]` (strides), `.data` (pointer) |
| Index tuple | `choreo::rtti::bounded_ituple<N>` | `.data[]` (values), `.ub[]` (upper bounds) |
| Integer tuple | `choreo::rtti::ituple<N>` | `.data[]` (values) |
| Multi-dim span | `choreo::rtti::mdspan<N>` | `.data[]` (extents) |

**Example session:**

```bash
cuda-gdb -q ./kernel_debug
(gdb) break my_kernel
(gdb) run
(gdb) ptype __dbg_lhs
type = struct choreo::rtti::spanned<int, 2>
(gdb) print __dbg_lhs.span.data[0]
$1 = 32
(gdb) print __dbg_lhs.span.data[1]
$2 = 64
(gdb) print __dbg_lhs.data != 0
$3 = true
```

The `__dbg_` prefix on variable names is generated by the compiler — it makes Croqtile variables visible alongside the generated C++ intermediates.

## Putting it together: a systematic workflow

The ordering is deliberate: **cheap compile-time checks first**, then **narrowed runtime prints**, then **pipeline semantics**, **layout**, and finally **the debugger**.

![Debugging workflow: shapes -> one tile -> sync -> layout -> GDB](../assets/images/ch09/fig1_debug_workflow_dark.png#only-dark)
![Debugging workflow: shapes -> one tile -> sync -> layout -> GDB](../assets/images/ch09/fig1_debug_workflow_light.png#only-light)

**1. Check shapes.** Use `println!` at compile time to verify that all `chunkat`, `subspan`, and `span` expressions produce the tile sizes you expect.

**2. Check one tile.** Add a guarded `println` inside the K loop that fires only for block `(0, 0)` and one thread. Print the accumulator after each K iteration and compare with a hand-computed reference.

**3. Check synchronization.** If the value is wrong only at large K, suspect event ordering. Print `iv_k` and `stage` in both producer and consumer to verify they visit the same sequence. A common bug: the consumer's `trigger empty[stage]` fires too early.

**4. Check layout.** If the result is wrong in a pattern (every sixteenth element correct, the rest shifted), suspect a row-major vs column-major mismatch in `mma.row.row` or a swizzle mode inconsistency between `tma.copy.swiz<N>` and `mma.load.swiz<N>`.

**5. Use GDB for pointer bugs.** If `println` shows values that make no sense (NaN, huge integer, zero when nonzero is expected), compile with `-g` and step through in `cuda-gdb`. Inspect the tensor pointer (`__dbg_x.data`).

**Performance caveat.** `print` / `println` are expensive: each call serializes through a global `printf` buffer and destroys throughput. Debug RTTI adds register pressure. Remove all prints and drop `-g` before benchmarking. A practical compromise is `#ifdef DEBUG_PRINT` (Chapter 8):

```choreo
#ifdef DEBUG_PRINT
  println("tile_k=", iv_k, " accum=", mc);
#endif
```

Compile with `croqtile kernel.co -DDEBUG_PRINT` to enable, or without the flag for production runs.

## Summary

| Tool | Level | Cost |
|------|-------|------|
| `print!` / `println!` | Compile-time | Free — no kernel launch |
| `assert(expr, "msg")` | Runtime | Low — aborts on violation |
| `print` / `println` | Runtime | Medium — serialized `printf` |
| `-rtc high` | Runtime | Medium — bounds checks |
| `-v` / `--verbose` | Compiler | Free — shows subprocess invocations |
| `-g` + `cuda-gdb` + RTTI | Runtime | High — debug symbols, no optimization |

You started from element-wise addition in Chapter 1, learned data movement (Chapter 2), parallelism (Chapter 3), tensor cores (Chapter 4), control flow (Chapter 5), pipelining (Chapter 6), TMA (Chapter 7), C++ interop (Chapter 8), and debugging (Chapter 9). The next step is to open a benchmark kernel from the `croqtile/benchmark/` directory, map its regions to the chapters here, change one constant, rebuild, and measure.
