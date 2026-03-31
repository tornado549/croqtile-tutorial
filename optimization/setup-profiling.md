# Setting Up: TimerOption, TFLOPS, and HW Efficiency

This page is the preamble for the Level 1 performance optimization guide. Before you read the case studies, you need a repeatable way to **measure** kernels: how Choreo times code, how to turn milliseconds into TFLOPS, and how to compare results against hardware peak. The matmul benchmarks in the Choreo tree all follow the same pattern, so once you understand this setup, every example speaks the same language.

## How `choreo::timing` Works

Benchmarks wrap the kernel under test in `choreo::timing` and pass a `choreo::TimerOption`:

- **`warmup`** — number of runs that are executed but **not** included in the average. This lets caches, TLBs, and steady-state behavior settle before measurement.
- **`repeat`** — number of timed iterations. The reported value is the **mean** elapsed time over these runs, in **milliseconds**.

Typical host code reads optional overrides from the environment, then calls timing on a lambda that launches the kernel and synchronizes the device:

```cpp
int warmup = 10;
int repeat = 500;
const char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP");
const char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT");
if (warmup_env) { int value = std::atoi(warmup_env); if (value >= 0) warmup = value; }
if (repeat_env) { int value = std::atoi(repeat_env); if (value > 0) repeat = value; }
choreo::TimerOption topt;
topt.warmup = warmup;
topt.repeat = repeat;
auto avg_ms = choreo::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
std::cout << "Timing avg ms: " << avg_ms << "\n";
```

The lambda should include whatever you need for a fair wall-clock measurement — usually the kernel launch plus `cudaDeviceSynchronize()` so the GPU has finished before the timer stops.

## From Average Milliseconds to TFLOPS

For a **dense** matrix multiply `C = A × B` with shapes `(M, K)` and `(K, N)`, the canonical floating-point operation count is:

`FLOPs = 2 * M * N * K`

Each output element involves `K` multiply-add pairs, hence the factor of two. **Sparse** variants (for example structured sparsity) use the same idea but count **effective** multiply-adds for the nonzeros; when the benchmark already defines a FLOP count, **double** it if the formula counts MACs as one op instead of two — match whatever the specific `matmul_*` host program documents.

Given average time `avg_ms`, convert seconds to TFLOPS:

`TFLOPS = (FLOPs / (avg_ms / 1000.0)) / 1e12`

In code:

```cpp
double flops = 2.0 * double(M) * double(N) * double(K);
double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
std::cout << "TFLOPS: " << tflops << "\n";
```

## Hardware Efficiency (Percent of Peak)

Throughput alone is hard to interpret without a ceiling. Benchmarks often print **efficiency** as TFLOPS divided by a documented peak for the GPU and precision, times 100:

```cpp
double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
std::cout << "HW efficiency: " << eff << "%\n";
```

Reference peaks used in the Choreo matmul benchmarks for **H800 PCIe**:

| Constant | Value (TFLOPS) | Use |
| -------- | ------------- | --- |
| `H800_PCIE_PEAK_F16_TFLOPS` | 1513 | FP16 dense / similar workloads |
| `H800_PCIE_PEAK_F8_TFLOPS` | 3026 | FP8 dense / similar workloads |

These are **theoretical** peaks; real kernels rarely sit at 100%. The point is to compare **before and after** an optimization on the same formula and peak constant, not to treat the percentage as an absolute grade.

## Environment Variables

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `CHOREO_TIMING_WARMUP` | `10` | Warmup iterations (non-negative; `0` means no warmup). |
| `CHOREO_TIMING_REPEAT` | `500` | Timed iterations (must be positive to take effect). |
| `CHOREO_DISABLE_TIMING` | unset | Set to `1` to skip timing (useful when you only want compile or correctness checks). |
| `CHOREO_SKIP_VERIFY` | unset | Set to `1` to skip numerical verification (see below). |

## Compile and Run Workflow

Performance `.co` files are built and executed through the Choreo driver. A typical pattern:

1. Invoke `choreo` with **`-gs`** (generate script), **`-t cute`**, the target **`-arch`**, and any codegen flags.
2. Point **`-o`** at a shell script path (often under `/tmp`).
3. Run that script with **`--execute`** to compile and run the generated host + device code.

Example:

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/matmul/matmul_f16_dyn_sm90.co \
  -o /tmp/matmul.cute.result && bash /tmp/matmul.cute.result --execute
```

Adjust the `.co` path and architecture for your benchmark. The important habit is: **same flags for A/B comparisons** — change one knob at a time when you are hunting regressions.

## Common Codegen Flags

These flags show up frequently in SM90-class matmul builds. Exact semantics live in Choreo’s CLI help and docs; treat this table as a **quick map** of what to look for when reading benchmark command lines.

| Flag | Role (high level) |
| ---- | ----------------- |
| `--use-warpspec` | Enable warp-specialized codegen paths where applicable. |
| `--stmatrix` | Use `stmatrix`-related lowering for shared-memory matrix fragments where supported. |
| `--hoist-offset` | Hoist offset calculations to reduce per-iteration work. |
| `--hoist-scale` | Hoist scale-related address or indexing work similarly. |
| `--ptx-barrier` | Influence barrier lowering in the PTX pipeline. |
| `--tma-cluster-aware` | Cluster-aware behavior for TMA-oriented kernels. |
| `--wgmma-wait-depth=N` | Tune wait-depth for WGMMA pipelines (`N` is an integer). |

Not every benchmark needs every flag. Copy the flags from the reference `matmul_*` recipe you are reproducing, then vary them deliberately.

## Verification vs Timing-Only Runs

Correctness checks compare the kernel output against a reference implementation. They add host-side work and can dominate short dev cycles. For **timing-focused** iterations where you already trust the math, set:

```bash
export CHOREO_SKIP_VERIFY=1
```

so the run measures the kernel path without verification overhead. Turn verification back on when you change data layout, precision, or tiling — a fast wrong answer is still wrong.

---

With `TimerOption`, TFLOPS, efficiency against peak, and the compile-and-run loop in place, you can read the optimization case studies as a series of controlled experiments: same measurement harness, different kernels and flags.
