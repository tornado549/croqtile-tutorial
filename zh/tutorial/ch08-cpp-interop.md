# C++ 互操作：`__device__` 函数、内联代码与预处理器

每一种高级语言都需要一条通往更底层机制的逃生通道。Python 有 `ctypes` 与 C 扩展，Rust 有 `unsafe` 块与 `extern "C"`，Java 有 JNI。原因总是一样的：无论抽象多么富有表现力，总有某些硬件特性、某些遗留库、某些性能攸关的内建能力处在该语言域之外。

鳄霸（Croqtile）的互操作叙事包含三个部分：

1. **`__device__` 函数** — 标准 CUDA device 函数，与 `__co__` 函数共存于同一 `.co` 文件中。编译器对它们原样透传。
2. **`__cpp__`** — 将逐字 C++ 或 PTX 注入到生成代码中，用于 DSL 无法发出的硬件相关指令。
3. **预处理器** — `#define`、`#if`、`#ifdef`、`#error`，用于编译期配置，与 C/C++ 代码库中 `#define` 所扮演的角色相同。

至此，本教程已堆叠起一整座抽象栈：tile、并行、MMA、流水线、事件、TMA。那是你希望在鳄霸中长期栖居的世界。本章讨论的是你需要暂时跨出去的那些时刻。

![Croqtile kernel body with an embedded __cpp__ island for verbatim PTX/C++](../assets/images/ch08/fig1_escape_hatch_dark.png#only-dark)
![Croqtile kernel body with an embedded __cpp__ island for verbatim PTX/C++](../assets/images/ch08/fig1_escape_hatch_light.png#only-light)

## `.co` 文件如何编译

`.co` 文件可包含三类代码：host C++（普通函数、`main()`）、`__co__` 函数（由鳄霸管理的 kernel），以及 `__device__` 函数（标准 CUDA device 代码）。鳄霸编译器对它们的处理方式各不相同：

![.co file compilation flow: source -> compiler -> host C++ and device CUDA -> nvcc -> binary](../assets/images/ch08/fig2_compilation_flow_dark.png#only-dark)
![.co file compilation flow: source -> compiler -> host C++ and device CUDA -> nvcc -> binary](../assets/images/ch08/fig2_compilation_flow_light.png#only-light)

- **`__co__` 函数** 被变换为带生成 launch 配置、共享内存声明与寄存器分配的 `__global__` CUDA kernel。
- **`__device__` 函数** 原样进入 device 编译单元，**不做改写**。鳄霸编译器不会重写它们 —— 它们在你生成的 CUDA 中与你书写时完全一致。
- **Host 代码**（其余一切）成为负责建立缓冲区、启动 kernel 与处理 I/O 的 host 侧 C++。

## `__device__` 函数：CUDA 与 Croqtile 并存

当算法需要在线程或 warp 粒度上运作的辅助函数 —— 例如自定义归约、排序网络、特殊数学函数 —— 将其写成标准 CUDA `__device__` 函数往往是自然之选。鳄霸无需管理这些；它们就是普通 CUDA。

`__co__` 函数可使用 `call` 关键字调用 `__device__` 函数：

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

**`__device__ void warp_topk<K>`** — 使用 shuffle 指令实现 warp 级 top-K 选择的模板 device 函数。它操作原始指针（`f32*`、`s32*`），而非 Croqtile spans。

**`__device__ T SHFL_XOR`** — warp shuffle 内建封装。`__forceinline__` 与 `__shfl_xor_sync` 为标准 CUDA —— Croqtile 原样透传。

**`call warp_topk<8>(...)`** — `call` 关键字在 `__co__` 主体内调用 `__device__` 函数。参数按指针传递；编译器负责在 Croqtile span 与原始指针之间完成地址转换。

**`inthreads (tid == 0)`** — 每个 warp 中只有线程 0 写回结果。注意此处使用不带 `.async` 的 `inthreads` —— 为顺序线程过滤，而非并发区域。

这一模式 —— Croqtile 负责并行、tiling 与内存编排；`__device__` 函数负责 per-warp 算法 —— 在 MoE（mixture-of-experts）top-K、自定义归约与专用数学等生产级 kernel 中十分常见。

## `__cpp__`：逐字 C++ 注入

`__cpp__` 接收一个字符串字面量，并逐字符粘贴到生成的 CUDA 文件中。置于该处的任何内容在拼接点都必须合法。鳄霸编译器既不解析也不改写其内容。

**两种形式：**

- **`__cpp__("...")`** — 普通字符串；最适合短单行。
- **`__cpp__(R"(...)")`** — 原始字符串字面量；在需要 `asm volatile` 且为每个 `"` 转义会很痛苦时使用。

### 寄存器提示：`setmaxnreg`

典型的 `__cpp__` 用例是 warp 特化流水线中的寄存器再分配（[第 5 章](ch05-branch-control.md)）。producer warpgroup 寄存器占用较轻（多为 TMA load），而 consumer 较重（MMA 累加器）。PTX 的 `setmaxnreg` 指令用于移动寄存器预算：

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

**放置位置** — 将提示放在每个 `inthreads.async` 分支的顶部、重循环之前。

### 提前返回与守卫

MoE 风格 kernel 常处理可变长度的 expert 段；某些 launch 宽度为零：

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

标识符须与周围生成代码所声明的一致。

### `__cpp__` 字符串内的宏

常见误区：以为预处理器会在字符串字面量内展开宏。

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

这会失败 —— 预处理器不会在字符串内展开。在 `__cpp__` 中使用数字字面量，并在其外使用 `#if` / `#error` 以强制一致性：

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## 预处理器

鳄霸的预处理器在主编译遍之前运行。宏在同一 `.co` 文件的 `__co__` 区域与 host C++ 中都会展开，因此一份定义即可使 tile 几何与 host 侧检查保持一致。

| 指令 | 作用 |
|-----------|------|
| `#define NAME value` | 类对象宏：文本替换 |
| `#if` / `#elif` / `#else` / `#endif` | 条件包含 |
| `#ifdef` / `#ifndef` | 宏是否已定义的简写 |
| `#error message` | 以消息强制编译期失败 |

不支持函数式宏（`#define MAX(a, b) ...`）。请在普通 C++ 中使用 `constexpr` 辅助函数。

### 以宏表示 tile 几何

生产级 matmul 源码通常在文件顶部集中定义 tile 尺寸：

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

相同名字出现在 `parallel`、共享内存声明、`foreach` 边界以及 host 侧校验中。

### 使用 `#error` 的编译期断言

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

将这些守卫视为对硬件约束的文档化说明。

### 条件代码路径

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

预处理器在鳄霸解析 `__co__` 主体之前保留一分支并丢弃另一分支。命令行定义：`croqtile kernel.co -DMATMUL_TILE_K=128`。

## 如何阅读生产级 `.co` 文件

打开基准 kernel 时，自上而下阅读：

1. **宏与 `#error` 守卫** — 契约：允许的 tile 尺寸、swizzle 规则、架构标志。
2. **`__device__` 函数** — Croqtile 原样透传的辅助算法（top-K、归约、shuffle 封装等）。
3. **Host 设置** — 缓冲区、launch 配置、计时；普通 C++。
4. **`__co__` 函数** — 编排：`parallel`、`foreach`、TMA/MMA、`inthreads.async`、事件。将每个区域映射回先前章节。
5. **`__cpp__` 岛** — 通常寥寥数行。在每一处停顿，追问硬件收到了哪些 DSL 未显式写出的内容。

## 本章小结

| 主题 | 要点 |
|-------|----------|
| `__device__` 函数 | `.co` 文件中的标准 CUDA device 代码；原样透传；在 `__co__` 中用 `call` 调用 |
| `__cpp__` | 逐字粘贴到生成的 CUDA；`asm` 宜用原始字符串；名称须与生成 C++ 一致 |
| `call` 关键字 | 从 `__co__` 主体内调用 `__device__` 函数 |
| `setmaxnreg` | 寄存器再分配：producer 上 `dec`，consumer 上 `inc` |
| 预处理器 | 用 `#define` 表示 tile 几何；`#if` / `#error` 表示约束；`#ifdef` 表示变体；`-D` 用于扫描 |

**新语法**

| 语法 | 含义 |
|--------|---------|
| `__device__ fn()` | 标准 CUDA device 函数（透传） |
| `call fn(args)` | 从 `__co__` 调用 `__device__` 函数 |
| `__cpp__("...")` | 注入逐字 C++（普通字符串） |
| `__cpp__(R"(...)")` | 注入逐字 C++（原始字符串字面量） |
| `#define NAME value` | 类对象宏 |
| `#if expr` / `#elif` / `#else` / `#endif` | 条件编译 |
| `#ifdef NAME` / `#ifndef NAME` | 测试宏是否已定义 |
| `#error "message"` | 编译期断言失败 |

[下一章](ch09-debug-verbose.md)转向工作流的另一面：kernel 已能编译但结果错误时该如何处理 —— 调试、verbose 模式与系统化缩小问题范围。
