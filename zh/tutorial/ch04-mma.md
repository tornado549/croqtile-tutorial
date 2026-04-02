# 张量核心：`mma` 运算

第 3 章中的分块矩阵乘法在内层收缩上仍采用类 CPU 的做法：通过 `foreach k` 循环，每次迭代对标量相乘并将结果累加到输出元素上。该循环在语义上正确且易读，但会使 GPU 的**张量核心（tensor core）**处于空闲状态。张量核心专为另一类工作而设计：在固定 tile 尺寸上（FP16 常见为 16×16×16）以单条宏指令完成小块矩阵–矩阵乘法，其吞吐远高于同一芯片上的标量乘加（FMA）。

在硬件层面，张量核心是一个专用的矩阵乘法单元：从寄存器取操作数 tile，一次性完成乘加并将结果经同一寄存器路径写回。典型的 FP16 路径每个指令族约处理 16×16×16 的 tile；具体几何形状取决于数据类型与架构。

原始 CUDA 通过 **WMMA 风格 API** 暴露这一能力：需声明特定形状的 `wmma::fragment` 对象，调用 `load_matrix_sync`、`mma_sync`、`store_matrix_sync`，并仔细处理行主序与列主序变体及暂存，使每条线程的寄存器映射符合 ISA 预期。该方法可行，但仪式繁琐且易出错。

鳄霸（Croktile）将同一能力封装为可一眼读懂的**四步生命周期**：**fill** 初始化累加器，**load** 从共享内存加载操作数 tile，**multiply**–**accumulate**，最后 **store** 将结果写回共享内存。该生命周期即本章主线——先在 SM86（Ampere）上由单个 warp 驱动一次 MMA，再在 SM90（Hopper）上由四个 warp 组成 warpgroup。

![MMA 生命周期：fill、load、multiply、store——寄存器与共享内存](../assets/images/ch04/fig1_mma_lifecycle_dark.png#only-dark)
![MMA 生命周期：fill、load、multiply、store——寄存器与共享内存](../assets/images/ch04/fig1_mma_lifecycle_light.png#only-light)

*完成步骤 2–3 的一次遍历消耗一段 K 切片；对完整收缩重复执行，最后由步骤 4 写出累加后的 tile。*

## MMA 生命周期

每个基于张量核心的矩阵乘法都遵循同一节奏：

1. **`mma.fill 0.0`** — 在寄存器中分配累加器 tile `mc` 并置零。
2. **`mma.load`** — 将 A、B 操作数 tile 从共享内存载入 MMA 操作数寄存器 `ma` 与 `mb`。
3. **`mma.row.row mc, ma, mb`** — 发出张量核心指令：在 `mc` 上执行 **C += A × B**。
4. **`mma.store mc, dst`** — 将 `mc` 从寄存器写入共享内存（之后通常再通过 `dma.copy` 拷到全局内存）。

在 K 维上对步骤 2–3 循环，复用同一 `mc`，K 完成后执行一次步骤 4 以刷出完整输出 tile。名称 `mc`、`ma`、`mb` 表示不透明寄存器 tile——无需逐线程声明布局；编译器根据目标架构与布局选择（此处为 `row.row`）推导。

## SM86（Ampere）：一个 warp，一块 MMA tile

在 SM86 上，张量核心 MMA 的作用域为**单个 warp**（32 条线程）。在鳄霸中对应 `: group`——大小为一个 warp 的协作线程组。

![Ampere 与 Hopper 的 MMA 协作范围](../assets/images/ch04/fig2_sm86_vs_sm90_dark.png#only-dark)
![Ampere 与 Hopper 的 MMA 协作范围](../assets/images/ch04/fig2_sm86_vs_sm90_light.png#only-light)

*SM86：每个 MMA 对应一个 warp。SM90：四个 warp（`: group-4`）协作执行 WGMMA。*

下面是一个完整的 SM86 FP16 矩阵乘法内核。tile 尺寸与鳄霸基准默认一致：所有 `MATMUL_*` 常量均为 16，故一个 block tile 在 M、N 方向各等于一块 MMA tile。

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

**`__co__ void` 与原地输出。** 内核无返回值；结果经 `output` 写出。这与通过全局指针写回的典型 GPU 模式一致。

**Block 网格。** `cdiv(M, MATMUL_TILE_M)` 为向上取整除法——沿 M 方向需要多少块 tile（含部分 tile）。`block_m` 与 `block_n` 指定本 block 负责的输出 tile。

**共享暂存。** `output_s` 保存 block 的结果 tile，再经一次 `dma.copy` 推送到全局内存。

**Warp 网格与 `mma.fill`。** `parallel {warp_m, warp_n} ... : group` 将 MMA tile 映射到 warp。当各维均为 16 时，范围为 1×1——单个 warp 覆盖整块 block tile；更宽的 block tile 会增加 warp 数，每个 warp 拥有各自的 `mc`。

**K 循环与 DMA。** 每个 `iv_k` 阶段通过 `dma.copy` 与 `subspan(...).at(...)` 将 A、B 条带拉入 `lhs_load_s` / `rhs_load_s`。第 7 章对 `subspan` 与 `chunkat` 有更深入讨论。

**操作数加载。** `mma.load` 将 warp 对应 tile 从共享内存移入 `ma` / `mb`。`chunkat(warp_m, iv_warp_k)` 选取本 warp 及内层 K 步对应的 M×K 切片。

**MMA 操作码。** `mma.row.row mc, ma, mb` 为张量核心乘加。**`row.row` 是布局约定**——两操作数在 MMA 寄存器格式中均按行主序解释。选错变体属于正确性错误，而非性能提示。

**存储与收尾。** K 完成后，`mma.store` 将 `mc` 写入 `output_s` 中该 warp 的子矩形，随后 `dma.copy` 将整个 block tile 发送到全局内存。

## SM90（Hopper）：WGMMA 与 warpgroup

Hopper 引入 **Warpgroup Matrix Multiply-Accumulate（WGMMA）**：同样是 **C += A × B**，但由**四个 warp**（128 条线程）协作发出。在鳄霸中，这一更宽的作用域使用 `: group-4` 而非 `: group`。

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

**四步不变。** Fill、load、multiply、store——心智模型不变；变化的是协作范围。

**`mma.fill.f16`。** 在 Hopper 上常显式写出累加器精度——`.f16`、`.f32` 等。FP16 操作数、FP32 累加是长 K 维的常见模式。SM86 多用较短形式 `mma.fill 0.0` 并依赖推断。

**`parallel p by 1 : group-4`。** 一个 warpgroup（四个 warp）执行内层 load 与 MMA。助记符 `mma.row.row` 与 Ampere 一致，但硬件发射更宽。

**`chunkat(_, iv_warp)`。** `_` 表示“该维此处不再分块”——对本 block 已驻留在共享内存中的完整 M（或 N）范围保持不动；仅沿 K 按 MMA 切片划分。

| 方面 | SM86（Ampere） | SM90（Hopper） |
|--------|---------------|---------------|
| 线程范围 | 一个 warp — `: group` | 四个 warp — `: group-4` |
| 累加器初始化 | `mma.fill 0.0` | `mma.fill.f16 0.0f`（精度后缀） |
| 全局 → 共享 | `dma.copy` | 相同（TMA 见第 7 章） |
| 核心运算 | `mma.row.row mc, ma, mb` | 相同助记符，硬件更宽 |
| 存储 | `mma.store` 写入各 warp 的 tile | `mma.store` 写入 warpgroup 的 tile |

## 多 warpgroup MMA

第 3 章介绍了 `parallel p1 by 2 : group-4`——一个 block 内两个 warpgroup。在 MMA 场景下，两组可共享同一块 B tile，同时加载 A 的不同行：

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

**沿 M 用 `p1` 划分。** 当 `MATMUL_TILE_M = 128` 且 `MATMUL_WARP_M = 64` 时，block 跨 128 行；`p1` 选取上或下 64 行条带。`lhs_load_s.subspan(MATMUL_WARP_M, ...).at(p1, 0)` 为每个 warpgroup 提供其 A 行；两者共用同一 `rhs_load_s`。

**对称存储。** `mma.store` 目标为 `output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0)`，使各 warpgroup 写入暂存缓冲区的一半，随后一次 `dma.copy` 发出合并后的 tile。

## `mma.row.row.scale`：block 反量化

上述 `mma.row.row mc, ma, mb` 形式适用于 FP16×FP16。对于 **FP8**（`f8_e4m3` 或 `f8_e5m2`），累加常为 FP32，而操作数需要 **block 缩放**：每个操作数 tile 一个缩放因子，与矩阵数据并列存放。

```choreo
mma.row.row.scale mc, ma, mb, sc_a, sc_b;
```

**融合反量化。** 乘加与 `mma.row.row` 一致，随后每个结果元素再按 `sc_a` 与 `sc_b` 缩放——无需单独的反量化内核。

**缩放作为操作数。** 通常从元数据张量加载 `sc_a` 与 `sc_b`；编译器将 per-tile 缩放广播到正确 lane。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `mc = mma.fill 0.0` | 将 MMA 累加器 tile 初始化为零 |
| `ma = mma.load src.chunkat(...)` | 将操作数 tile 从共享内存加载到 MMA 寄存器 |
| `mma.row.row mc, ma, mb` | 在张量核心上执行 C += A × B（行主序操作数） |
| `mma.store mc, dst` | 将累加器 tile 从寄存器写入共享内存 |
| `mma.fill.f16 0.0f` / `mma.fill.f32 0.0f` | 显式精度的累加器 |
| `mma.row.row.scale mc, ma, mb, sc_a, sc_b` | 带 per-tile block 反量化的 MMA（FP8 + 缩放） |
| `cdiv(a, b)` | 向上取整除法：tile 数量，向上舍入 |
| `__co__ void fn(...)` | 原地写回结果的内核（无返回值） |
| `subspan(M, K).at(i, j)` | 显式 tile 范围的视图，按索引选取 |
| `chunkat(_, iv_warp)` | `_` 通配符：该维不分块 |

## 本章小结

| 主题 | 要点 |
|-------|----------|
| 为何使用张量核心 | 每条指令固定小 tile；吞吐远高于标量内层循环 |
| 原始 CUDA 代价 | fragment、同步 load/store、布局约束——易配置错误 |
| 鳄霸生命周期 | **fill → load → multiply → store**；在 K 上循环 load/multiply，store 一次 |
| SM86 | `: group`——32 线程 warp，单一 MMA 作用域 |
| SM90 | `: group-4`——128 线程 warpgroup，WGMMA |
| 布局 | `mma.row.row` 须与实际存储顺序一致 |
| FP8 / 缩放 | `mma.row.row.scale` 将 block 反量化与 MMA 融合 |

鳄霸将 `mc`、`ma`、`mb` 映射到正确的寄存器布局与屏障——`wmma::fragment` / 手动同步的细节留在表层之下。你仍需负责布局约定、与硬件 MMA 形状匹配的 tile 尺寸，以及 `: group` 与 `: group-4` 的选择。张量核心与内存仍交替成为瓶颈：乘加热时，加载往往空闲，反之亦然。下一章 [ch05-branch-control.md](ch05-branch-control.md) 讨论 **warp 特化与条件控制**——不同线程承担不同角色，从而在数据搬运与计算之间重叠，而非严格串行单循环。
