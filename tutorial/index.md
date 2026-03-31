# Choreo Tutorial

Welcome to the Choreo tutorial. This guide walks you through writing high-performance GPU kernels using Choreo, starting from scratch and building up to production-grade patterns.

Each chapter introduces a small set of new concepts by evolving a running example. By the end, you will have encountered every major Choreo construct in a concrete, working program.

## Chapters

1. [Hello Choreo: From Zero to Running Kernel](ch01-hello-choreo.md)
2. [Data Movement Basics: Moving Data Blocks as a Whole](ch02-data-movement.md)
3. [Overlapping Compute and DMA: Pipeline Patterns](ch03-pipeline.md)
4. [High-Performance Data Movement: TMA and Swizzle](ch04-tma-swizzle.md)
5. [Enable Tensor Cores in One Primitive: the mma Operations](ch05-mma.md)
6. [Async Pipelining: inthreads, Events, and Warp Roles](ch06-warpspec.md)
7. [Persistent Matmul: if-guards, step, and Tile Iteration](ch07-persistent.md)
8. [Multi-Warpgroup Matmul: Scaling with Multiple Accumulators](ch08-multi-warpgroup.md)
9. [Beyond chunkat: view and from for Irregular Access](ch09-view-from.md)
10. [C++ Inline and Macros: Interfacing with Low-Level Control](ch10-cpp-inline-macros.md)

## Prerequisites

- Basic C++ knowledge (functions, pointers, arrays)
- Familiarity with GPU programming concepts (threads, blocks, shared memory)
- A built Choreo compiler (see [Getting Started](../documentation/getting-started-with-choreo.src.md))
