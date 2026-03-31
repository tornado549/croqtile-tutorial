# Croktile

Documentation and tutorials for **Choreo**, a C++ embedded DSL for DMA orchestration in high-performance kernel programming.

## Structure

### [Tutorial](tutorial/index.md)

A step-by-step walkthrough for newcomers. Starts with the simplest possible kernel and progressively introduces Choreo syntax through increasingly sophisticated matmul variants. No prior Choreo experience required — just basic C++ and GPU programming concepts.

### [Performance Tuning Demos](optimization/index.md)

Case-oriented performance engineering. Each section profiles a real kernel family, identifies bottlenecks, and applies optimization patterns to push toward hardware peak. Covers profiling, benchmarking, compiler flags, and AI-tuning workflows.

### [Coding Reference](documentation/index.md)

Exhaustive syntax and semantics reference. Every Choreo construct — shapes, DMA, loops, parallelism, events, memory, macros, and more — is documented in detail.

### [Design Rationales](advanced/index.md)

Deep-dive articles on Choreo's design rationale, tradeoffs, and frontier challenges. *(Planned)*
