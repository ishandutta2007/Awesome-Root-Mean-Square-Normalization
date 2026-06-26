# The Single-Pass Memory Advantage

A major performance bottleneck in training modern Large Language Models is not the computational arithmetic (FLOPs) but the memory bandwidth (HBM/DRAM reads and writes). The **Single-Pass Memory Advantage** of RMSNorm addresses this by reducing memory operations.

---

## 1. The GPU Memory Bottleneck: LayerNorm vs. RMSNorm

GPU execution speed is often bound by how quickly data can be transferred from high-bandwidth global memory (HBM) to on-chip SRAM registers.

*   **LayerNorm (Two-Pass):**
    1. **Pass 1:** Read inputs from global memory, compute the mean $\mu$, and discard or write temporary results.
    2. **Pass 2:** Read inputs again from global memory, compute the variance $\sigma^2$ using the mean, compute the normalized values, and write them back.
    This structure requires multiple roundtrips to GPU global memory.

*   **RMSNorm (Single-Pass):**
    Computes the squares of elements progressively and aggregates them. The normalization factor (RMS) can be computed in a **single pass** over the input tensor, immediately scaling and outputting the result.

---

## 2. Memory Flow Comparison

```mermaid
graph TD
    subgraph LayerNorm (Two-Pass)
        Read1[Read x from DRAM] --> ComputeMean[Compute Mean]
        ComputeMean --> Temp[Write Mean to DRAM/SRAM]
        Temp --> Read2[Read x & Mean from DRAM]
        Read2 --> ComputeVar[Compute Variance]
        ComputeVar --> Normalize[Normalize & Scale]
        Normalize --> Write1[Write y to DRAM]
    end

    subgraph RMSNorm (Single-Pass)
        ReadRMS[Read x from DRAM] --> ComputeRMS[Compute RMS in Register]
        ComputeRMS --> ScaleRMS[Scale and Normalize in Register]
        ScaleRMS --> WriteRMS[Write y to DRAM]
    end
```

---

## 3. Real-world Impact
By eliminating the two-pass requirement, RMSNorm cuts the memory bandwidth usage of the normalization kernel in half. In memory-bound workloads (such as autoregressive generation/inference), this results in significant wall-clock speedups.

---

[← Back to README](../README.md)
