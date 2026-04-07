# CUDA Programming - Concepts & Guide

This lab covers fundamental CUDA programming concepts, from device querying to parallel kernel execution on GPUs.

---

## Table of Contents

1. [CUDA Overview](#cuda-overview)
2. [Hardware Stack](#hardware-stack)
3. [Memory Model](#memory-model)
4. [Thread Hierarchy](#thread-hierarchy)
5. [Kernel Execution Flow](#kernel-execution-flow)
6. [Key Concepts](#key-concepts)
7. [Best Practices](#best-practices)

---

## CUDA Overview

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that enables developers to harness GPU power for general-purpose computation. It allows for massive parallelism through thousands of lightweight threads executing simultaneously.

### Why CUDA?

- **Massive Parallelism**: Execute thousands of threads in parallel
- **High Bandwidth**: GPU memory bandwidth far exceeds CPU memory bandwidth
- **Energy Efficiency**: GPUs perform more operations per watt than CPUs
- **Accessibility**: C/C++ with minimal extensions

---

## Hardware Stack

### GPU Architecture Hierarchy

```
GPU Device
├── Streaming Multiprocessor (SM) [Multiple]
│   ├── Warp Scheduler
│   ├── CUDA Cores (32 threads per warp)
│   ├── Control Unit
│   ├── Registers
│   └── Shared Memory (L1 Cache)
├── Global Memory (VRAM)
└── Constant/Texture Memory
```

### Key Hardware Specifications

| Property                    | Description                                | Tesla T4 Example           |
| --------------------------- | ------------------------------------------ | -------------------------- |
| **Device Name**             | GPU model identifier                       | Tesla T4                   |
| **Compute Capability**      | Architecture version (major.minor)         | 7.5                        |
| **Max Threads per Block**   | Maximum threads in single block            | 1024                       |
| **Warp Size**               | Native parallelism unit (threads per warp) | 32                         |
| **Max Grid Dimensions**     | Max blocks in each dimension               | 2147483647 × 65535 × 65535 |
| **Global Memory**           | Total VRAM available                       | ~16 GB                     |
| **Shared Memory per Block** | Fast local memory per block                | 48 KB                      |
| **Constant Memory**         | Read-only cached memory                    | 64 KB                      |

---

## Memory Model

CUDA devices have a multi-level memory hierarchy with different access latencies and sizes.

### Memory Hierarchy Diagram

```
┌─────────────────────────────────────┐
│         HOST (CPU) Memory           │
│    Ram (GB) - Highest Latency       │
└────────────┬────────────────────────┘
             │ PCIe Transfer
             ↓
┌─────────────────────────────────────┐
│       DEVICE (GPU) Memory           │
│                                     │
│  ┌─ Registers (per thread)         │
│  │  • Size: 2-4 KB per thread       │
│  │  • Latency: ~1 cycle             │
│  │  • Scope: Single thread          │
│  │                                   │
│  ├─ Shared Memory (per block)       │
│  │  • Size: 48-96 KB per block      │
│  │  • Latency: ~1-2 cycles          │
│  │  • Scope: All threads in block   │
│  │  • Synchronized within block     │
│  │                                   │
│  ├─ Global Memory                   │
│  │  • Size: Total VRAM (GB range)   │
│  │  • Latency: 400-800 cycles       │
│  │  • Scope: All threads            │
│  │  • Persistent across kernels     │
│  │                                   │
│  └─ Constant Memory                 │
│     • Size: 64 KB                   │
│     • Latency: 1 cycle (cached)     │
│     • Scope: All threads (read-only)│
│                                     │
└─────────────────────────────────────┘
```

### Memory Access Characteristics

| Memory Type         | Size           | Latency           | Bandwidth | Access    | Scope         |
| ------------------- | -------------- | ----------------- | --------- | --------- | ------------- |
| **Registers**       | 2-4 KB/thread  | ~1 cycle          | High      | R/W       | Single Thread |
| **Shared Memory**   | 48-96 KB/block | ~2 cycles         | Very High | R/W       | Block         |
| **Global Memory**   | ~GB            | 400-800 cycles    | Medium    | R/W       | All Threads   |
| **Constant Memory** | 64 KB          | ~1 cycle (cached) | Medium    | Read-only | All Threads   |
| **Texture Memory**  | ~GB            | Variable          | Medium    | Read-only | All Threads   |

---

## Thread Hierarchy

CUDA organizes threads in a hierarchical structure for efficient execution and synchronization.

### Thread Organization Model

```
Grid (Kernel Launch)
│
├─ Block [0,0]
│  ├─ Thread (0,0,0) ─── Register Set
│  ├─ Thread (0,0,1) ─── Register Set
│  ├─ Thread (0,0,2) ─── Register Set
│  └─ ... (up to 1024 threads)
│  └─ Shared Memory (All threads in block)
│
├─ Block [0,1]
│  ├─ Thread (0,1,0) ─── Register Set
│  ├─ Thread (0,1,1) ─── Register Set
│  └─ ...
│
└─ Block [1,0]
   └─ ...
```

### 1D Thread Structure

```
Grid Structure: 1D blocks
blocksPerGrid = 4

┌───────────────────────────────────────┐
│ Grid                                  │
│ ┌─────────┬─────────┬─────────┬─────┐ │
│ │ Block 0 │ Block 1 │ Block 2 │ ... │ │
│ │         │         │         │     │ │
│ │ ┌───┐   │ ┌───┐   │ ┌───┐   │     │ │
│ │ │T₀ │   │ │T₀ │   │ │T₀ │   │     │ │
│ │ ├───┤   │ ├───┤   │ ├───┤   │     │ │
│ │ │T₁ │   │ │T₁ │   │ │T₁ │   │     │ │
│ │ ├───┤   │ ├───┤   │ ├───┤   │     │ │
│ │ │...│   │ │...│   │ │...│   │     │ │
│ │ └───┘   │ └───┘   │ └───┘   │     │ │
│ └─────────┴─────────┴─────────┴─────┘ │
└───────────────────────────────────────┘

Global Thread ID = blockIdx.x * blockDim.x + threadIdx.x
```

### 2D Thread Structure

```
Grid Structure: 2D blocks (e.g., for matrix operations)
blocksPerGrid = (16, 16)  |  threadsPerBlock = (16, 16)

┌────────────────────────────────────────────┐
│ Grid (2D)                                  │
│ ┌──────────┬──────────┬──────────┐        │
│ │Block(0,0)│Block(0,1)│Block(0,2)│ ...   │
│ │┌─┬─┬───┐│┌─┬─┬───┐│┌─┬─┬───┐│        │
│ ││T││T│...││T││T│...││T││T│...││        │
│ │├─┼─┼───┤│├─┼─┼───┤│├─┼─┼───┤│        │
│ ││T││T│...││T││T│...││T││T│...││        │
│ │└─┴─┴───┘│└─┴─┴───┘│└─┴─┴───┘│        │
│ ├──────────┼──────────┼──────────┤        │
│ │Block(1,0)│Block(1,1)│Block(1,2)│ ...   │
│ │   ...    │   ...    │   ...    │        │
│ └──────────┴──────────┴──────────┘        │
└────────────────────────────────────────────┘

Row = blockIdx.y * blockDim.y + threadIdx.y
Col = blockIdx.x * blockDim.x + threadIdx.x
```

### Thread ID Calculation

| Dimension                | Calculation                             |
| ------------------------ | --------------------------------------- |
| **1D Thread ID**         | `blockIdx.x * blockDim.x + threadIdx.x` |
| **2D Row (Y)**           | `blockIdx.y * blockDim.y + threadIdx.y` |
| **2D Col (X)**           | `blockIdx.x * blockDim.x + threadIdx.x` |
| **Flat Index (2D → 1D)** | `row * cols + col`                      |

---

## Kernel Execution Flow

### Complete Execution Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ 1. SETUP PHASE                                          │
│    • Allocate device memory (cudaMalloc)               │
│    • Initialize host data                              │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ 2. TRANSFER PHASE                                       │
│    • Copy data from host to device                      │
│    • cudaMemcpy(device, host, size, Host→Device)      │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ 3. CONFIGURATION PHASE                                  │
│    • Calculate grid dimensions                          │
│    • Calculate block dimensions                         │
│    • Determine threads per block                        │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ 4. EXECUTION PHASE                                      │
│    • Launch kernel <<<blocksPerGrid, threadsPerBlock>>>│
│    • GPU executes kernel in thousands of threads       │
│    • Synchronization at block and warp levels          │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ 5. TRANSFER BACK PHASE                                  │
│    • Copy results from device to host                   │
│    • cudaMemcpy(host, device, size, Device→Host)      │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ 6. CLEANUP PHASE                                        │
│    • Free device memory (cudaFree)                      │
│    • Free host memory (free)                            │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Host (CPU)                    Device (GPU)
┌──────────────┐             ┌──────────────────┐
│ h_data (RAM) │ ────────────│ d_data (VRAM)    │
│              │ cudaMemcpy  │                  │
│ Initialize   │ (Compute)   │ Kernel Execution │
│              │ ────────────│                  │
│ h_result     │             │ d_result         │
│              │    PCIe     │                  │
└──────────────┘             └──────────────────┘
```

---

## Key Concepts

### 1. Warps and Warp Execution

**Definition**: A warp is the smallest unit of thread execution on a GPU. Each warp contains 32 threads executing in lock-step (SIMD fashion).

```
One Warp (32 threads)
┌─────────────────────────────────────────────────┐
│ Instruction: Load from Global Memory            │
├──┬──┬──┬──┐  ┌──┬──┬──┬──┐  ┌──┬──┬──┬──┐      │
│T₀│T₁│T₂│T₃│  │T₈│T₉│..│..│  │..│..│..│T₃₁│   │
└──┴──┴──┴──┘  └──┴──┴──┴──┘  └──┴──┴──┴──┘      │
All 32 threads execute same instruction simultaneously
```

### 2. Block Synchronization

Threads within a block can synchronize using `__syncthreads()` barrier.

```
Thread Execution Timeline
Time →

Block with 4 threads:
Thread 0: [Code] ──|barrier|── [Code]
Thread 1: [Code] ──|barrier|── [Code]
Thread 2: [Code] ──|barrier|── [Code]
Thread 3: [Code] ──|barrier|── [Code]
         (All threads wait at barrier)
```

### 3. Memory Coalescing

Global memory accesses are efficient when threads access contiguous memory.

**Good Pattern (Coalesced Access)**:

```
Thread 0 → Address 0
Thread 1 → Address 1
Thread 2 → Address 2
...
Thread 31 → Address 31

All 32 threads load from consecutive addresses → 1 transaction
```

**Bad Pattern (Non-Coalesced Access)**:

```
Thread 0 → Address 0
Thread 1 → Address 100
Thread 2 → Address 200
...

Scattered memory access → Multiple transactions (slow)
```

### 4. Occupancy

Occupancy is the ratio of active warps to maximum possible warps on an SM.

| Factor                   | Impact on Occupancy                                               |
| ------------------------ | ----------------------------------------------------------------- |
| **Registers per Thread** | More registers = fewer blocks fit = lower occupancy               |
| **Shared Memory**        | Larger blocks need more shared memory = lower occupancy           |
| **Block Size**           | Smaller blocks = more threads per SM = potential higher occupancy |
| **Active Threads**       | More active threads = better resource utilization                 |

---

## Best Practices

### 1. Block Size Selection

- **Typical**: 128, 256, or 512 threads per block
- **Range**: 32 to 1024 threads per block
- Consider register usage and shared memory requirements
- Warp-aligned (multiples of 32) for efficiency

### 2. Grid Size Calculation

```
Elements: N = 1000
Threads per Block: 256

Blocks Needed = (N + threadsPerBlock - 1) / threadsPerBlock
              = (1000 + 256 - 1) / 256
              = 1255 / 256
              = 5 blocks (rounded up)
```

### 3. Memory Access Patterns

- **Coalesce global memory accesses**: Use contiguous addresses
- **Use shared memory**: When data is reused within a block
- **Minimize divergence**: Avoid conditional statements based on threadIdx
- **Avoid bank conflicts**: Structure shared memory access carefully

### 4. Kernel Launch Configuration

- **1D Problems**: Use 1D blocks and grids for simplicity
- **2D Problems**: Use 2D blocks for spatial data (matrices, images)
- **3D Problems**: Use 3D blocks/grids for volumetric data

### 5. Performance Optimization Tips

| Technique           | Benefit                                       | Difficulty |
| ------------------- | --------------------------------------------- | ---------- |
| **Coalescing**      | Reduces memory transactions                   | Low        |
| **Shared Memory**   | Much faster than global memory                | Medium     |
| **Loop Unrolling**  | Increases ILP (Instruction-Level Parallelism) | Medium     |
| **Texture Caching** | Fast reads + spatial locality                 | Medium     |
| **Constant Memory** | Fast broadcast to all threads                 | Low        |

---

## Practical Execution Examples

### Example 1: Device Query

**Purpose**: Determine GPU capabilities before launching kernels

**Key Metrics Retrieved**:

- Device name and compute capability
- Maximum thread block dimensions
- Memory capacities (global, shared, constant)
- Warp size (always 32 for NVIDIA GPUs)

### Example 2: Array Operations

**Purpose**: Parallel element-wise operations

**Pattern**:

- Each thread processes one array element
- Block size typically 256-512 threads
- Grid size depends on array length

### Example 3: Matrix Operations

**Purpose**: 2D data parallel processing

**Pattern**:

- 2D block structure (e.g., 16×16)
- Threads map to matrix elements
- Natural row/column indexing
- Suitable for tiling and shared memory optimization

---

## Common Pitfalls to Avoid

| Pitfall                      | Consequence        | Solution                             |
| ---------------------------- | ------------------ | ------------------------------------ |
| **Not checking CUDA errors** | Silent failures    | Use `cudaGetLastError()`             |
| **Insufficient threads**     | Underutilized GPU  | Calculate proper grid size           |
| **Excessive shared memory**  | Low occupancy      | Balance with register usage          |
| **Unaligned memory access**  | Poor performance   | Use coalescing patterns              |
| **Race conditions**          | Incorrect results  | Use `__syncthreads()` and atomic ops |
| **Reading garbage data**     | Undefined behavior | Validate array bounds                |

---

## Summary Table: CPU vs GPU

| Aspect               | CPU                       | GPU                        |
| -------------------- | ------------------------- | -------------------------- |
| **Threads**          | Few (4-16)                | Thousands (10,000+)        |
| **Latency**          | ~100 cycles               | ~400-800 cycles            |
| **Throughput**       | Lower                     | Much Higher                |
| **Best For**         | Sequential, complex logic | Parallel, regular patterns |
| **Memory BW**        | ~100 GB/s                 | ~300+ GB/s                 |
| **Power Efficiency** | Lower                     | Higher (ops/watt)          |

---

## References

- NVIDIA CUDA Compute Capability Documentation
- CUDA C++ Programming Guide
- GPU Architecture and Thread Execution Model
