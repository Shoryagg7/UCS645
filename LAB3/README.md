# LAB 3: Correlation Coefficient Optimization

<div align="center">

![Correlation](https://img.shields.io/badge/Algorithm-Correlation-blue?style=flat-square)
![C++](https://img.shields.io/badge/Language-C%2B%2B17-brightgreen?style=flat-square)
![Optimization](https://img.shields.io/badge/Focus-Performance-orange?style=flat-square)

**Optimizing Parallel Matrix Correlation Coefficient Calculations**

</div>

---

## 📋 Lab Overview

This assignment focuses on optimizing the calculation of correlation coefficients between all pairs of row vectors in a matrix. Starting from a sequential baseline, you will parallelize and optimize using OpenMP, SIMD vectorization, loop unrolling, and memory access pattern improvements.

**Key Topics:**

- Correlation coefficient mathematics
- Parallel reduction patterns
- SIMD vectorization with OpenMP
- Loop unrolling and instruction-level parallelism
- Memory access optimization
- Performance profiling and analysis

---

## 🎯 Assignment Requirements

### Task Description

Given a matrix with **ny** rows and **nx** columns (all representing vectors), calculate the Pearson correlation coefficient between every pair of rows. Store results in lower triangular matrix form.

### Interface

**Function Signature:**
```
correlate(ny, nx, data, result)
```

**Parameters:**

- `ny`: Number of vectors (rows)
- `nx`: Dimension of each vector (columns)
- `data`: Input matrix (row-major storage)
- `result`: Output correlation matrix (lower triangular)

### Correlation Coefficient Formula

```
correlation(i, j) = cov(X_i, X_j) / (σ_i × σ_j)

Where:
  cov(X, Y) = Σ(x_i - mean_x)(y_i - mean_y) / n
  σ = sqrt(Σ(x_i - mean)² / n)
```

---

## 📁 File Structure

```
LAB3/
├── main.cpp              # Command-line interface & profiling
├── correlate.cpp         # Four implementations
├── correlate.h           # Function declarations
├── Makefile              # Build automation
└── README.md             # This file
```

---

## 🚀 Compilation & Usage

### Build

```bash
g++ -std=c++17 -O3 -fopenmp main.cpp correlate.cpp -o correlate -lm

# Or with Makefile (if available)
make
make fast          # Aggressive optimizations
make debug         # Debug symbols
```

### Run

```bash
# Basic run (sequential)
./correlate -ny 1000 -nx 1000

# Compare all implementations
./correlate -ny 1000 -nx 1000 -all

# Parallel with 8 threads
./correlate -ny 2000 -nx 5000 -threads 8 -impl 2

# Verify correctness
./correlate -ny 500 -nx 1000 -verify -all

# Help
./correlate -h
```

### Command-Line Options

```
-ny NUM          Number of vectors (rows)           [default: 1000]
-nx NUM          Vector dimension (columns)         [default: 1000]
-threads NUM     Number of OpenMP threads           [default: auto]
-verify          Verify parallel correctness        [default: off]
-all             Run all implementations            [default: off]
-impl [1-4]      Run specific implementation        [default: none]
```

---

## 🔧 Implementation Levels

### Level 1: Sequential Baseline
**Location:** `correlate_sequential()`

**Description:** Three nested loops calculating means, standard deviations, and correlation coefficients sequentially.

**Characteristics:**
- No parallelism
- Reference implementation for correctness verification
- Single-threaded baseline for performance comparison
- **~2000 MFLOPS** on typical hardware
- Two-phase algorithm: compute statistics first, then correlations

---

### Level 2: Parallel with OpenMP
**Location:** `correlate_parallel()`

**Description:** Parallelizes the outer loops using OpenMP `parallel for` directives. Both the mean calculation phase and correlation calculation phase are parallelized independently.

**Improvements:**
- Outer loop parallelization with OpenMP
- Automatic thread distribution and management
- Load balancing across cores
- Two independent parallel sections (mean, then correlation)
- **Expected: 4-8x speedup** @ 8 threads

---

### Level 3: Optimized with Loop Unrolling
**Location:** `correlate_optimized()`

**Description:** Adds instruction-level parallelism through 4x loop unrolling. Processes four vector elements simultaneously to enable better instruction scheduling and reduced loop overhead.

**Improvements:**
- 4x loop unrolling increases instruction-level parallelism (ILP)
- Reduces loop iteration count and branch overhead
- Better compiler opportunities for vectorization
- Improved resource utilization on superscalar CPUs
- SIMD pragma hints to enable auto-vectorization
- **Expected: 8-12x speedup** @ 8 threads
- 15-25% additional speedup over level 2

---

### Level 4: Highly Optimized with SIMD
**Location:** `correlate_highly_optimized()`

**Description:** Combines multi-threading with explicit SIMD vectorization directives. Uses `omp simd` and `reduction` clauses for fine-grained parallelization.

**Improvements:**
- Explicit SIMD parallelization with `#pragma omp simd`
- SIMD reduction for efficient vector accumulation
- Aligned memory allocations and vectorization hints
- Combined benefits of multi-threading AND vector instructions
- Compiler auto-vectorizes inner loops with explicit guidance
- **Expected: 10-14x speedup** @ 8 threads
- Up to 50% additional speedup over level 3 on AVX systems

---

## 📊 Expected Performance

###Performance Table (Typical 8-Core System)

```
┌──────────────────────┬─────────────┬──────────┬──────────────┐
│ Implementation       │ Time (s)    │ Speedup  │ GFLOPS       │
├──────────────────────┼─────────────┼──────────┼──────────────┤
│ Sequential (Level 1) │ 2.5         │ 1.0x     │ 2.0          │
├──────────────────────┼─────────────┼──────────┼──────────────┤
│ Parallel (Level 2)   │ 0.35        │ 7.1x     │ 14.3         │
├──────────────────────┼─────────────┼──────────┼──────────────┤
│ Optimized (Level 3)  │ 0.28        │ 8.9x     │ 17.9         │
├──────────────────────┼─────────────┼──────────┼–──────────────┤
│ Highly Optim (L4)    │ 0.25        │ 10.0x    │ 20.0         │
└──────────────────────┴─────────────┴──────────┴──────────────┘

Test: ny=2000, nx=5000, 8 threads
Flops: 2 * ny * ny * nx = 200 billion FLOPs
```

### Optimization Impact

```
Speedup vs Threads (ny=2000, nx=5000)

       12 │
          │            Level 4
          │          (Highly Opt)
       10 │         ●
          │        ╱
        8 │      ●      Level 3
          │     ╱        (Optimized)
        6 │   ●
          │  ╱           Level 2
        4 │●             (Parallel)
          │ ╲
        2 │  ╲
          │   ╲
        0 └────●─────────────────────
          0  1  2  4  8 16
             Threads > Level 1 (Sequential)
```

---

## 📈 Performance Analysis Tasks

### Task 1: Single Threaded Performance

```bash
./correlate -ny 1000 -nx 5000 -impl 1
```

Measure:

- Execution time
- GFLOPS (Flops per second)
- Compare with theoretical peak

### Task 2: Scaling Analysis

```bash
for threads in 1 2 4 8 16; do
    OMP_NUM_THREADS=$threads ./correlate -ny 2000 -nx 5000 -impl 2
done
```

Plot speedup curve and calculate:

- Linear speedup line
- Efficiency at each core count
- Identify saturation point

### Task 3: Size Scaling

```bash
for size in 500 1000 2000 4000; do
    ./correlate -ny $size -nx $size -threads 8 -all
done
```

Analyze:

- How speedup changes with problem size
- Memory bandwidth effects
- Cache behavior (L1/L2/L3/RAM)

### Task 4: Profiling with perf

```bash
# Sequential baseline
perf stat -e cycles,instructions,cache-references,cache-misses \
    ./correlate -ny 2000 -nx 5000 -impl 1

# Parallel optimized
perf stat -e cycles,instructions,cache-references,cache-misses \
    ./correlate -ny 2000 -nx 5000 -threads 8 -impl 4

# Focus on vectorization
perf stat -e "cpu/event=0xc0,umask=0x00/" \
    ./correlate -ny 2000 -nx 5000 -impl 4
```

---

## 📋 Deliverables Checklist

- [ ] **Implementations**
  - [ ] Sequential baseline (correct reference)
  - [ ] Parallel with OpenMP
  - [ ] Optimized with unrolling
  - [ ] Highly optimized with SIMD hints

- [ ] **Code Quality**
  - [ ] Compiles without warnings
  - [ ] Produces correct results (verify flag)
  - [ ] Handles edge cases (mean=0, etc.)
  - [ ] Efficient memory usage

- [ ] **Performance Measurements**
  - [ ] Timing data for all implementations
  - [ ] Speedup graphs (threads vs time)
  - [ ] Efficiency analysis
  - [ ] GFLOPS measurements

- [ ] **Analysis Report**
  - [ ] Explain each optimization
  - [ ] Theoretical vs measured speedup
  - [ ] Cache behavior analysis
  - [ ] bottleneck identification
  - [ ] Conclusions and recommendations

---

## 🔍 Key Optimization Techniques

### 1. Loop Unrolling

**Concept:** Process multiple array elements in each loop iteration instead of one, reducing loop overhead and enabling better CPU instruction scheduling.

**Benefit:** 15-25% speedup typical, especially on CPUs with deep pipelines

**When to use:**
- Inner loops with simple operations
- When iteration count is known
- When compiler can't auto-unroll effectively

---

### 2. OpenMP SIMD Vectorization

**Concept:** Use explicit SIMD pragmas to tell the compiler to vectorize loops. Combined with multi-threading for maximum parallelism.

**Benefit:** 2-4x speedup with AVX-256 vectors, 4-8x with AVX-512

**When to use:**
- Loops with independent iterations
- Regular memory access patterns
- Data-parallel computations

---

### 3. Parallel Reduction

**Concept:** Safely accumulate results from parallel threads without explicit locks. OpenMP handles synchronization automatically.

**Benefit:** Correct synchronization with minimal overhead

**When to use:**
- Computing sums, products, min/max across parallel data
- Avoiding lock contention
- Maintaining numerical stability

---

### 4. Memory Access Pattern Optimization

**Concept:** Access memory in row-major order (iterate through contiguous elements) to maximize cache utilization and minimize miss rates.

**Key principles:**
- Process data sequentially in the order it's stored in memory
- Minimize cache line misses through spatial locality
- Keep working set small (fit in L1/L2 cache)
- Avoid random jumps through memory that cause cache misses

**Benefit:** 2-5x improvement in cache efficiency and memory throughput

---

## 📚 Mathematical Background

### Pearson Correlation Coefficient

```
r = Σ(x_i - x̄)(y_i - ȳ) / √(Σ(x_i - x̄)² × Σ(y_i - ȳ)²)

Efficient computation:
1. Compute means: x̄ and ȳ
2. Compute standard deviations: σ_x and σ_y
3. Compute covariance: cov
4. r = cov / (σ_x × σ_y)
```

### Properties Used

- Symmetry: corr(i, j) = corr(j, i)
- Storage: Only lower triangle needed (ny\*(ny+1)/2 values)
- Stability: Use Welford's algorithm for large datasets
- Normalization: Result always in [-1, 1]

---
