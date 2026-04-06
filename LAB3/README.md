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

```cpp
void correlate(int ny, int nx, const float* data, float* result)
```

**Parameters:**

- `ny`: Number of vectors (rows)
- `nx`: Dimension of each vector (columns)
- `data`: Input matrix (row-major storage: `data[i*nx + j]`)
- `result`: Output correlation matrix (lower triangular: `result[i + j*ny]` for `j ≤ i`)

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

```cpp
for (int i = 0; i < ny; i++) {
    for (int j = 0; j <= i; j++) {
        covariance = 0.0;
        for (int k = 0; k < nx; k++) {
            covariance += (data[i][k] - mean_i) * (data[j][k] - mean_j);
        }
        result[i + j*ny] = covariance / (stddev_i * stddev_j);
    }
}
```

**Characteristics:**

- No parallelism
- Reference for correctness
- Single-threaded baseline
- **~2000 MFLOPS** on typical hardware

---

### Level 2: Parallel with OpenMP

**Location:** `correlate_parallel()`

```cpp
#pragma omp parallel for        // Mean calculation
for (int i = 0; i < ny; i++) { ... }

#pragma omp parallel for        // Correlation calculation
for (int i = 0; i < ny; i++) {
    for (int j = 0; j <= i; j++) { ... }
}
```

**Improvements:**

- Outer loop parallelization
- Automatic thread management
- Load distribution across cores
- **Expected: 4-8x speedup** @ 8 threads

---

### Level 3: Optimized with Unrolling

**Location:** `correlate_optimized()`

```cpp
// 4x loop unrolling for ILP
int k = 0;
for (; k + 4 <= nx; k += 4) {
    xi0 = data[i*nx + k] - mean_i;
    xi1 = data[i*nx + k + 1] - mean_i;
    xi2 = data[i*nx + k + 2] - mean_i;
    xi3 = data[i*nx + k + 3] - mean_i;

    covariance += xi0*y0 + xi1*y1 + xi2*y2 + xi3*y3;
}
```

**Improvements:**

- Instruction-level parallelism (ILP)
- Reduced loop overhead
- Better compiler vectorization
- SIMD pragma hints

---

### Level 4: Highly Optimized

**Location:** `correlate_highly_optimized()`

```cpp
#pragma omp parallel for simd    // Explicit SIMD
for (int i = 0; i < ny; i++) {
    #pragma omp simd reduction(+:sum, sum_sq)
    for (int j = 0; j < nx; j++) {
        sum += data[i*nx + j];
        sum_sq += data[i*nx + j] * data[i*nx + j];
    }
}
```

**Improvements:**

- SIMD reduction operations
- Explicit vectorization hints
- Compiler auto-vectorization enabled
- Combined multi-threading + vectorization

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

Reduces loop overhead and enables instruction scheduling:

```cpp
// 4x unroll
for (k = 0; k +4 <= nx; k += 4)
    result += a[k]*b[k] + a[k+1]*b[k+1] +
              a[k+2]*b[k+2] + a[k+3]*b[k+3];
```

**Benefit:** 15-25% speedup typical

### 2. OpenMP SIMD

Explicit vectorization hints:

```cpp
#pragma omp simd reduction(+:sum)
for (int i = 0; i < N; i++)
    sum += a[i] * b[i];
```

**Benefit:** 2-4x speedup with AVX-256, 4-8x with AVX-512

### 3. Parallel Reduction

Efficient pattern for accumulation:

```cpp
#pragma omp parallel for reduction(+:total_sum)
for (int i = 0; i < N; i++)
    total_sum += expensive_compute(i);
```

**Benefit:** Correct synchronization, no locks needed

### 4. Memory Access Pattern

Improve cache utilization:

```cpp
// Good: Row-major access (cache-friendly)
for (int i = 0; i < ny; i++)
    for (int j = 0; j < nx; j++)
        process(data[i*nx + j]);  ✓

// Bad: Random jumping
for (int j = 0; j < nx; j++)
    for (int i = 0; i < ny; i++)
        process(data[i*nx + j]);  ✗
```

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

## 🐛 Debugging & Verification

### Correctness Check

```bash
./correlate -ny 500 -nx 1000 -verify -all
```

Compares:

- Sequential vs Parallel
- Parallel vs Optimized
- Optimized vs Highly Optimized

Tolerance: 1e-5f (relative or absolute)

### Corner Cases to Test

1. Single vector (ny=1): Result should be 1.0
2. Identical vectors: Correlation = 1.0
3. Orthogonal vectors: Correlation ≈ 0
4. Constant vector: Correlation = undefined → 0

---

## 📖 References & Resources

### OpenMP Documentation

- [OpenMP 5.0 Specification](https://www.openmp.org)
- [GCC libgomp](https://gcc.gnu.org/onlinedocs/libgomp/)
- [Intel OpenMP](https://www.intel.com/content/www/en/us/docs/cpp-compiler/developer-guide/top.html)

### SIMD & Vectorization

- [GCC Vectorization Guide](https://gcc.gnu.org/projects/tree-ssa/vectorization.html)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/en/us/docs/intrinsics-guide/index.html)
- [ARM NEON Guide](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

### Performance Analysis

- [perf Tutorial](https://perf.wiki.kernel.org/)
- [Intel VTune](https://www.intel.com/content/www/en/us/docs/vtune/user-guide/top.html)
- [Roofline Model](https://www.eecs.berkeley.edu/~demmel/cs267/lecture26.pdf)

### Parallel Computing

- [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
- [OpenMP by Example](https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf)
- [Scott Meany - Parallel Computing](https://www.youtube.com/watch?v=nE0Z077-4fk)

---

## 🎓 Learning Outcomes

After completing this lab, you will understand:

✅ **Parallel Patterns**

- Data parallelism across rows
- Reduction operations
- Load balancing strategies

✅ **Optimization Techniques**

- SIMD vectorization
- Loop unrolling for ILP
- Cache-friendly access patterns

✅ **Performance Analysis**

- Measuring GFLOPS and speedup
- Scaling efficiency
- Bottleneck identification

✅ **Practical OpenMP**

- `parallel for` constructs
- `simd` directives
- `reduction` clauses

---

<div align="center">

### 🌟 Performance Optimization Mastery! 🌟

_From sequential to vectorized: Understanding parallel performance_

---

**UCS645 - Parallel & Distributed Computing**
**Assignment: LAB3 - Correlation Coefficient Optimization**

</div>
