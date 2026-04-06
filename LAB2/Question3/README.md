# Question 3: Scientific Computing - Heat Diffusion Simulation

## Problem Statement

Simulate heat diffusion in a 2D metal plate using the finite difference method with parallel OpenMP implementation. This demonstrates solving partial differential equations (PDEs) in parallel.

## Mathematical Background

### Heat Equation (Parabolic PDE)

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

Where:
- **u(x,y,t)**: Temperature at position (x,y) and time t
- **α**: Thermal diffusivity coefficient (material property)
- **∇²u**: Laplacian (sum of second derivatives)

### Finite Difference Discretization

Using **explicit forward Euler** time stepping and **central differences** for space:

$$u_{i,j}^{n+1} = u_{i,j}^{n} + r\left[u_{i+1,j}^{n} + u_{i-1,j}^{n} + u_{i,j+1}^{n} + u_{i,j-1}^{n} - 4u_{i,j}^{n}\right]$$

Where:
- **r = α·Δt/(Δx)²**: Stability parameter
- **Δt**: Time step
- **Δx**: Spatial grid spacing

### Stability Condition

For numerical stability (CFL condition):

$$r \leq 0.25 \text{ for 2D}$$

This ensures the explicit method doesn't diverge. Our implementation uses r ≈ 0.0025.

## Implementation Details

### Key Features

1. **Grid Structure**: 500×500 spatial grid
2. **Time Stepping**: 1000 iterations with small time steps
3. **Initial Condition**: Heat source in center (100°C)
4. **Boundary Conditions**: Zero temperature at edges (Dirichlet)
5. **Double Buffering**: Separate arrays for current and next temperature

### Data Dependencies Analysis

**Critical observation:**
- Current timestep (n) only depends on previous timestep (n-1)
- No spatial dependencies within the same timestep! ✅
- Each grid point (i,j) reads from 5 cells and writes to unique output cell

```
Computation for u(i,j) at timestep n+1:
- Reads:  u(i±1,j,n), u(i,j±1,n), u(i,j,n)
- Writes: u(i,j,n+1)
- No race conditions possible!
```

## Parallelization Strategies

### Strategy 1: Static Scheduling

```cpp
#pragma omp parallel for collapse(2) num_threads(num_threads) \
        schedule(static)
for (int i = 1; i < grid_size - 1; i++) {
    for (int j = 1; j < grid_size - 1; j++) {
        // Compute new temperature
    }
}
```

**Characteristics:**
- Fixed chunk assignment at compile time
- Minimal synchronization overhead
- Good cache locality if iterations are contiguous
- May have load imbalance due to boundary effects

### Strategy 2: Dynamic Scheduling

```cpp
#pragma omp parallel for collapse(2) num_threads(num_threads) \
        schedule(dynamic, chunk_size)
```

**Characteristics:**
- Runtime work distribution
- Better load balancing
- More synchronization overhead
- Useful for workload imbalance

### Strategy 3: Guided Scheduling

```cpp
#pragma omp parallel for collapse(2) num_threads(num_threads) \
        schedule(guided)
```

**Characteristics:**
- Chunk size decreases over time
- Balances static and dynamic efficiency
- Chunk size = remaining_iterations / (2 * num_threads)
- Good general-purpose strategy

## Performance Analysis

### Computation Characteristics

| Aspect | Details |
|--------|---------|
| Grid Size | 500 × 500 = 250,000 cells |
| Operations per cell | 5 reads + 1 write + 8 ops = ~14 FLOPs |
| Total FLOPs/timestep | 250,000 × 14 = 3.5 MFLOP |
| Total FLOPs (1000 steps) | 3.5 GFLOP |
| Memory per grid | 500 × 500 × 8 bytes = 2 MB (< L3 cache) |
| Memory bandwidth needed | High for double-buffered access |

### Cache Behavior

- **Single grid**: 2 MB fits in L3 cache (8-20 MB typical)
- **Double buffer**: 4 MB total, still cache-friendly
- **Access pattern**: Stencil (5-point) - good spatial locality
- **Reuse factor**: High (each cell accessed 5x per iteration)

### Expected Speedup

```
Ideal parallel speedup:
- Threads 1-4: Nearly linear (good cache, no contention)
- Threads 4-8: Good scaling (memory bandwidth limited)
- Threads 8-16: Diminishing returns (memory bandwidth plateau)
```

## Profiling and Measurement

### Using perf stat

```bash
perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \
    ./q3
```

Expected metrics:
- IPC (Instructions per Cycle): ~2-3
- L1 cache hit rate: >95%
- LLC misses: Low (working set fits in cache)

### Using LIKWID

```bash
likwid-perfctr -C 0-3 -g FLOPS_SP q3
likwid-perfctr -C 0-3 -g MEM q3
```

**FLOPS Analysis:**
- Unrolled FLOPS (actual operations): ~50-100 GB/s
- Nominal FLOPS (theoretical): Higher

**Memory Analysis:**
- Bandwidth required: ~2-4 GB/s
- Peak system bandwidth: ~50-100 GB/s (depending on CPU)
- Memory efficiency: ~5-10%

## Compilation and Execution

### Compile
```bash
g++ -fopenmp -O3 -std=c++11 -march=native q3.cpp -o q3
```

**Compiler Flags:**
- `-O3`: Aggressive optimization
- `-march=native`: Use CPU-specific optimizations
- `-fopenmp`: Enable OpenMP

### Run
```bash
./q3
```

### Output
- Results saved to `heatsim_results.txt`
- Format: `Threads, Schedule, Time, Speedup, Total_Heat, Max_Temp`
- Tests all three scheduling strategies

## Results Interpretation

### Validity Checks

1. **Temperature Conservation**: Total heat should decrease over time (diffusion)
2. **Max Temperature**: Should monotonically decrease from 100°C
3. **Smoothness**: No oscillations or instabilities (indicates CFL satisfied)

### Performance Metrics

| Schedule | Overhead | Load Balance | Cache Locality |
|----------|----------|--------------|-----------------|
| Static | Lowest | Fair | Best |
| Dynamic | High | Excellent | Moderate |
| Guided | Medium | Good | Good |

## Optimization Opportunities

### 1. Loop Tiling/Cache Blocking

Process grid in tiles to improve cache reuse:

```cpp
#pragma omp parallel for num_threads(num_threads)
for (int ti = 1; ti < grid_size-1; ti += tile_size) {
    for (int tj = 1; tj < grid_size-1; tj += tile_size) {
        for (int i = ti; i < min(ti+tile_size, grid_size-1); i++) {
            for (int j = tj; j < min(tj+tile_size, grid_size-1); j++) {
                // Compute
            }
        }
    }
}
```

Benefits:
- Better L1/L2 cache utilization
- Reduced memory traffic
- Improved temporal locality

### 2. SIMD Vectorization

Compiler typically auto-vectorizes stencil operations. Ensure:
- Array alignment: `-mavx` requires 32-byte alignment
- No pointer aliasing (restrict keyword)
- Simple loop structure

### 3. Time Integration Methods

Current: explicit Euler (O(Δt) accuracy, restrictive CFL)

Alternatives:
- **Implicit method**: Unconditionally stable but requires solver
- **Higher-order RK**: Better accuracy-to-time tradeoff

## Extensions and Variations

### 3D Heat Diffusion

Extend to 3D: 5-point stencil becomes 7-point stencil
- 2x computational complexity
- 3x memory
- Communication patterns become more complex with GPU

### Adaptive Mesh Refinement (AMR)

Use finer grids where temperature gradients are large
- Improves accuracy
- Reduces computation in smooth regions
- Complex parallelization strategy

### Non-uniform Materials

Different α in different regions
- Physical realism
- Minimal additional computational cost
- Interesting load imbalance scenarios

## References

- Heat Equation: https://en.wikipedia.org/wiki/Heat_equation
- Finite Difference Methods: https://en.wikipedia.org/wiki/Finite_difference_method
- CFL Condition: https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
- OpenMP Scheduling: https://www.openmp.org/spec-html/5.0/openmpsu41.html#x54-1440005

---

**Lab**: UCS645 - Parallel & Distributed Computing
**Assignment**: LAB2 - Question 3
**Date**: 2026
