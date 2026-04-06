# Question 1: Molecular Dynamics - Force Calculation

## Problem Statement

Implement a parallel computation of Lennard-Jones potential forces in a molecular dynamics simulation. Given N particles in 3D space, calculate the total potential energy and forces acting on each particle.

## Lennard-Jones Model

The Lennard-Jones potential describes the interaction between two particles:

$$V(r) = 4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$

Where:
- **ε (sigma)**: Energy scale parameter (set to 1.0)
- **σ (sigma)**: Distance scale parameter (set to 1.0)
- **r**: Distance between particles
- **cutoff radius**: 2.5σ

The force is derived from the potential:

$$\vec{F} = -\nabla V(r) = 48\epsilon \frac{1}{r^2}\left[\left(\frac{\sigma}{r}\right)^{12} - 0.5\left(\frac{\sigma}{r}\right)^6\right]\hat{r}$$

## Implementation Details

### Key Features

1. **Particle Structure**: Each particle stores position (x, y, z), velocity, and force components
2. **Force Calculation**: O(N²) algorithm computing pairwise interactions
3. **Newton's Third Law**: Forces satisfy action-reaction principle (f_ij = -f_ji)
4. **Cutoff Radius**: Interactions beyond 2.5σ are ignored
5. **Race Condition Handling**: Uses `#pragma omp atomic` for force accumulation

### Parallelization Strategy

```cpp
#pragma omp parallel for collapse(2) num_threads(num_threads) \
        schedule(static) reduction(+:total_energy)
for (int i = 0; i < num_particles; i++) {
    for (int j = i + 1; j < num_particles; j++) {
        // Compute pairwise interactions
        #pragma omp atomic  // Prevent race conditions
        particles[i].fx += fx;
        // ...
    }
}
```

**Techniques Used:**
- `collapse(2)`: Parallelize nested loops for better work distribution
- `schedule(static)`: Predictable work distribution
- `reduction(+:total_energy)`: Thread-safe energy accumulation
- `#pragma omp atomic`: Atomic updates for force components

## Compilation and Execution

### Compile
```bash
g++ -fopenmp -O3 -std=c++11 q1.cpp -o q1
```

### Run
```bash
./q1
```

### Output
- Prints thread count, execution time, speedup, and total energy for 1-16 threads
- Saves results to `md_results.txt` with format: `Threads, Time, Speedup, Total_Energy`

## Expected Results

### Performance Characteristics
- **Initial Speedup (2-4 threads)**: Should show good scaling due to embarrassingly parallel nature
- **Scaling Plateau**: Performance may plateau due to:
  - Memory bandwidth limitations
  - Atomic operation contention
  - Load imbalance in force accumulation

### Data Dependencies
- **Output Dependencies**: Each particle's force is computed independently (after synchronization)
- **Anti-dependencies**: None (forward references only)
- **Flow Dependencies**: Minimal (atomic operations on force accumulation)

## Load Balancing Analysis

The nested loop structure creates potential load imbalance:
- Particle i=0 interacts with N-1 particles
- Particle i=N/2 interacts with N/2 particles
- Particle i=N-1 interacts with 0 particles

**Solution**: `collapse(2)` directive distributes iterations across thread pool more evenly.

## Optimization Opportunities

1. **Spatial Decomposition**: Divide space into cells; only check neighboring cells
2. **Force Lists**: Pre-compute which particle pairs interact
3. **SIMD Vectorization**: Use SIMD for distance calculations
4. **Memory Alignment**: Ensure particle data is aligned for cache efficiency

## References

- Lennard-Jones potential: https://en.wikipedia.org/wiki/Lennard-Jones_potential
- OpenMP Atomic operations: https://www.openmp.org/spec-html/5.0/openmpsu59.html#x262-10910009.3.3

---

**Lab**: UCS645 - Parallel & Distributed Computing
**Assignment**: LAB2 - Question 1
**Date**: 2026
