# LAB 1: Introduction to OpenMP Parallel Programming

## Lab Overview

This lab focuses on understanding parallel programming concepts using OpenMP, exploring parallelization techniques, and analyzing performance improvements through speedup measurements across different problem domains.

---

## Questions

### Question 1: DAXPY Operation Parallelization
**Objective:** Implement and analyze parallel DAXPY (Double-precision A·X Plus Y) operation

**Implementation:**
- Computed `Y = a*X + Y` for vectors of 16 million elements
- Used `#pragma omp parallel for` to parallelize the vector operation
- Measured execution time and speedup for 1 to 16 threads
- Generated speedup data saved to `daxpy_speedup.txt`

**Key Learnings:**
- Basic OpenMP parallel loop constructs
- Work distribution across threads for embarrassingly parallel tasks
- Performance measurement using `omp_get_wtime()`
- Understanding speedup metrics and Amdahl's law implications

---

### Question 2: Matrix Multiplication with 1D and 2D Parallelization
**Objective:** Compare different parallelization strategies for matrix multiplication

**Implementation:**
- Multiplied two 1000×1000 matrices using two approaches:
  1. **1D Parallelization:** Parallelized outer loop only
  2. **2D Parallelization:** Used `collapse(2)` to parallelize both outer loops with static scheduling
- Measured performance for both approaches with 1 to 16 threads
- Compared speedup characteristics between both methods

**Key Learnings:**
- Different loop parallelization strategies (`collapse` directive)
- Impact of work granularity on parallel performance
- Static scheduling and chunk sizes for load balancing
- Trade-offs between parallelization overhead and computation

---

### Question 3: π Calculation using Numerical Integration
**Objective:** Compute π using parallel numerical integration with reduction

**Implementation:**
- Calculated π using Riemann sum approximation with 10 million steps
- Used the formula: π = ∫₀¹ 4/(1+x²) dx
- Employed `reduction(+:sum)` clause to safely accumulate partial sums
- Measured speedup across 1 to 16 threads

**Key Learnings:**
- OpenMP reduction operations for safe parallel accumulation
- Avoiding race conditions in shared variable updates
- Performance characteristics of reduction-based parallelism
- Numerical integration as a parallel computing problem

---

## Overall Insights

### Parallel Programming Fundamentals
1. **Parallelization Patterns:** Learned data-parallel patterns (parallel for), work distribution strategies, and reduction operations
2. **Performance Analysis:** Understood how to measure and interpret speedup, efficiency, and scalability
3. **OpenMP Directives:** Gained hands-on experience with core OpenMP constructs: `parallel for`, `num_threads`, `reduction`, `collapse`, and `schedule`

### Performance Characteristics
- **Speedup vs Threads:** Observed how performance scales with increasing thread counts
- **Parallel Overhead:** Recognized that parallelization introduces overhead that may limit speedup
- **Problem Size Dependency:** Larger problem sizes generally achieve better speedup due to reduced relative overhead
- **Hardware Limitations:** Physical core count and memory bandwidth affect maximum achievable speedup

### Best Practices
- Always measure serial (1-thread) baseline for accurate speedup calculations
- Choose appropriate problem sizes to make parallelization worthwhile
- Use proper synchronization mechanisms (e.g., reduction) to avoid race conditions
- Consider data locality and cache effects in parallel programs

---
