# OpenMP Examples

This folder contains introductory examples demonstrating fundamental OpenMP concepts and common pitfalls in parallel programming.

---

## Example 1: Race Condition Demonstration (`ex1.cpp`)

**Purpose:** Illustrates a classic race condition bug in parallel programming

**What Happens:**
- Multiple threads simultaneously try to update the shared variable `sum`
- Results in a **race condition** - outcome depends on thread scheduling
- Output is non-deterministic and incorrect (expected: 5050, actual: varies)

**Key Learning:**
- Demonstrates why unprotected shared variable updates are dangerous
- Shows the need for proper synchronization mechanisms
- Introduces the concept of data races in parallel programs

---

## Example 2: Fixing Race Conditions with Reduction (`ex2.cpp`)

**Purpose:** Shows the correct way to handle accumulation in parallel loops


**What Happens:**
- The `reduction(+:sum)` clause creates private copies of `sum` for each thread
- Each thread accumulates its partial sum independently
- OpenMP automatically combines all partial sums at the end
- Produces correct result: 5050

**Key Learning:**
- Proper use of the `reduction` clause for safe parallel accumulation
- Understanding how OpenMP handles shared vs. private variables
- Comparison with Example 1 shows the importance of synchronization

---

## Example 3: Performance Measurement (`ex3.cpp`)

**Purpose:** Demonstrates timing parallel code execution


**What Happens:**
- Uses `omp_get_wtime()` to measure wall-clock time
- Computes sum of 100 million integers in parallel
- Reports execution time in seconds

**Key Learning:**
- How to measure parallel program performance using OpenMP timing functions
- Understanding wall-clock time vs. CPU time
- Foundation for speedup analysis in larger experiments
- Demonstrates that large problem sizes benefit more from parallelization

---

## Summary

These examples progressively introduce:

1. **Problem Identification:** Recognizing race conditions and their consequences
2. **Solution Application:** Using OpenMP reduction to fix synchronization issues  
3. **Performance Analysis:** Measuring execution time to evaluate parallel efficiency

Together, they provide the foundational knowledge needed for the lab questions, where these concepts are applied to more complex problems like DAXPY operations, matrix multiplication, and numerical integration.
