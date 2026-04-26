# LAB5: Advanced MPI Patterns - Performance Analysis & Master-Slave Architecture

## Overview
This lab explores advanced MPI communication patterns including performance optimization, algorithmic efficiency analysis, and the master-slave computation model. You will measure speedup, analyze communication overhead, and understand practical parallel computing limitations.

---

## Exercise 1: DAXPY Loop (Double Precision A·X + Y)

### Problem Description
Implement parallel DAXPY operation on vectors of size 2^16 (65,536 elements). Measure speedup compared to serial implementation. DAXPY is fundamental in linear algebra and scientific computing.

### Operation: X[i] = a·X[i] + Y[i]

```
DAXPY OPERATION VISUALIZATION:
────────────────────────────────────────

Scalar: a = 2.5
Vector X: [x₀, x₁, x₂, x₃, ...]
Vector Y: [y₀, y₁, y₂, y₃, ...]

Element-wise: x₀ = 2.5·x₀ + y₀
              x₁ = 2.5·x₁ + y₁
              x₂ = 2.5·x₂ + y₂
              ...
```

### Parallel Decomposition

```
SERIAL (1 process):
┌─────────────────────────────────────┐
│ FOR i = 0 TO 65535:                 │
│   X[i] = a * X[i] + Y[i]           │
└─────────────────────────────────────┘
      Time = T_serial

PARALLEL (4 processes):
P0: X[0:16383]    = a·X[0:16383]    + Y[0:16383]
P1: X[16384:32767]= a·X[16384:32767]+ Y[16384:32767]
P2: X[32768:49151]= a·X[32768:49151]+ Y[32768:49151]
P3: X[49152:65535]= a·X[49152:65535]+ Y[49152:65535]
      Time = T_parallel
```

### Expected Speedup

```
SPEEDUP = T_serial / T_parallel

Ideal Speedup:  |              Linear (Ideal)
              4 │                   /
              3 │          Actual  /
Speedup       2 │              /
              1 │___________/______
              0 └──┬──┬──┬──┬──
                   1  2  4  8
                   Number of Processes
```

### Communication Pattern

```
SCATTER PHASE:
Master (P0): X_full[65536], Y_full[65536]
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼         ▼
   P0:         P1:           P2:       P3:
  16384      16384          16384     16384
 elements   elements       elements  elements
```

### Key Concepts
- **Embarrassingly Parallel**: Each element computed independently
- **Communication Cost**: Scatter/Gather overhead
- **Speedup Measurement**: Serial vs Parallel execution time
- **Efficiency**: Speedup / Number of Processes

---

## Exercise 2: The Broadcast Race (Linear vs Tree Communication)

### Problem Description
Compare custom linear broadcast (Rank 0 sends to all) vs optimized tree-based MPI_Bcast. Data: 10 million doubles (~80 MB).

### Communication Strategy Comparison

```
LINEAR BROADCAST (Naive):
─────────────────────────────────────────

          P0 (Master)
          │
   ┌──────┼──────┐
   ▼      ▼      ▼       ▼
  P1     P2     P3  ... P15

Time: O(N) = 15 sequential sends
P0 → P1  (wait)
P0 → P2  (wait)
P0 → P3  (wait)
...
P0 → P15 (wait)

Total: ~15 network round-trips (with process count)
```

```
TREE-BASED BROADCAST (Optimized):
──────────────────────────────────────────

            P0
           /  \
         P1    P2
        / \    / \
      P3  P4  P5  P6
      / \ / \ / \ / \
    P7 P8 ... P14 P15

Time: O(log N) = 4 rounds
Round 1: P0 → {P1, P2}
Round 2: {P0,P1} → {P2,P3,P4,P5}
Round 3: All send to remaining processes
Round 4: Complete

Total: ~4 network rounds (logarithmic with process count)
```

### Performance Comparison

```
EXECUTION TIME vs PROCESS COUNT:

Time (seconds)
     │
 0.8 │  Linear Broadcast ────┐
     │                         │    ╱
 0.6 │                     ╱   ╱
     │                 ╱   ╱
 0.4 │             ╱   ╱
     │         ╱   ╱
 0.2 │     ╱───  MPI_Bcast (Tree)
     │  ╱────────────────
   0 └──┬──┬──┬──┬──┬──┬──
        2  4  6  8 12 16
        Number of Processes
```

### Scaling Analysis

```
PROCESS COUNT   LINEAR (O(N))    TREE (O(log N))
───────────────────────────────────────────────
        2         2 sends          1 round
        4         4 sends          2 rounds
        8         8 sends          3 rounds
       16        16 sends          4 rounds
       32        32 sends          5 rounds

Speedup at 16 processes: 
  Linear / Tree = 16 / 4 = 4× improvement
```

### Key Concepts
- **Algorithmic Efficiency**: O(N) vs O(log N)
- **Communication Overhead**: Direct observation
- **MPI Optimization**: Built-in collective operations
- **Theoretical vs Practical**: Understanding MPI implementation

---

## Exercise 3: Distributed Dot Product & Amdahl's Law

### Problem Description
Compute dot product of 500M-element vectors in parallel. Analyze speedup and efficiency using Amdahl's Law. Identify communication vs computation bottlenecks.

### Vector Size & Distribution

```
VECTOR SIZE: 500 MILLION ELEMENTS

Total elements: 500M
4 processes:    125M each
8 processes:    62.5M each

Expected Dot Product = Sum of all (A[i] × B[i])
```

### Communication Flow

```
PHASE 1: BROADCAST MULTIPLIER
──────────────────────────────
Master P0:           scaling_multiplier = 2.0
     │
     ├─ MPI_Bcast ──┤
     │
   P0  P1  P2  P3
 (2.0)(2.0)(2.0)(2.0)

PHASE 2: LOCAL COMPUTATION
──────────────────────────
P0: Generate A[0:125M], B[0:125M]     Compute local_dot
P1: Generate A[125M:250M], B[125M:250M] Compute local_dot
P2: Generate A[250M:375M], B[250M:375M] Compute local_dot
P3: Generate A[375M:500M], B[375M:500M] Compute local_dot

PHASE 3: REDUCTION
──────────────────
   local_dot₀  local_dot₁  local_dot₂  local_dot₃
        │            │            │            │
        └─── MPI_Reduce (MPI_SUM) ────┘
                  │
                  ▼
           global_dot (P0)
```

### Amdahl's Law Analysis

```
SPEEDUP FORMULA:
S(n) = 1 / (f + (1-f)/n)

Where:
  n = number of processors
  f = fraction of serial code

EXECUTION TIME BREAKDOWN:
─────────────────────────────
Total Time = Computation + Communication

T_parallel = T_compute/N + T_comm

Speedup = T_serial / T_parallel
        = T_serial / (T_compute/N + T_comm)
```

### Performance Scaling

```
SPEEDUP WITH AMDAHL'S LAW:

Speedup (Ideal vs Real):

    │ Perfect Linear (S=N)
  8 │        /
    │       / Real (with 10% overhead)
  6 │      /  
    │     /
  4 │    /───────
    │  /
  2 │─/
    │/
  1 └─┬─┬─┬─┬─
    1 2 4 8 16
    Number of Cores

Communication Overhead Effect:
 1-2 cores:  minimal overhead
 4-8 cores:  noticeable (5-15%)
 16+ cores:  significant (20-40%)
```

### Efficiency Calculation

```
EFFICIENCY = Speedup / Number of Processors

Example with 4 processes:
  Speedup = 3.2×
  Efficiency = 3.2 / 4 = 0.8 = 80%
  
Perfect scaling = 100% efficiency (4/4 = 1.0)
Actual < Perfect due to overhead
```

### Key Concepts
- **Amdahl's Law**: Theoretical speedup limit
- **Communication Overhead**: MPI_Bcast and MPI_Reduce cost
- **Parallel Efficiency**: Measure of speedup quality
- **Scaling Limits**: Why perfect speedup is unattainable

---

## Exercise 4: Prime Number Finder (Master-Slave)

### Problem Description
Find all prime numbers up to 10,000. Master distributes work, slaves test for primality. Demonstrates master-slave load balancing.

### Master-Slave Architecture

```
MASTER-SLAVE COMMUNICATION PATTERN:
─────────────────────────────────────────

         MASTER (P0)
          │      ▲
   Send ──┘      └── Receive Result
   Number
          │      ▲
    ┌─────┴─┬────┴──────┬─────┐
    │       │           │     │
    ▼       ▼           ▼     ▼
   SLAVE1  SLAVE2      SLAVE3 SLAVE4
   (P1)    (P2)        (P3)   (P4)

Flow:
1. Slave sends "ready" (0)
2. Master sends number to test
3. Slave tests and sends result
4. Loop until all numbers tested
```

### Work Distribution

```
WORK QUEUE MODEL:
─────────────────────

Initial: Numbers 2-10000 queued

P0 (Master) keeps track of next number = 2

[2, 3, 4, 5, 6, 7, 8, 9, 10, ...]

Send to P1: 2  ──► Result: 2 (prime) ──► Send 5
Send to P2: 3  ──► Result: 3 (prime) ──► Send 6
Send to P3: 4  ──► Result: -4 (not)  ──► Send 7
Send to P4: ... ──► Continue...

This continues until all numbers are processed.
```

### Result Codes

```
SLAVE SENDS BACK:
─────────────────
Positive number  → Number IS PRIME
Negative number  → Number is NOT PRIME
Magnitude = number tested

Example:
  7  → 7 is prime
 -8  → 8 is not prime
 11  → 11 is prime
 -9  → 9 is not prime
```

### Prime Detection Algorithm

```
PRIMALITY TEST (Simple):
────────────────────────

int is_prime(n):
  if n < 2: return FALSE
  if n = 2: return TRUE
  if n % 2 = 0: return FALSE
  for i from 3 to √n by 2:
    if n % i = 0: return FALSE
  return TRUE

Complexity: O(√n)
Example: is_prime(17)
  Check: 17 % 3, 17 % 5
  Both fail → 17 is prime
```

### Primes up to 10,000

```
EXPECTED RESULTS:
─────────────────

Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...
        ..., 9973

Total primes up to 10,000: 1,229 primes

Distribution:
  0-100:     25 primes
  100-1000:  143 primes
  1000-10000: 1,061 primes
```

### Key Concepts
- **Master-Slave Pattern**: Work distribution model
- **Load Balancing**: Slaves work on available tasks
- **Dynamic Scheduling**: Work queue adjusted at runtime
- **Result Encoding**: Positive/negative for prime/not-prime

---

## Exercise 5: Perfect Numbers Finder (Master-Slave)

### Problem Description
Find all perfect numbers up to 10,000. A perfect number equals the sum of its proper divisors. Uses master-slave pattern with dynamic load balancing.

### Perfect Numbers Concept

```
PERFECT NUMBER DEFINITION:
──────────────────────────

A perfect number N equals the sum of its proper divisors
(divisors excluding N itself)

EXAMPLES:

6:  Divisors of 6: 1, 2, 3, 6
    Proper divisors: 1, 2, 3
    Sum: 1 + 2 + 3 = 6 ✓ PERFECT

28: Divisors of 28: 1, 2, 4, 7, 14, 28
    Proper divisors: 1, 2, 4, 7, 14
    Sum: 1 + 2 + 4 + 7 + 14 = 28 ✓ PERFECT

496: Proper divisors: 1, 2, 4, 8, 16, 31, 62, 124, 248
     Sum: 1 + 2 + 4 + 8 + 16 + 31 + 62 + 124 + 248 = 496 ✓

8:   Divisors: 1, 2, 4, 8
     Proper divisors: 1, 2, 4
     Sum: 1 + 2 + 4 = 7 ≠ 8 ✗ NOT PERFECT
```

### Master-Slave Workflow

```
MASTER-SLAVE FOR PERFECT NUMBERS:
──────────────────────────────────────

         MASTER (P0)
       (Number queue: 2-10000)
          │        ▲
   Send ──┤        ├── Return Result
   Number │        │   (±N)
          │        │
    ┌─────┴────────┴───────┐
    │    │    │    │       │
    ▼    ▼    ▼    ▼       ▼
   SLAVE SLAVES... (Compute sum of divisors)

Communication:
1. Slave: Send ready (0) → Master receives
2. Master: Send number N → Slave receives
3. Slave: Compute sum of divisors
4. Slave: Send +N if perfect, -N if not
5. Master: Send next number → Go to step 3
6. Master: Send -1 (terminate) when done
```

### Divisor Computation

```
ALGORITHM: Sum of Proper Divisors
──────────────────────────────────

sum_of_divisors(N):
  if N ≤ 1: return 0
  sum = 1  (1 is always a proper divisor)
  for i from 2 to √N:
    if N % i = 0:
      sum += i
      if i ≠ N/i and N/i ≠ N:
        sum += N/i
  return sum

Example: N = 12
  Check i = 2: 12 % 2 = 0  → add 2 and 6
  Check i = 3: 12 % 3 = 0  → add 3 and 4
  Sum = 1 + 2 + 3 + 4 + 6 = 16

Verification: 16 ≠ 12, so 12 is NOT perfect
```

### Perfect Numbers in Range

```
PERFECT NUMBERS UP TO 10,000:
─────────────────────────────

Total: 3 perfect numbers

1. N = 6
   Divisors: 1 + 2 + 3 = 6 ✓

2. N = 28
   Divisors: 1 + 2 + 4 + 7 + 14 = 28 ✓

3. N = 496
   Divisors: 1 + 2 + 4 + 8 + 16 + 31 + 62 + 124 + 248 = 496 ✓

Next perfect number: 8,128 (>10,000)
Then: 33,550,336 (very large!)

Perfect numbers are EXTREMELY RARE!
```

### Result Encoding

```
RETURN VALUE ENCODING:
──────────────────────

Slave returns:
  +N  if N is a perfect number
  -N  if N is not a perfect number

Master processes:
  if result > 0:
    → Found perfect number
    → Print number and divisors
  if result < 0:
    → Not perfect, continue
```

### Key Concepts
- **Perfect Numbers**: Mathematical properties
- **Rare Occurrences**: Very few perfect numbers exist
- **Dynamic Load Balancing**: Master distributes work
- **Result Verification**: Sum divisors to confirm

---

## Compilation & Execution

### Compilation
```bash
cd LAB5

# Compile all programs
mpicc -Wall -std=c99 -lm -o daxpy daxpy.c
mpicc -Wall -std=c99 -lm -o broadcast_race broadcast_race.c
mpicc -Wall -std=c99 -lm -o dot_product_amdahl dot_product_amdahl.c
mpicc -Wall -std=c99 -lm -o prime_finder prime_finder.c
mpicc -Wall -std=c99 -lm -o perfect_numbers perfect_numbers.c

# Or use Makefile
make all
```

### Running Programs
```bash
# Single runs
mpirun -np 4 ./daxpy
mpirun -np 4 ./broadcast_race
mpirun -np 4 ./dot_product_amdahl
mpirun -np 4 ./prime_finder
mpirun -np 4 ./perfect_numbers

# Performance testing with multiple process counts
mpirun -np 1 ./daxpy
mpirun -np 2 ./daxpy
mpirun -np 4 ./daxpy
mpirun -np 8 ./daxpy

# Test broadcast with varying sizes
for np in 2 4 8 16; do
  echo "Running with $np processes"
  mpirun -np $np ./broadcast_race
done
```

### Using Makefile Targets
```bash
make run-daxpy          # Run Q1 (4 processes)
make run-broadcast      # Run Q2 (4 processes)
make run-dot-product    # Run Q3 (4 processes)
make run-prime          # Run Q4 (4 processes)
make run-perfect        # Run Q5 (4 processes)

make test-daxpy         # Test Q1 with 1,2,4,8 processes
make test-broadcast     # Test Q2 with 2,4,8,16 processes
make test-dot-product   # Test Q3 with 1,2,4,8 processes

make clean              # Remove compiled programs
```

---

## Performance Analysis & Reporting

### Data Collection Template

#### Exercise 1 (DAXPY)
```
Vector Size: 2^16 = 65,536 elements

Processes | Serial Time (s) | MPI Time (s) | Speedup | Efficiency
──────────┼─────────────────┼──────────────┼─────────┼────────────
    1     |     T_s         |     T_s      |  1.0×   |   100%
    2     |     T_s         |     T₂       |  S₂     |   E₂
    4     |     T_s         |     T₄       |  S₄     |   E₄
    8     |     T_s         |     T₈       |  S₈     |   E₈

Calculate: Speedup = T_serial / T_parallel
           Efficiency = Speedup / Processes
```

#### Exercise 2 (Broadcast Race)
```
Data: 10 million doubles (~80 MB)

Processes | Linear Time (s) | MPI_Bcast (s) | Speedup | Improvement
──────────┼─────────────────┼───────────────┼─────────┼─────────────
    2     |      T_lin2     |      T_bc2    |   S₂    |   X₂
    4     |      T_lin4     |      T_bc4    |   S₄    |   X₄
    8     |      T_lin8     |      T_bc8    |   S₈    |   X₈
   16     |     T_lin16     |     T_bc16    |   S₁₆   |   X₁₆

Calculate: Speedup = T_linear / T_mpi_bcast
```

#### Exercise 3 (Dot Product & Amdahl's Law)
```
Vector Size: 500 million elements

Processes | Total Time (s) | Speedup | Efficiency | Overhead
──────────┼────────────────┼─────────┼────────────┼──────────
    1     |     T₁         |   1.0×  |   100%     |    0%
    2     |     T₂         |   S₂    |   S₂/2     |   100-E₂
    4     |     T₄         |   S₄    |   S₄/4     |   100-E₄
    8     |     T₈         |   S₈    |   S₈/8     |   100-E₈

Calculate speedup and efficiency
Identify communication overhead as percentage
```

### Key Metrics
- **Speedup**: How many times faster with N processes
- **Efficiency**: Speedup / N (% of ideal parallelism)
- **Communication Time**: Overhead that limits speedup
- **Scaling**: How speedup changes with process count

---

## Expected Results Summary

| Exercise | Input Size | Expected Output | Key Finding |
|----------|-----------|-----------------|------------|
| DAXPY | 2^16 vectors | X[i] = a·X[i]+Y[i] | Speedup depends on data size |
| Broadcast | 80 MB array | Linear vs Tree times | Tree is O(log N) faster |
| Dot Product | 500M elements | Speedup, Efficiency | Amdahl's Law limiting factor |
| Prime Finder | 2-10,000 | 1,229 primes | Load balancing effectiveness |
| Perfect Numbers | 2-10,000 | 6, 28, 496 | Extreme rarity, slow computation |

---

## Key Learning Outcomes

After completing these exercises, you will understand:

✓ **Performance Measurement**: Using MPI_Wtime() for precise timing  
✓ **Speedup Calculation**: Serial vs parallel execution analysis  
✓ **Communication Patterns**: Broadcast, Reduce, scatter/gather  
✓ **Load Balancing**: Master-slave work distribution  
✓ **Amdahl's Law**: Theoretical limits on parallelism  
✓ **Communication Overhead**: Impact on scaling efficiency  
✓ **Algorithmic Efficiency**: O(N) vs O(log N) in practice  
✓ **Performance Analysis**: Creating graphs and identifying bottlenecks  

---

## Common Performance Issues & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Low speedup | High communication overhead | Increase computation per message |
| Uneven load | Dynamic work assignment issues | Better load balancing strategy |
| Scalability ceiling | Communication dominates | Use faster network or larger problem |
| Unexpected slowdown | Cache/memory effects | Profile with system tools |
| Linear broadcast slower than MPI_Bcast | Obvious! | Use optimized collective ops |

---

## References
- MPI Standard: https://www.mpi-forum.org/
- Amdahl's Law: Understanding parallel speedup limits
- OpenMPI: https://www.open-mpi.org/doc/
- Performance Analysis: Tau, Scalasca, or built-in MPI tools
