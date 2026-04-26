# LAB4: MPI Communication Patterns & Collective Operations

## Overview
This lab explores fundamental MPI communication patterns using collective operations to solve real-world parallel computing problems. Each exercise demonstrates key concepts: point-to-point communication, data distribution, aggregation, and reduction.

---

## Exercise 1: Ring Communication

### Problem Description
Processes communicate in a circular topology, passing a value around the ring where each process modifies the value before forwarding.

### Communication Topology

```
          ┌─────────┐
          │ P0 (100)│
          └────┬────┘
               │ sends to P1
               ▼ (add rank 0)
          ┌─────────┐
          │ P1 (100)│
          └────┬────┘
               │ sends to P2
               ▼ (add rank 1)
          ┌─────────┐
          │ P2 (101)│
          └────┬────┘
               │ sends to P3
               ▼ (add rank 2)
          ┌─────────┐
          │ P3 (103)│
          └────┬────┘
               │ sends to P0
               ▼ (add rank 3)
          ┌─────────┐
          │ P0 (106)│
          └─────────┘
```

### Data Flow Pattern

```
Process Rank  →  Action           →  Value After Action
   P0        →  Initialize       →  100
   P1        →  Receive + Add 1  →  101
   P2        →  Receive + Add 2  →  103
   P3        →  Receive + Add 3  →  106
   P0        →  Receive         →  106 (Final)
```

### Key Concepts
- **Ring Topology**: Circular communication pattern
- **MPI_Sendrecv**: Non-blocking send-receive operation
- **Modulo Arithmetic**: `next_rank = (rank + 1) % size`

### Testing
```
Expected with 4 processes:
  Final Value = 100 + (0 + 1 + 2 + 3) = 106
  Formula: 100 + sum(0 to size-1) = 100 + (size-1)*size/2
```

---

## Exercise 2: Parallel Array Sum

### Problem Description
Distribute an array of 100 elements across processes, compute local sums in parallel, then aggregate results.

### Communication Pattern (MPI_Scatter → MPI_Reduce)

```
SCATTER PHASE:
─────────────────────────────────────────────────────

Master (P0): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... 100]
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼            ▼
   [1-25]      [26-50]      [51-75]     [76-100]
       │            │            │            │
      P0           P1           P2           P3
```

### Local Computation & Reduction

```
REDUCE PHASE:
─────────────────────────────────────────────────────

   P0            P1           P2           P3
Local Sum:     Local Sum:    Local Sum:   Local Sum:
 325            975          1575          2175
    \            /              \            /
     \          /                \          /
      ▼        ▼                  ▼        ▼
       ├─ MPI_SUM ─┤         ├─ MPI_SUM ─┤
       │             │        │             │
       ▼             ▼        ▼             ▼
      1300          3750
       │             │
       └─── MPI_SUM ──┘
            │
            ▼
        GLOBAL SUM
         = 5050 ✓
```

### Array Distribution

```
Array Size: 100 elements (values 1-100)
Processes: 4
Elements per Process: 25

Process 0: elements 1-25      sum = 325
Process 1: elements 26-50     sum = 975
Process 2: elements 51-75     sum = 1575
Process 3: elements 76-100    sum = 2175
                              ───────
                       TOTAL = 5050 ✓
```

### Bonus Feature
- **Average Calculation**: Global Sum ÷ Array Size = 5050 ÷ 100 = 50.5

### Key Concepts
- **MPI_Scatter**: Distribute data from one process to all
- **MPI_Reduce**: Aggregate data from all processes to one using operation (MPI_SUM)
- **Load Balancing**: Equal data distribution among processes

---

## Exercise 3: Finding Maximum and Minimum

### Problem Description
Each process generates random numbers, finds local extrema, then determines global maximum and minimum across all processes using custom reduction operations.

### Data Generation & Local Analysis

```
PROCESS LOCAL ANALYSIS:
─────────────────────────────────────────────────────

P0: [842, 156, 923, 401, 215, 789, 634, 91, 502, 310]
    Local Max: 923  Local Min: 91

P1: [701, 445, 332, 988, 267, 654, 123, 815, 456, 509]
    Local Max: 988  Local Min: 123

P2: [567, 832, 245, 678, 912, 234, 445, 756, 389, 654]
    Local Max: 912  Local Min: 234

P3: [445, 723, 156, 834, 612, 289, 901, 524, 367, 445]
    Local Max: 901  Local Min: 156
```

### Global Reduction with Rank Tracking

```
MAXLOC/MINLOC REDUCTION:
─────────────────────────────────────────────────────

Value + Rank Pairs:

P0: MAX(923, rank 0)    MIN(91, rank 0)
P1: MAX(988, rank 1)    MIN(123, rank 1)
P2: MAX(912, rank 2)    MIN(234, rank 2)
P3: MAX(901, rank 3)    MIN(156, rank 3)
     │                  │
     ├─ MPI_MAXLOC ─┤  ├─ MPI_MINLOC ─┤
     │              │  │              │
     ▼              ▼  ▼              ▼

Global Maximum:        Global Minimum:
  988 (from P1)        91 (from P0)
```

### Result Summary

```
┌─────────────────────────────────┐
│ GLOBAL EXTREMA                  │
├─────────────────────────────────┤
│ Maximum: 988 (Process 1)        │
│ Minimum: 91  (Process 0)        │
└─────────────────────────────────┘
```

### Key Concepts
- **Custom MPI Datatype**: `ValueWithRank` struct containing value and rank
- **MPI_MAXLOC/MPI_MINLOC**: Built-in reduction operations
- **Rank Tracking**: Identifies which process holds the extrema

---

## Exercise 4: Parallel Dot Product

### Problem Description
Compute the dot product of two vectors by distributing portions to each process, computing partial products in parallel, then aggregating.

### Vector Definition

```
Vector A: [1, 2, 3, 4, 5, 6, 7, 8]
Vector B: [8, 7, 6, 5, 4, 3, 2, 1]
           │ │ │ │ │ │ │ │
           × × × × × × × ×
           = = = = = = = =
           8 14 18 20 20 18 14 8  → Sum = 120 ✓
```

### Data Distribution (4 Processes)

```
SCATTER PHASE:
─────────────────────────────────────────────────────

Master: A=[1,2,3,4,5,6,7,8]  B=[8,7,6,5,4,3,2,1]
             │
     ┌───────┼───────┐
     ▼       ▼       ▼       ▼
    P0:     P1:     P2:     P3:
  A=[1,2]  A=[3,4]  A=[5,6]  A=[7,8]
  B=[8,7]  B=[6,5]  B=[4,3]  B=[2,1]
```

### Parallel Computation & Aggregation

```
COMPUTATION & REDUCE PHASE:
─────────────────────────────────────────────────────

Process 0:           Process 1:          Process 2:          Process 3:
1×8 + 2×7           3×6 + 4×5           5×4 + 6×3           7×2 + 8×1
= 8 + 14            = 18 + 20           = 20 + 18           = 14 + 8
= 22                = 38                = 38                = 22

    22     +    38     +    38     +    22
    │           │           │           │
    └─────────── MPI_SUM ───────────────┘
                  │
                  ▼
         Global Dot Product = 120 ✓
```

### Computation Pattern

```
Partial Products:
┌───────┬─────────┬─────────┬─────────┐
│ P0: 22│ P1: 38  │ P2: 38  │ P3: 22  │
└───────┴─────────┴─────────┴─────────┘
           Reduction (SUM)
                  │
                  ▼
              FINAL: 120
```

### Verification

```
Expected A · B = 120
Calculation: (1×8) + (2×7) + (3×6) + (4×5) + (5×4) + (6×3) + (7×2) + (8×1)
           = 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8
           = 120 ✓
```

### Key Concepts
- **MPI_Scatter**: Distribute both vectors
- **Embarrassingly Parallel**: Each process computes independently
- **MPI_Reduce**: Aggregate partial products with MPI_SUM

---

## Compilation & Execution

### Compilation
```bash
# Compile all programs
make

# Or compile individual programs
make ring_comm
make array_sum
make max_min
make dot_product
```

### Running Programs
```bash
# Run with 4 processes (standard)
mpirun -np 4 ./ring_comm
mpirun -np 4 ./array_sum
mpirun -np 4 ./max_min
mpirun -np 4 ./dot_product

# Run with different process counts
mpirun -np 2 ./ring_comm
mpirun -np 8 ./array_sum

# Run all exercises
make run-all
```

### Cleanup
```bash
make clean
```

---

## Performance Characteristics

### Communication Overhead

```
RING COMMUNICATION:
Processes: 1        2        4        8
Hops:      0   │    1   │    3   │    7
           └────────────────────────── O(n)

SCATTER/REDUCE:
Processes: 1        2        4        8
Rounds:    0   │    1   │    2   │    3
           └────────────────────────── O(log n)
```

### Scalability Illustration

```
STRONG SCALING (Fixed Problem Size):
Speedup
  │
 8├─────────────────────── Linear
  │    ╱
  │   ╱  Ideal
  │  ╱
 4├─────
  │  ╱   Actual
  │ ╱     ╱
 2├─────────
  │╱       ╱
  1└──────────────
    1  2  4  8
    Number of Processes
```

---

## Key MPI Concepts Summary

### Communication Patterns Used

```
┌──────────────────────────────────┐
│ COLLECTIVE OPERATIONS            │
├──────────────────────────────────┤
│ MPI_Scatter  → One-to-Many      │
│ MPI_Reduce   → Many-to-One      │
│ MPI_Sendrecv → Bidirectional    │
│ MPI_Barrier  → Synchronization  │
└──────────────────────────────────┘
```

### Learning Outcomes

After completing these exercises, you will understand:
- ✓ Point-to-point communication patterns (Ring topology)
- ✓ Data distribution strategies (Scatter)
- ✓ Parallel computation decomposition
- ✓ Aggregation and reduction operations
- ✓ Custom MPI datatypes for complex data
- ✓ Synchronization and load balancing
- ✓ Writing efficient parallel algorithms

---

## Expected Output Examples

### Ring Communication (4 processes)
```
Process 0: Initial value = 100
Process 0: Received value = 100 (added 0 to it)
Process 1: Received value = 101 (added 1 to it)
Process 2: Received value = 103 (added 2 to it)
Process 3: Received value = 106 (added 3 to it)

FINAL RESULT:
Process 0: Final value returned = 106
```

### Array Sum (4 processes)
```
RESULTS:
Global sum: 5050
Expected sum: 5050
✓ CORRECT!
Bonus - Average value: 50.50
```

### Max/Min (4 processes)
```
RESULTS:
Global Maximum: 988 (found in Process 1)
Global Minimum: 91 (found in Process 0)
```

### Dot Product (4 processes)
```
RESULTS:
Global dot product: 120
Expected result: 120
✓ CORRECT!
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Compilation error | Ensure `mpicc` is in PATH; install OpenMPI/MPICH |
| Array division error | Use process count that divides array size evenly |
| Unpredictable results in max_min | Random seeds differ per process (intentional) |
| Slow performance | Reduce process count or problem size for testing |

---

## References
- MPI Standard: https://www.mpi-forum.org/
- OpenMPI Documentation: https://www.open-mpi.org/doc/
- Exercise Concepts: Distributed parallel computing with collective operations
