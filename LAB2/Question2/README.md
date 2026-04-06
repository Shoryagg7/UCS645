# Question 2: Bioinformatics - DNA Sequence Alignment (Smith-Waterman)

## Problem Statement

Implement a parallel version of the Smith-Waterman local sequence alignment algorithm for comparing two DNA sequences. The Smith-Waterman algorithm finds the optimal local alignment between two sequences, which is crucial for discovering similar protein or DNA subsequences.

## Algorithm Overview

The Smith-Waterman algorithm uses dynamic programming to find the highest-scoring local alignment:

$$H(i,j) = \max\begin{cases}
0 \\
H(i-1,j-1) + s(a_i, b_j) & \text{match/mismatch} \\
H(i-1,j) + d & \text{deletion} \\
H(i,j-1) + d & \text{insertion}
\end{cases}$$

Where:
- **H(i,j)**: Alignment score at position (i,j)
- **s(a_i, b_j)**: Similarity score (+2 for match, -1 for mismatch)
- **d**: Gap penalty (-1)

## Implementation Details

### Key Features

1. **Scoring Matrix**: (M+1) × (N+1) matrix where M and N are sequence lengths
2. **Recurrence Relation**: Each cell depends on three neighbors (diagonal, top, left)
3. **Anti-dependencies**: Critical challenge - (i,j) depends on (i-1,j-1), (i-1,j), and (i,j-1)
4. **Local Alignment**: Returns highest score in matrix (unlike global Needleman-Wunsch)

### Data Dependencies

```
Matrix computation pattern (dependencies shown as arrows):
    j-1    j
i-1  X  <- H(i-1,j-1)
     ^  |
     |  v
     <- H(i,j-1)  H(i,j)
```

This anti-diagonal dependency structure is crucial for parallelization!

## Parallelization Strategies

### Strategy 1: Wavefront Parallelization (Implemented)

Process matrix along anti-diagonals (diagonals). Cells on the same anti-diagonal are independent:

```cpp
for (int diagonal = 2; diagonal < rows + cols; diagonal++) {
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 1; i < rows; i++) {
        int j = diagonal - i;
        if (j >= 1 && j < cols) {
            // Compute H(i,j) safely - all dependencies available
            int match = matrix[i-1][j-1] + score;
            int delete_op = matrix[i-1][j] + GAP;
            int insert_op = matrix[i][j-1] + GAP;
            matrix[i][j] = max({0, match, delete_op, insert_op});
        }
    }
}
```

**Advantages:**
- Natural parallelism along anti-diagonals
- No synchronization needed between parallel iterations
- Dynamic scheduling handles load imbalance

**Complexity:**
- Total diagonals: O(M+N)
- Cells per diagonal: varies from 1 to min(M,N)
- Parallel overhead amortized across many cells

### Strategy 2: Hybrid Approach (Advanced)

Combine row-level and column-level parallelism with strategic blocking.

## Compilation and Execution

### Compile
```bash
g++ -fopenmp -O3 -std=c++11 q2.cpp -o q2
```

### Run
```bash
./q2
```

### Output
- Matrix size: 2001 × 2001 (2000 length sequences + 1 for initialization)
- Results saved to `smithwaterman_results.txt`
- Format: `Threads, Time, Speedup, Max_Score`

## Scheduling Analysis

Three scheduling strategies recommended for experimentation:

### 1. Static Scheduling
- Fixed chunk size assigned to threads
- Predictable but may have load imbalance
- Minimal synchronization overhead

### 2. Dynamic Scheduling
- Chunks assigned at runtime as threads become free
- Better load balancing but more overhead
- Use smaller chunk sizes for fine-grained load distribution

### 3. Guided Scheduling
- Chunk size decreases over time
- Balance between static and dynamic efficiency
- Good for loop iterations with varying costs

## Performance Considerations

### Anti-dependency Challenges
The recurrence relation creates a **wavefront** of computations:
- Cannot parallelize arbitrary row or column positions
- Must process in specific order to respect dependencies
- Wavefront approach ensures correctness while enabling parallelism

### Communication Pattern
- **Stage 1** (diagonal 1): 1 cell computed
- **Stage 2** (diagonal 2): 2 cells computed in parallel ✅
- ...
- **Stage k** (diagonal k): min(k, M, N) cells in parallel ✅
- **Stage (M+N-1)** (diagonal final): 1 cell computed

### Speedup Expectations
- Early diagonals: Low parallelism (≤4 cells)
- Middle diagonals: High parallelism (≈min(M,N)/2 cells) ✅
- Late diagonals: Low parallelism (≤4 cells)
- **Overall speedup**: Limited to average parallelism across all stages

## Algorithm Extensions

### Local vs Global Alignment
Current implementation (local):
```cpp
matrix[i][j] = max({0, match, delete_op, insert_op});
                 ↑
              Reset to 0
```

For global alignment (Needleman-Wunsch), remove the `0` option.

### Backtracking (Traceback)
Find optimal local alignment:
1. Locate cell with maximum score
2. Traceback while score > 0:
   - Move diagonally if came from match
   - Move up if came from deletion
   - Move left if came from insertion

## Optimization Opportunities

1. **Cache Blocking**: Process blocks of anti-diagonals to improve cache locality
2. **SIMD**: Vectorize similarity score calculations
3. **Cutoff Threshold**: Stop computations below score threshold
4. **GPU Offloading**: Massive parallelism on GPU (100s of threads)

## References

- Smith-Waterman Algorithm: https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
- Sequence Alignment: https://www.ncbi.nlm.nih.gov/Class/NAWBIS/Modules/Alignment/align_intro.html
- OpenMP Wavefront Patterns: https://www.openmp.org/

---

**Lab**: UCS645 - Parallel & Distributed Computing
**Assignment**: LAB2 - Question 2
**Date**: 2026
