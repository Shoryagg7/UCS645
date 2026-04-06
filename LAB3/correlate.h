#ifndef CORRELATE_H
#define CORRELATE_H

// Interface for correlation coefficient calculations
// Calculates correlation between all pairs of input vectors (matrix rows)
//
// Parameters:
//   ny: Number of rows (vectors)
//   nx: Number of columns (dimension of each vector)
//   data: Input matrix (ny rows × nx columns, row-major)
//   result: Output correlation matrix (ny × ny, lower triangular)

// Sequential baseline - correctness reference
void correlate_sequential(int ny, int nx, const float *data, float *result);

// Parallel with OpenMP - basic parallelization
void correlate_parallel(int ny, int nx, const float *data, float *result);

// Optimized - SIMD vectorization and blocking
void correlate_optimized(int ny, int nx, const float *data, float *result);

// Highly optimized - aligned memory, aggressive SIMD, cache blocking
void correlate_highly_optimized(int ny, int nx, const float *data, float *result);

#endif
