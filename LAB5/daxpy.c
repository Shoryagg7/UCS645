/*
 * Question 1: DAXPY Loop
 * D = Double precision, A = scalar, X,Y = vectors
 * Operation: X[i] = a*X[i] + Y[i]
 * Measure speedup: MPI vs Uniprocessor
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int n = (1 << 16);  // 2^16 = 65536 elements
    double scalar_a = 2.5;
    double *x_local, *y_local;
    double *x_full = NULL, *y_full = NULL;
    int local_n;
    double start_time, end_time;
    double mpi_time, serial_time;
    double speedup;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verify n is divisible by size
    if (n % size != 0) {
        if (rank == 0) {
            printf("Error: Vector size (%d) must be divisible by processes (%d)\n", n, size);
        }
        MPI_Finalize();
        return 1;
    }

    local_n = n / size;

    // Allocate local arrays
    x_local = (double *)malloc(local_n * sizeof(double));
    y_local = (double *)malloc(local_n * sizeof(double));

    // Rank 0 initializes and broadcasts scalar
    MPI_Bcast(&scalar_a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        x_full = (double *)malloc(n * sizeof(double));
        y_full = (double *)malloc(n * sizeof(double));

        printf("=== DAXPY OPERATION ===\n");
        printf("Vector size: %d (2^16)\n", n);
        printf("Scalar value: %.2f\n", scalar_a);
        printf("Number of processes: %d\n\n", size);

        // Initialize vectors
        for (int i = 0; i < n; i++) {
            x_full[i] = (double)i + 1.0;
            y_full[i] = (double)(n - i);
        }
    }

    // Scatter vectors
    MPI_Scatter(x_full, local_n, MPI_DOUBLE, x_local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y_full, local_n, MPI_DOUBLE, y_local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI DAXPY Operation
    start_time = MPI_Wtime();

    for (int i = 0; i < local_n; i++) {
        x_local[i] = scalar_a * x_local[i] + y_local[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    mpi_time = end_time - start_time;

    // Gather results
    double *result = NULL;
    if (rank == 0) {
        result = (double *)malloc(n * sizeof(double));
    }
    MPI_Gather(x_local, local_n, MPI_DOUBLE, result, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Serial computation on rank 0 (for comparison)
    if (rank == 0) {
        // Reinitialize for serial computation
        for (int i = 0; i < n; i++) {
            x_full[i] = (double)i + 1.0;
            y_full[i] = (double)(n - i);
        }

        start_time = MPI_Wtime();
        for (int i = 0; i < n; i++) {
            x_full[i] = scalar_a * x_full[i] + y_full[i];
        }
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;

        // Calculate speedup
        speedup = serial_time / mpi_time;

        printf("=== RESULTS ===\n");
        printf("Serial time:     %.6f seconds\n", serial_time);
        printf("MPI time (p=%d):  %.6f seconds\n", size, mpi_time);
        printf("Speedup:         %.2f×\n", speedup);
        printf("Efficiency:      %.2f%%\n", (speedup / size) * 100.0);

        // Verify some values
        printf("\nVerification (first 5 elements):\n");
        for (int i = 0; i < 5 && i < n; i++) {
            printf("  result[%d] = %.2f (expected: %.2f)\n", i, result[i],
                   scalar_a * (i + 1.0) + (n - i));
        }

        free(x_full);
        free(y_full);
        free(result);
    }

    free(x_local);
    free(y_local);
    MPI_Finalize();
    return 0;
}
