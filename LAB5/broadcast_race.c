/*
 * Question 2: The Broadcast Race (Linear vs Tree Communication)
 * Compare custom MPI_Send loop vs optimized MPI_Bcast
 * Data: 10 million doubles (~80 MB)
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int array_size = 10000000;  // 10 million doubles (~80 MB)
    double *data;
    double start_time, end_time;
    double linear_time = 0.0, bcast_time = 0.0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate large array
    data = (double *)malloc(array_size * sizeof(double));

    printf("=== BROADCAST RACE ===\n");
    if (rank == 0) {
        printf("Array size: %d elements (%.1f MB)\n", array_size,
               (array_size * sizeof(double)) / (1024.0 * 1024.0));
        printf("Number of processes: %d\n\n", size);

        // Initialize data on rank 0
        for (int i = 0; i < array_size; i++) {
            data[i] = (double)i * 1.5 + 2.3;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ============================================
    // PART A: Custom Broadcast (Linear - Rank 0 sends to all)
    // ============================================
    printf("Part A: Custom Linear Broadcast (MPI_Send loop)\n");
    printf("========================================\n");

    start_time = MPI_Wtime();

    if (rank == 0) {
        // Rank 0 sends to all other ranks
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(data, array_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        // Other ranks receive from rank 0
        MPI_Recv(data, array_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    linear_time = end_time - start_time;

    if (rank == 0) {
        printf("Linear Broadcast Time: %.6f seconds\n\n", linear_time);
    }

    // Reset and synchronize
    MPI_Barrier(MPI_COMM_WORLD);

    // ============================================
    // PART B: Optimized MPI_Bcast (Tree-based)
    // ============================================
    printf("Part B: Optimized MPI_Bcast (Tree-based)\n");
    printf("=======================================\n");

    // Reinitialize data on rank 0
    if (rank == 0) {
        for (int i = 0; i < array_size; i++) {
            data[i] = (double)i * 1.5 + 2.3;
        }
    }

    start_time = MPI_Wtime();

    // Use optimized MPI_Bcast
    MPI_Bcast(data, array_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    bcast_time = end_time - start_time;

    if (rank == 0) {
        printf("MPI_Bcast Time: %.6f seconds\n\n", bcast_time);
    }

    // ============================================
    // RESULTS AND ANALYSIS
    // ============================================
    if (rank == 0) {
        printf("=== PERFORMANCE ANALYSIS ===\n");
        printf("Linear Broadcast Time:  %.6f seconds\n", linear_time);
        printf("MPI_Bcast Time:         %.6f seconds\n", bcast_time);
        printf("Improvement Factor:     %.2f×\n", linear_time / bcast_time);
        printf("Time Saved:             %.2f%%\n",
               ((linear_time - bcast_time) / linear_time) * 100.0);

        printf("\n=== THEORETICAL ANALYSIS ===\n");
        printf("Linear approach:  O(N) = %d sends\n", size - 1);
        printf("Tree approach:    O(log N) = %d rounds\n",
               (int)(log(size) / log(2) + 1));

        printf("\nScaling comparison:\n");
        printf("  Linear:  Time increases linearly with process count\n");
        printf("  Tree:    Time increases logarithmically with process count\n");

        printf("\nVerification (first 5 elements): %.2f, %.2f, %.2f, %.2f, %.2f\n",
               data[0], data[1], data[2], data[3], data[4]);
    }

    free(data);
    MPI_Finalize();
    return 0;
}
