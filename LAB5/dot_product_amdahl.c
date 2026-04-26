/*
 * Question 3: Distributed Dot Product & Amdahl's Law
 * Use MPI_Bcast and MPI_Reduce
 * Analyze Speedup and Parallel Efficiency
 * Vectors: 500 million elements each
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size;
    long total_size = 500000000;  // 500 million elements
    double scaling_multiplier = 1.0;
    long local_n;
    double *local_a, *local_b;
    double local_dot_product = 0.0;
    double global_dot_product = 0.0;
    double total_time, compute_time, comm_time;
    double start_time, end_time;
    double speedup, efficiency;
    static double serial_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verify total_size is divisible by size
    if (total_size % size != 0) {
        if (rank == 0) {
            printf("Error: Total size must be divisible by number of processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    local_n = total_size / size;

    printf("=== DISTRIBUTED DOT PRODUCT & AMDAHL'S LAW ===\n");

    // ============================================
    // Rank 0 reads multiplier (simulated)
    // ============================================
    if (rank == 0) {
        printf("Total vector size: %ld elements (500M)\n", total_size);
        printf("Number of processes: %d\n", size);
        printf("Elements per process: %ld\n\n", local_n);
        scaling_multiplier = 2.0;  // Default multiplier
    }

    // ============================================
    // COMMUNICATION: MPI_Bcast scaling multiplier
    // ============================================
    start_time = MPI_Wtime();
    MPI_Bcast(&scaling_multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - start_time;

    // Allocate local vectors
    local_a = (double *)malloc(local_n * sizeof(double));
    local_b = (double *)malloc(local_n * sizeof(double));

    // ============================================
    // COMPUTATION: Generate local vectors
    // ============================================
    start_time = MPI_Wtime();

    for (long i = 0; i < local_n; i++) {
        local_a[i] = 1.0;
        local_b[i] = 2.0 * scaling_multiplier;
    }

    // ============================================
    // COMPUTATION: Local dot product
    // ============================================
    for (long i = 0; i < local_n; i++) {
        local_dot_product += local_a[i] * local_b[i];
    }

    compute_time = MPI_Wtime() - start_time;

    // ============================================
    // COMMUNICATION: MPI_Reduce to global sum
    // ============================================
    start_time = MPI_Wtime();
    MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    double reduce_time = MPI_Wtime() - start_time;
    comm_time += reduce_time;

    total_time = compute_time + comm_time;

    // ============================================
    // SERIAL COMPUTATION (Rank 0 only for reference)
    // ============================================
    if (rank == 0) {
        double *full_a = (double *)malloc(total_size * sizeof(double));
        double *full_b = (double *)malloc(total_size * sizeof(double));

        start_time = MPI_Wtime();
        for (long i = 0; i < total_size; i++) {
            full_a[i] = 1.0;
            full_b[i] = 2.0 * scaling_multiplier;
        }
        double serial_dot = 0.0;
        for (long i = 0; i < total_size; i++) {
            serial_dot += full_a[i] * full_b[i];
        }
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;

        free(full_a);
        free(full_b);

        // ============================================
        // RESULTS AND ANALYSIS
        // ============================================
        printf("=== EXECUTION TIMES ===\n");
        printf("Serial time (1 process):      %.6f seconds\n", serial_time);
        printf("Parallel time (%d processes): %.6f seconds\n", size, total_time);
        printf("  - Computation time:        %.6f seconds\n", compute_time);
        printf("  - Communication time:      %.6f seconds\n", comm_time);

        printf("\n=== SPEEDUP & EFFICIENCY ===\n");
        if (size > 1) {
            speedup = serial_time / total_time;
            efficiency = speedup / size;
            printf("Speedup (S): %.2f×\n", speedup);
            printf("Efficiency (E): %.2f%% (%.4f)\n", efficiency * 100, efficiency);

            printf("\n=== AMDAHL'S LAW ANALYSIS ===\n");
            printf("Ideal speedup for %d cores: %.2f×\n", size, (double)size);
            printf("Actual speedup: %.2f×\n", speedup);
            printf("Lost speedup: %.2f× (%.1f%%)\n", size - speedup,
                   ((size - speedup) / size) * 100);

            printf("\nOverhead Analysis:\n");
            printf("  Communication/Computation Ratio: %.2f%%\n",
                   (comm_time / compute_time) * 100);
            printf("  Parallel fraction: %.2f%%\n", 100.0 - (comm_time / total_time) * 100);
        }

        printf("\n=== VERIFICATION ===\n");
        printf("Expected dot product: %.2e\n", (double)total_size * 2.0 * scaling_multiplier);
        printf("Computed dot product: %.2e\n", global_dot_product);
        printf("Difference: %.2e\n",
               fabs((double)total_size * 2.0 * scaling_multiplier - global_dot_product));
    }

    free(local_a);
    free(local_b);
    MPI_Finalize();
    return 0;
}
