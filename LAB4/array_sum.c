/*
 * Exercise 2: Parallel Array Sum
 * Distribute array portions using MPI_Scatter, compute local sums,
 * and combine them using MPI_Reduce
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int array_size = 100;
    int *full_array = NULL;
    int *local_array;
    int local_sum = 0;
    int global_sum = 0;
    double average;
    int local_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verify array can be evenly divided
    if (array_size % size != 0) {
        if (rank == 0) {
            printf("Error: Array size (%d) must be divisible by number of processes (%d)\n",
                   array_size, size);
        }
        MPI_Finalize();
        return 1;
    }

    local_count = array_size / size;
    local_array = (int *)malloc(local_count * sizeof(int));

    // Process 0 creates and initializes the array
    if (rank == 0) {
        full_array = (int *)malloc(array_size * sizeof(int));
        printf("=== PARALLEL ARRAY SUM ===\n");
        printf("Array size: %d, Number of processes: %d\n", array_size, size);
        printf("Each process gets: %d elements\n\n", local_count);

        // Initialize array with values 1 to 100
        for (int i = 0; i < array_size; i++) {
            full_array[i] = i + 1;
        }
        printf("Process 0: Array created [1, 2, 3, ..., 100]\n");
    }

    // MPI_Scatter: Distribute array portions to all processes
    MPI_Scatter(full_array, local_count, MPI_INT,
                local_array, local_count, MPI_INT,
                0, MPI_COMM_WORLD);

    // Print local portions
    printf("Process %d: Received elements from index %d to %d\n",
           rank, rank * local_count, (rank + 1) * local_count - 1);

    // Each process computes its local sum
    for (int i = 0; i < local_count; i++) {
        local_sum += local_array[i];
    }
    printf("Process %d: Local sum = %d\n", rank, local_sum);

    // MPI_Reduce: Combine all local sums to compute global sum
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 prints the result
    if (rank == 0) {
        printf("\n=== RESULTS ===\n");
        printf("Global sum: %d\n", global_sum);
        printf("Expected sum: 5050\n");

        // Verify result
        if (global_sum == 5050) {
            printf("✓ CORRECT!\n");
        } else {
            printf("✗ INCORRECT!\n");
        }

        // Bonus: Compute and print average
        average = (double)global_sum / array_size;
        printf("\nBonus - Average value: %.2f\n", average);

        free(full_array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
