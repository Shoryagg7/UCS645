/*
 * Exercise 3: Finding Maximum and Minimum
 * Each process generates random numbers and finds global max/min
 * Uses MPI_MAXLOC and MPI_MINLOC to track which process has the values
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

typedef struct {
    int value;
    int rank;
} ValueWithRank;

int main(int argc, char *argv[]) {
    int rank, size;
    int local_numbers[10];
    int local_max, local_min;
    ValueWithRank max_data, min_data;
    ValueWithRank global_max, global_min;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Seed random number generator differently for each process
    srand(time(NULL) + rank);

    printf("=== FINDING GLOBAL MAX AND MIN ===\n");
    printf("Process %d: Generating 10 random numbers (0-1000):\n", rank);

    // Generate 10 random numbers and find local max/min
    local_max = -1;
    local_min = 1001;

    for (int i = 0; i < 10; i++) {
        local_numbers[i] = rand() % 1001;  // 0-1000
        printf("  %d", local_numbers[i]);

        if (local_numbers[i] > local_max) {
            local_max = local_numbers[i];
        }
        if (local_numbers[i] < local_min) {
            local_min = local_numbers[i];
        }
    }
    printf("\n");
    printf("Process %d: Local max = %d, Local min = %d\n\n", rank, local_max, local_min);

    // Create MPI datatype for ValueWithRank
    MPI_Datatype mpi_value_rank;
    MPI_Type_contiguous(2, MPI_INT, &mpi_value_rank);
    MPI_Type_commit(&mpi_value_rank);

    // Prepare data with rank information
    max_data.value = local_max;
    max_data.rank = rank;

    min_data.value = local_min;
    min_data.rank = rank;

    // Find global maximum using MPI_MAXLOC
    MPI_Reduce(&max_data, &global_max, 1, mpi_value_rank, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    // Find global minimum using MPI_MINLOC
    MPI_Reduce(&min_data, &global_min, 1, mpi_value_rank, MPI_MINLOC, 0, MPI_COMM_WORLD);

    MPI_Type_free(&mpi_value_rank);

    // Process 0 prints results
    if (rank == 0) {
        printf("=== RESULTS ===\n");
        printf("Global Maximum: %d (found in Process %d)\n", global_max.value, global_max.rank);
        printf("Global Minimum: %d (found in Process %d)\n", global_min.value, global_min.rank);
    }

    MPI_Finalize();
    return 0;
}
