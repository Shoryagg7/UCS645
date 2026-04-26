/*
 * Exercise 4: Parallel Dot Product
 * Compute the dot product of two vectors in parallel
 * Vector A = [1, 2, 3, 4, 5, 6, 7, 8]
 * Vector B = [8, 7, 6, 5, 4, 3, 2, 1]
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int *vector_a = NULL;
    int *vector_b = NULL;
    int *local_a;
    int *local_b;
    int vector_size = 8;
    int local_size;
    int local_product = 0;
    int global_product = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verify vector can be evenly divided
    if (vector_size % size != 0) {
        if (rank == 0) {
            printf("Error: Vector size (%d) must be divisible by number of processes (%d)\n",
                   vector_size, size);
        }
        MPI_Finalize();
        return 1;
    }

    local_size = vector_size / size;
    local_a = (int *)malloc(local_size * sizeof(int));
    local_b = (int *)malloc(local_size * sizeof(int));

    // Process 0 creates and initializes vectors
    if (rank == 0) {
        vector_a = (int *)malloc(vector_size * sizeof(int));
        vector_b = (int *)malloc(vector_size * sizeof(int));

        printf("=== PARALLEL DOT PRODUCT ===\n");
        printf("Vector size: %d, Number of processes: %d\n", vector_size, size);
        printf("Each process gets: %d elements\n\n", local_size);

        // Initialize vectors
        // A = [1, 2, 3, 4, 5, 6, 7, 8]
        // B = [8, 7, 6, 5, 4, 3, 2, 1]
        for (int i = 0; i < vector_size; i++) {
            vector_a[i] = i + 1;
            vector_b[i] = vector_size - i;
        }

        printf("Vector A: ");
        for (int i = 0; i < vector_size; i++) {
            printf("%d ", vector_a[i]);
        }
        printf("\nVector B: ");
        for (int i = 0; i < vector_size; i++) {
            printf("%d ", vector_b[i]);
        }
        printf("\n\n");
    }

    // MPI_Scatter: Distribute vector portions to all processes
    MPI_Scatter(vector_a, local_size, MPI_INT,
                local_a, local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(vector_b, local_size, MPI_INT,
                local_b, local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // Print local portions
    printf("Process %d: Elements %d to %d\n", rank, rank * local_size, (rank + 1) * local_size - 1);
    printf("  A: ");
    for (int i = 0; i < local_size; i++) {
        printf("%d ", local_a[i]);
    }
    printf("\n  B: ");
    for (int i = 0; i < local_size; i++) {
        printf("%d ", local_b[i]);
    }
    printf("\n");

    // Each process computes its partial dot product
    for (int i = 0; i < local_size; i++) {
        local_product += local_a[i] * local_b[i];
    }
    printf("Process %d: Partial dot product = %d\n", rank, local_product);

    // MPI_Reduce: Sum all partial products
    MPI_Reduce(&local_product, &global_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 prints the result
    if (rank == 0) {
        printf("\n=== RESULTS ===\n");
        printf("Global dot product: %d\n", global_product);
        printf("Expected result: 120\n");

        // Verify result
        // A · B = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
        if (global_product == 120) {
            printf("✓ CORRECT!\n");
        } else {
            printf("✗ INCORRECT!\n");
        }

        free(vector_a);
        free(vector_b);
    }

    free(local_a);
    free(local_b);
    MPI_Finalize();
    return 0;
}
