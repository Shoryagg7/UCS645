/*
 * Exercise 1: Ring Communication
 * Each process sends a message to the next process in a ring topology
 * Process 0 sends to Process 1, ..., last process sends to Process 0
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value = 100;  // Initial value in Process 0
    int received_value;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate neighbors in ring topology
    int next_rank = (rank + 1) % size;
    int prev_rank = (rank - 1 + size) % size;

    // Process 0 starts with initial value 100
    if (rank == 0) {
        value = 100;
        printf("Process %d: Initial value = %d\n", rank, value);
    }

    // Send to next process and receive from previous process
    // Using sendrecv to avoid deadlock
    value += rank;  // Each process adds its rank
    MPI_Sendrecv_replace(&value, 1, MPI_INT, next_rank, 0,
                         prev_rank, 0, MPI_COMM_WORLD, &status);

    printf("Process %d: Received value = %d (added %d to it)\n", rank, value, rank);

    // Synchronize before final message
    MPI_Barrier(MPI_COMM_WORLD);

    // Process 0 waits to receive the final value
    if (rank == 0) {
        MPI_Recv(&received_value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, &status);
        printf("\n=== FINAL RESULT ===\n");
        printf("Process 0: Final value returned = %d\n", received_value);
        printf("Expected: 100 + 0 + 1 + 2 + 3 + ... + (size-1) = %d\n",
               100 + (size - 1) * size / 2);
    } else if (rank == prev_rank) {
        // Last process in ring sends back to Process 0
        MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
