/*
 * Question 4: Find all positive primes up to maximum value
 * Master-Slave pattern using MPI_Recv and MPI_Send
 * Master distributes numbers, slaves test for primality
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Function to check if a number is prime
int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i <= sqrt(n); i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int max_value = 10000;  // Find primes up to 10,000
    int number_to_test;
    int prime_count = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least 2 processes (1 master + 1 slave)\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // MASTER PROCESS
        printf("=== PRIME NUMBER FINDER (Master-Slave Pattern) ===\n");
        printf("Finding all primes up to %d\n", max_value);
        printf("Master: rank 0\n");
        printf("Slaves: ranks 1 to %d\n\n", size - 1);

        int next_number = 2;

        // Main loop: Master distributes work
        for (int i = 1; i < size; i++) {
            // Send initial numbers to all slaves
            MPI_Send(&next_number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            next_number++;
        }

        while (next_number <= max_value) {
            // Receive result from any slave (MPI_ANY_SOURCE)
            int result;
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int slave_rank = status.MPI_SOURCE;

            // Process result
            if (result > 0) {
                prime_count++;
                printf("Prime found: %d (from slave %d)\n", result, slave_rank);
            }

            // Send next number to the slave that just finished
            if (next_number <= max_value) {
                MPI_Send(&next_number, 1, MPI_INT, slave_rank, 0, MPI_COMM_WORLD);
                next_number++;
            } else {
                // Send termination signal (-1)
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, slave_rank, 0, MPI_COMM_WORLD);
            }
        }

        // Send termination signals to remaining slaves
        for (int i = 1; i < size; i++) {
            int result;
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            if (result > 0) {
                prime_count++;
                printf("Prime found: %d (from slave %d)\n", result, status.MPI_SOURCE);
            }
            int terminate = -1;
            MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }

        printf("\n=== RESULTS ===\n");
        printf("Total primes found up to %d: %d\n", max_value, prime_count);

    } else {
        // SLAVE PROCESS
        while (1) {
            // Send request for next number (0 for initial request)
            int request = 0;
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Receive number to test
            MPI_Recv(&number_to_test, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            // Check termination signal
            if (number_to_test < 0) {
                break;
            }

            // Test for primality
            if (is_prime(number_to_test)) {
                // Send positive number if prime
                MPI_Send(&number_to_test, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            } else {
                // Send negative number if not prime
                int not_prime = -number_to_test;
                MPI_Send(&not_prime, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
