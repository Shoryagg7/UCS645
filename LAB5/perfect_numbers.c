/*
 * Question 5: Find all perfect numbers up to maximum value
 * Perfect numbers = sum of their proper divisors
 * Example: 6 = 1 + 2 + 3 (proper divisors of 6)
 * Master-Slave pattern using MPI_Recv and MPI_Send
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to calculate sum of proper divisors
int sum_of_divisors(int n) {
    if (n <= 1) return 0;
    int sum = 1;  // 1 is always a proper divisor for n > 1
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i && n / i != n) {
                sum += n / i;
            }
        }
    }
    return sum;
}

// Check if number is perfect
int is_perfect(int n) {
    return sum_of_divisors(n) == n;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int max_value = 10000;  // Find perfect numbers up to 10,000
    int number_to_test;
    int perfect_count = 0;
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
        printf("=== PERFECT NUMBER FINDER (Master-Slave Pattern) ===\n");
        printf("Finding all perfect numbers up to %d\n", max_value);
        printf("Master: rank 0\n");
        printf("Slaves: ranks 1 to %d\n\n", size - 1);

        printf("Perfect Numbers are numbers that equal the sum of their proper divisors.\n");
        printf("Examples: 6 = 1+2+3, 28 = 1+2+4+7+14\n\n");

        int next_number = 2;

        // Send initial numbers to all slaves (0 = initial request)
        for (int i = 1; i < size; i++) {
            MPI_Send(&next_number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            next_number++;
        }

        while (next_number <= max_value) {
            // Receive result from any slave (MPI_ANY_SOURCE)
            int result;
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int slave_rank = status.MPI_SOURCE;

            // Process result (positive = perfect, negative = not perfect)
            if (result > 0) {
                perfect_count++;
                int divisor_sum = sum_of_divisors(result);
                printf("Perfect number found: %d = ", result);

                // Print divisors
                int first = 1;
                for (int i = 1; i < result; i++) {
                    if (result % i == 0) {
                        if (!first) printf(" + ");
                        printf("%d", i);
                        first = 0;
                    }
                }
                printf(" (from slave %d)\n", slave_rank);
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
                perfect_count++;
                int divisor_sum = sum_of_divisors(result);
                printf("Perfect number found: %d = ", result);
                int first = 1;
                for (int i = 1; i < result; i++) {
                    if (result % i == 0) {
                        if (!first) printf(" + ");
                        printf("%d", i);
                        first = 0;
                    }
                }
                printf(" (from slave %d)\n", result, status.MPI_SOURCE);
            }
            int terminate = -1;
            MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }

        printf("\n=== RESULTS ===\n");
        printf("Total perfect numbers found up to %d: %d\n", max_value, perfect_count);
        printf("\nNote: In the range 1-10000, only 3 perfect numbers exist:\n");
        printf("  6, 28, 496\n");

    } else {
        // SLAVE PROCESS
        while (1) {
            // Send initial request (0) for the first number
            int request = 0;
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Receive number to test
            MPI_Recv(&number_to_test, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            // Check termination signal
            if (number_to_test < 0) {
                break;
            }

            // Test for perfect property
            if (is_perfect(number_to_test)) {
                // Send positive number if perfect
                MPI_Send(&number_to_test, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            } else {
                // Send negative number if not perfect
                int not_perfect = -number_to_test;
                MPI_Send(&not_perfect, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
