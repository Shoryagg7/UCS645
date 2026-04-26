# Makefile for MPI Exercises LAB4

# Compiler
CC = mpicc
CFLAGS = -Wall -std=c99

# Executables
TARGETS = ring_comm array_sum max_min dot_product

# Default target
all: $(TARGETS)

# Compile ring_comm
ring_comm: ring_comm.c
	$(CC) $(CFLAGS) -o ring_comm ring_comm.c

# Compile array_sum
array_sum: array_sum.c
	$(CC) $(CFLAGS) -o array_sum array_sum.c

# Compile max_min
max_min: max_min.c
	$(CC) $(CFLAGS) -o max_min max_min.c

# Compile dot_product
dot_product: dot_product.c
	$(CC) $(CFLAGS) -o dot_product dot_product.c

# Clean up compiled files
clean:
	rm -f $(TARGETS) *.o

# Run exercises
run-ring:
	@echo "=== Running Ring Communication with 4 processes ==="
	mpirun -np 4 ./ring_comm

run-array:
	@echo "=== Running Parallel Array Sum with 4 processes ==="
	mpirun -np 4 ./array_sum

run-maxmin:
	@echo "=== Running Max/Min with 4 processes ==="
	mpirun -np 4 ./max_min

run-dot:
	@echo "=== Running Dot Product with 4 processes ==="
	mpirun -np 4 ./dot_product

# Run all exercises
run-all: run-ring run-array run-maxmin run-dot

.PHONY: all clean run-ring run-array run-maxmin run-dot run-all
