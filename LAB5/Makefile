# Makefile for MPI Exercises LAB5

# Compiler
CC = mpicc
CFLAGS = -Wall -std=c99 -lm

# Executables
TARGETS = daxpy broadcast_race dot_product_amdahl prime_finder perfect_numbers

# Default target
all: $(TARGETS)

# Compile daxpy
daxpy: daxpy.c
	$(CC) $(CFLAGS) -o daxpy daxpy.c

# Compile broadcast_race
broadcast_race: broadcast_race.c
	$(CC) $(CFLAGS) -o broadcast_race broadcast_race.c

# Compile dot_product_amdahl
dot_product_amdahl: dot_product_amdahl.c
	$(CC) $(CFLAGS) -o dot_product_amdahl dot_product_amdahl.c

# Compile prime_finder
prime_finder: prime_finder.c
	$(CC) $(CFLAGS) -o prime_finder prime_finder.c

# Compile perfect_numbers
perfect_numbers: perfect_numbers.c
	$(CC) $(CFLAGS) -o perfect_numbers perfect_numbers.c

# Clean up compiled files
clean:
	rm -f $(TARGETS) *.o

# Run exercises
run-daxpy:
	@echo "=== Running DAXPY with 4 processes ==="
	mpirun -np 4 ./daxpy

run-broadcast:
	@echo "=== Running Broadcast Race with 4 processes ==="
	mpirun -np 4 ./broadcast_race

run-dot-product:
	@echo "=== Running Dot Product (Amdahl's Law) with 4 processes ==="
	mpirun -np 4 ./dot_product_amdahl

run-prime:
	@echo "=== Running Prime Finder with 4 processes ==="
	mpirun -np 4 ./prime_finder

run-perfect:
	@echo "=== Running Perfect Numbers with 4 processes ==="
	mpirun -np 4 ./perfect_numbers

# Test with multiple process counts for performance analysis
test-daxpy:
	@for np in 1 2 4 8; do \
		echo "\n=== DAXPY with $$np processes ==="; \
		mpirun -np $$np ./daxpy; \
	done

test-broadcast:
	@for np in 2 4 8 16; do \
		echo "\n=== Broadcast with $$np processes ==="; \
		mpirun -np $$np ./broadcast_race; \
	done

test-dot-product:
	@for np in 1 2 4 8; do \
		echo "\n=== Dot Product with $$np processes ==="; \
		mpirun -np $$np ./dot_product_amdahl; \
	done

# Run all exercises
run-all: run-daxpy run-broadcast run-dot-product run-prime run-perfect

.PHONY: all clean run-daxpy run-broadcast run-dot-product run-prime run-perfect test-daxpy test-broadcast test-dot-product run-all
