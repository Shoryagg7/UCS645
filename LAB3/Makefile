# =====================================================================
# Makefile for LAB3: Correlation Coefficient Optimization
# =====================================================================

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp -march=native
LDFLAGS = -fopenmp -lm

# Optimization levels
CXXFLAGS_FAST = $(CXXFLAGS) -Ofast
CXXFLAGS_PROFILE = $(CXXFLAGS) -g -pg

# Target executable
TARGET = correlate

# Source files
SOURCES = main.cpp correlate.cpp
HEADERS = correlate.h
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Fast build (aggressive optimizations)
fast: CXXFLAGS = $(CXXFLAGS_FAST)
fast: clean $(TARGET)
	@echo "Fast build complete"

# Debug build
debug: CXXFLAGS = -std=c++17 -g -O0 -fopenmp
debug: clean $(TARGET)
	@echo "Debug build complete"

# Profile build
profile: CXXFLAGS = $(CXXFLAGS_PROFILE)
profile: clean $(TARGET)
	@echo "Profile build complete"

# Run tests
test: $(TARGET)
	@echo "=== Running Tests ==="
	./$(TARGET) -ny 500 -nx 500 -verify
	@echo ""

test-large: $(TARGET)
	@echo "=== Running Large Test ==="
	./$(TARGET) -ny 2000 -nx 5000 -verify

# Run all implementations comparison
compare: $(TARGET)
	@echo "=== Comparing All Implementations ==="
	./$(TARGET) -ny 1000 -nx 1000 -all

# Run with perf
perf-seq: $(TARGET)
	@echo "=== Sequential Performance ==="
	perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
		./$(TARGET) -ny 2000 -nx 5000 -impl 1

perf-par: $(TARGET)
	@echo "=== Parallel Performance (8 threads) ==="
	OMP_NUM_THREADS=8 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
		./$(TARGET) -ny 2000 -nx 5000 -impl 2

perf-opt: $(TARGET)
	@echo "=== Optimized Performance (8 threads) ==="
	OMP_NUM_THREADS=8 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
		./$(TARGET) -ny 2000 -nx 5000 -impl 3

perf-all: $(TARGET)
	@echo "=== All Implementations Performance ==="
	OMP_NUM_THREADS=8 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
		./$(TARGET) -ny 2000 -nx 5000 -all

# Scaling test (vary thread count)
scaling: $(TARGET)
	@echo "=== Scaling Analysis (ny=2000, nx=5000) ==="
	@for threads in 1 2 4 8 16; do \
		echo "Threads: $$threads"; \
		OMP_NUM_THREADS=$$threads ./$(TARGET) -ny 2000 -nx 5000 -impl 2 | grep -E "Time|GFLOPS"; \
	done

# Size scaling test (vary matrix size)
size-scaling: $(TARGET)
	@echo "=== Matrix Size Scaling (8 threads) ==="
	@for size in 500 1000 2000 4000; do \
		echo "Size: $$size x $$size"; \
		OMP_NUM_THREADS=8 ./$(TARGET) -ny $$size -nx $$size -impl 2 | grep -E "Time|GFLOPS"; \
	done

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) *.o gmon.out callgrind.out

# Clean everything including executables
distclean: clean
	rm -f $(TARGET)

# Help target
help:
	@echo "Makefile targets:"
	@echo "  all              - Build main executable"
	@echo "  fast             - Build with aggressive optimizations"
	@echo "  debug            - Build with debug symbols"
	@echo "  profile          - Build with profiling support"
	@echo "  test             - Run basic test"
	@echo "  test-large       - Run larger test"
	@echo "  compare          - Compare all implementations"
	@echo "  perf-seq         - Profile sequential version"
	@echo "  perf-par         - Profile parallel version (8T)"
	@echo "  perf-opt         - Profile optimized version (8T)"
	@echo "  perf-all         - Profile all with perf stat"
	@echo "  scaling          - Test scaling with thread count"
	@echo "  size-scaling     - Test scaling with matrix size"
	@echo "  clean            - Remove build artifacts"
	@echo "  distclean        - Remove all generated files"
	@echo "  help             - Display this help message"

.PHONY: all fast debug profile test test-large compare perf-seq perf-par perf-opt perf-all scaling size-scaling clean distclean help
