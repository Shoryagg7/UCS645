CXX := g++
NVCC := nvcc
CXXFLAGS := -O3 -std=c++17 -march=native
OMPFLAGS := -fopenmp

.PHONY: all cpu cuda clean run-quick run-full

all: cpu cuda

cpu: mos omp_bruteforce input_generator benchmark

mos: mos.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

omp_bruteforce: omp_bruteforce.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

input_generator: input_generator.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

benchmark: benchmark.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

cuda:
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) -O3 -std=c++17 cuda_kernel.cu -o cuda_kernel; \
		echo "Built cuda_kernel"; \
	else \
		echo "nvcc not found: skipping cuda_kernel"; \
	fi

run-quick: all
	./benchmark --quick
	python3 plots/plot.py --input results/data.csv --out-dir plots

run-full: all
	./benchmark
	python3 plots/plot.py --input results/data.csv --out-dir plots

clean:
	rm -f mos omp_bruteforce cuda_kernel input_generator benchmark
