# LAB2 - Advanced Parallel Programming Makefile
# Compiles and runs all assignments with performance analysis

CXX = g++
CXXFLAGS = -fopenmp -O3 -std=c++17 -march=native
LDFLAGS = -lm

# Targets
Q1 = Question1/q1_md
Q2 = Question2/q2_sw
Q3 = Question3/q3_heat

# Python scripts
ANALYZE = Tools/analyze.py
PLOT = Tools/plot_results.py

# Default target
.PHONY: all
all: build

# Build all programs
.PHONY: build
build: $(Q1) $(Q2) $(Q3)
	@echo ""
	@echo "✅ All programs compiled successfully!"
	@echo ""
	@echo "Run individual problems:"
	@echo "  make run-q1    # Molecular Dynamics"
	@echo "  make run-q2    # Smith-Waterman"
	@echo "  make run-q3    # Heat Diffusion"
	@echo ""
	@echo "Run all and analyze:"
	@echo "  make run       # Run all problems"
	@echo "  make analyze   # Analyze results"
	@echo ""

# Q1: Molecular Dynamics
$(Q1): Question1/q1.cpp
	@echo "Compiling Q1: Molecular Dynamics..."
	$(CXX) $(CXXFLAGS) -o $(Q1) Question1/q1.cpp

# Q2: Smith-Waterman
$(Q2): Question2/q2.cpp
	@echo "Compiling Q2: Smith-Waterman..."
	$(CXX) $(CXXFLAGS) -o $(Q2) Question2/q2.cpp

# Q3: Heat Diffusion
$(Q3): Question3/q3.cpp
	@echo "Compiling Q3: Heat Diffusion..."
	$(CXX) $(CXXFLAGS) -o $(Q3) Question3/q3.cpp

# Run Q1
.PHONY: run-q1
run-q1: $(Q1)
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "Running: Q1 - Molecular Dynamics (Lennard-Jones)"
	@echo "════════════════════════════════════════════════════════════"
	@./$(Q1)

# Run Q2
.PHONY: run-q2
run-q2: $(Q2)
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "Running: Q2 - Smith-Waterman DNA Alignment"
	@echo "════════════════════════════════════════════════════════════"
	@./$(Q2)

# Run Q3
.PHONY: run-q3
run-q3: $(Q3)
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "Running: Q3 - Heat Diffusion Simulation"
	@echo "════════════════════════════════════════════════════════════"
	@./$(Q3)

# Run all programs
.PHONY: run
run: run-q1 run-q2 run-q3
	@echo ""
	@echo "✅ All programs completed!"
	@echo ""

# Analyze results
.PHONY: analyze
analyze:
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "Analyzing Results..."
	@echo "════════════════════════════════════════════════════════════"
	@python3 $(ANALYZE)

# Generate plots
.PHONY: plot
plot:
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "Generating Performance Plots..."
	@echo "════════════════════════════════════════════════════════════"
	@python3 $(PLOT)

# Full workflow: build, run, analyze, plot
.PHONY: full
full: build run analyze plot
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "🎉 LAB2 - FULL ANALYSIS COMPLETE! 🎉"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Results:"
	@echo "  Question1/md_results.txt"
	@echo "  Question2/smithwaterman_results.txt"
	@echo "  Question3/heatsim_results.txt"
	@echo ""
	@echo "Performance Plots:"
	@echo "  Results/lab2_performance_analysis.png"
	@echo "  Results/comparison.png"
	@echo ""

# Profile with perf (requires Linux and perf-tools)
.PHONY: profile-q1
profile-q1: $(Q1)
	@echo "Profiling Q1 with perf..."
	perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses ./$(Q1)

.PHONY: profile-q3
profile-q3: $(Q3)
	@echo "Profiling Q3 with perf..."
	perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses ./$(Q3)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(Q1) $(Q2) $(Q3)
	@rm -f Question1/md_results.txt
	@rm -f Question2/smithwaterman_results.txt
	@rm -f Question3/heatsim_results.txt
	@echo "✅ Clean complete!"

# Documentation
.PHONY: help
help:
	@echo ""
	@echo "LAB2 - Advanced Parallel Programming"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make build        - Compile all programs"
	@echo "  make run          - Run all programs"
	@echo "  make run-q1       - Run Q1 (Molecular Dynamics)"
	@echo "  make run-q2       - Run Q2 (Smith-Waterman)"
	@echo "  make run-q3       - Run Q3 (Heat Diffusion)"
	@echo "  make analyze      - Analyze performance results"
	@echo "  make plot         - Generate performance plots"
	@echo "  make full         - Build, run, analyze, and plot"
	@echo "  make profile-q1   - Profile Q1 with perf (Linux only)"
	@echo "  make profile-q3   - Profile Q3 with perf (Linux only)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
	@echo ""

.DEFAULT_GOAL := help
