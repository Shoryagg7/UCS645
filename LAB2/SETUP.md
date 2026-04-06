# Setup & Installation Guide

## System Requirements

- **OS**: Linux, macOS, or Windows (with GCC/MinGW)
- **Compiler**: GCC 7+ or Clang with OpenMP support
- **RAM**: 8 GB minimum (for heat simulation benchmarks)
- **Cores**: 4+ cores recommended for meaningful parallelization tests

## Installation

### Ubuntu/Debian

```bash
# Install development tools
sudo apt-get update
sudo apt-get install build-essential libomp-dev python3 python3-matplotlib

# Verify installation
gcc --version
python3 --version
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install gcc libomp python3 matplotlib

# Verify
gcc --version
python3 --version
```

### Windows (MSYS2/MinGW)

```bash
# Install MSYS2 from https://www.msys2.org/

# In MSYS2 terminal:
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-openmp python3 python3-matplotlib

# Verify
gcc --version
python3 --version
```

### Windows (Visual Studio)

Visual Studio 2019+ includes OpenMP support:
1. Install Visual Studio with C++ workload
2. Use MSVC compiler with `/openmp` flag
3. Or use Intel OneAPI Compiler (better OpenMP support)

## Quick Start

### 1. Navigate to LAB2

```bash
cd LAB2
```

### 2. Build All Programs

```bash
make build
```

Or compile individually:

```bash
# Q1: Molecular Dynamics
g++ -fopenmp -O3 -std=c++17 -march=native Question1/q1.cpp -o Question1/q1_md

# Q2: Smith-Waterman
g++ -fopenmp -O3 -std=c++17 -march=native Question2/q2.cpp -o Question2/q2_sw

# Q3: Heat Diffusion
g++ -fopenmp -O3 -std=c++17 -march=native Question3/q3.cpp -o Question3/q3_heat
```

### 3. Run Programs

```bash
# Run all
make run

# Or individually
make run-q1  # Molecular Dynamics
make run-q2  # Smith-Waterman
make run-q3  # Heat Diffusion
```

### 4. Analyze Results

```bash
# View text analysis
python3 Tools/analyze.py

# Generate plots
python3 Tools/plot_results.py
```

### 5. Complete Workflow

```bash
make full
```

This will:
1. Build all programs
2. Run all benchmarks
3. Analyze results
4. Generate performance plots

## Compiler Options

### Performance Optimization

```bash
# Aggressive optimization
g++ -O3 -march=native -ffast-math -o prog prog.cpp

# Enable SIMD (choose based on CPU)
g++ -O3 -mavx2 q3.cpp -o q3      # Intel/AMD with AVX2
g++ -O3 -mavx512f q3.cpp -o q3   # Modern Intel with AVX-512
g++ -O3 -msse4.2 q3.cpp -o q3    # Older systems

# Link OpenMP explicitly
g++ -O3 -fopenmp -lgomp prog.cpp -o prog
```

### Debugging Options

```bash
# Enable debug info
g++ -g -O0 -fopenmp q1.cpp -o q1_debug

# Run with gdb
gdb ./q1_debug

# LLVM AddressSanitizer (detect memory errors)
g++ -O1 -g -fsanitize=address -fopenmp q1.cpp -o q1_asan
./q1_asan

# ThreadSanitizer (detect data races)
g++ -O1 -g -fsanitize=thread -fopenmp q1.cpp -o q1_tsan
./q1_tsan
```

## Environment Variables

### Control OpenMP Behavior

```bash
# Set number of threads
export OMP_NUM_THREADS=8

# Set scheduling strategy
export OMP_SCHEDULE="dynamic,64"

# Pin threads to cores (Linux)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Verbose output
export OMP_DISPLAY_ENV=true

# Example run with settings
OMP_NUM_THREADS=4 OMP_PROC_BIND=close ./Question3/q3_heat
```

## Performance Profiling

### Using perf (Linux)

```bash
# Install perf
sudo apt-get install linux-tools-common

# Basic statistics
perf stat ./Question1/q1_md

# Count cycles and instructions
perf stat -e cycles,instructions,task-clock ./Question1/q1_md

# Count cache misses
perf stat -e L1-dcache-load-misses,LLC-load-misses ./Question1/q1_md

# Detailed report
perf record './Question1/q1_md'
perf report
```

### Using LIKWID (recommended for FLOPS analysis)

```bash
# Install LIKWID
git clone https://github.com/RRZE-HPC/likwid.git
cd likwid
make
sudo make install

# Measure FLOPS
likwid-perfctr -C 0-7 -g FLOPS_DP ./Question1/q1_md

# Measure memory bandwidth
likwid-perfctr -C 0-7 -g MEM ./Question1/q1_md

# Power analysis (requires PMU)
likwid-perfctr -C 0-7 -g ENERGY ./Question3/q3_heat
```

### Using Intel VTune

```bash
# Download from https://www.intel.com/content/www/en/us/docs/vtune/

# Profile application
vtune -collect hotspots -r results_dir ./Question1/q1_md

# View results
vtune -report summary -r results_dir
```

## Troubleshooting

### Compilation Issues

**Error: "omp.h: No such file or directory"**
```bash
# Install OpenMP development files
sudo apt-get install libomp-dev  # Ubuntu/Debian
brew install libomp              # macOS
```

**Error: "undefined reference to `omp_*`"**
```bash
# Add -fopenmp flag
g++ -fopenmp prog.cpp -o prog  # Add this flag!
```

**Error: Wrong compilation for Clang**
```bash
# Clang uses different syntax
clang++ -fopenmp prog.cpp -o prog  # Works!
```

### Runtime Issues

**All results identical across threads (parallelization not working)?**
```bash
export OMP_DISPLAY_ENV=true
./q1_md  # Check if OMP is actually running
```

**Program too slow or times out?**
```bash
# Reduce problem size (edit source code)
# For Q2, reduce seq_length from 2000 to 500
# For Q3, reduce grid_size from 500 to 250
```

**Memory issue / Out of memory?**
```bash
# Check available memory
free -h

# Reduce problem size:
# Q1: int num_particles = 1000; → 500
# Q2: int seq_length = 2000; → 1000
# Q3: int grid_size = 500; → 250
```

### Performance Issues

**Expected speedup not achieved?**
1. Check CPU throttling: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
2. Close background applications
3. Set CPU governor to performance:
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```
4. Use taskset to pin threads:
   ```bash
   taskset -c 0-3 ./program
   ```

**High variance in results?**
- Close background processes
- Run multiple times and average
- Disable CPU frequency scaling
- Use fixed frequency for consistent measurements

## Organization

After running `make full`, you'll have:

```
LAB2/
├── Question1/
│   ├── q1_md                     # Compiled executable
│   └── md_results.txt            # Performance data
├── Question2/
│   ├── q2_sw                     # Compiled executable
│   └── smithwaterman_results.txt # Performance data
├── Question3/
│   ├── q3_heat                   # Compiled executable
│   └── heatsim_results.txt       # Performance data
└── Results/
    ├── lab2_performance_analysis.png
    └── comparison.png
```

## Next Steps

1. **Read Documentation**: Review each Question's README.md
2. **Run Benchmarks**: Execute `make full` for complete analysis
3. **Analyze Results**: Study the generated plots and analysis
4. **Optimize Code**: Implement suggested optimizations
5. **Profile**: Use perf/LIKWID for deeper analysis

## Performance Targets

| Problem | Expected Speedup @ 16T | Efficiency |
|---------|------------------------|-----------|
| Q1 (MD) | 11-13x | 85-90% |
| Q2 (SW) | 3.5-4.0x | 22-25% |
| Q3 (Heat) | 13-14x | 85-90% |

If results are significantly lower, check:
- CPU throttling
- Background processes
- That -O3 optimization is enabled
- OpenMP is actually running

## Support

For issues or questions:
1. Check individual Question README files
2. Review OpenMP documentation: https://www.openmp.org/
3. Consult compiler documentation for your system

---

**Good luck with LAB2!** 🚀
