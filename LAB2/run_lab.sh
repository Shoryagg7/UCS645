#!/bin/bash
# LAB2 - Quick Setup and Execution Script
# Compiles and runs all assignments automatically

echo ""
echo "🚀 LAB2 - Advanced Parallel Programming"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check compiler
if ! command -v g++ &> /dev/null; then
    echo "❌ Error: g++ not found. Please install GCC with OpenMP support."
    exit 1
fi

echo "${BLUE}[1/4] Compiling programs...${NC}"
echo ""

# Compile Q1
echo "  Compiling Q1: Molecular Dynamics..."
g++ -fopenmp -O3 -std=c++17 -march=native Question1/q1.cpp -o Question1/q1_md || exit 1
echo "    ✅ Q1 compiled"

# Compile Q2
echo "  Compiling Q2: Smith-Waterman..."
g++ -fopenmp -O3 -std=c++17 -march=native Question2/q2.cpp -o Question2/q2_sw || exit 1
echo "    ✅ Q2 compiled"

# Compile Q3
echo "  Compiling Q3: Heat Diffusion..."
g++ -fopenmp -O3 -std=c++17 -march=native Question3/q3.cpp -o Question3/q3_heat || exit 1
echo "    ✅ Q3 compiled"

echo ""
echo "${BLUE}[2/4] Running benchmarks...${NC}"
echo ""

# Run Q1
echo "  Running Q1: Molecular Dynamics..."
./Question1/q1_md > /dev/null
echo "    ✅ Q1 complete"

# Run Q2
echo "  Running Q2: Smith-Waterman..."
./Question2/q2_sw > /dev/null
echo "    ✅ Q2 complete"

# Run Q3
echo "  Running Q3: Heat Diffusion..."
./Question3/q3_heat > /dev/null
echo "    ✅ Q3 complete"

echo ""
if command -v python3 &> /dev/null; then
    echo "${BLUE}[3/4] Analyzing results...${NC}"
    echo ""
    python3 Tools/analyze.py

    echo ""
    echo "${BLUE}[4/4] Generating plots...${NC}"
    echo ""
    python3 Tools/plot_results.py

    echo ""
    echo "${GREEN}✅ LAB2 ANALYSIS COMPLETE!${NC}"
    echo ""
    echo "Results generated in:"
    echo "  - Question1/md_results.txt"
    echo "  - Question2/smithwaterman_results.txt"
    echo "  - Question3/heatsim_results.txt"
    echo "  - Results/lab2_performance_analysis.png"
    echo "  - Results/comparison.png"
else
    echo "${BLUE}[3/4] Skipping Python analysis (python3 not found)${NC}"
    echo ""
    echo "Results are in:"
    echo "  - Question1/md_results.txt"
    echo "  - Question2/smithwaterman_results.txt"
    echo "  - Question3/heatsim_results.txt"
fi

echo ""
echo "📖 For detailed information, see:"
echo "  - README.md (Lab overview)"
echo "  - SETUP.md (Installation guide)"
echo "  - Question1/, Question2/, Question3/ (Individual READMEs)"
echo ""
