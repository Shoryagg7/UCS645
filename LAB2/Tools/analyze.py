#!/usr/bin/env python3
"""
Analysis script for LAB2 performance results
Generates CSV analysis from output files
"""

import os
import numpy as np
from pathlib import Path

def analyze_results(filename, output_name):
    """Parse and analyze results file"""
    try:
        data = np.genfromtxt(filename, delimiter='\t', skip_header=1, unpack=True)

        if len(data.shape) == 1:
            print(f"  Warning: {filename} has single row")
            return

        threads = data[0]
        times = data[1]
        speedups = data[2]

        baseline = times[0]
        efficiency = speedups / threads * 100

        print(f"\n{'='*70}")
        print(f"Analysis: {output_name}")
        print(f"{'='*70}")
        print(f"\n{'Threads':<10} {'Time(s)':<12} {'Speedup':<10} {'Efficiency':<12}")
        print("-" * 70)

        for i in range(len(threads)):
            print(f"{int(threads[i]):<10} {times[i]:<12.6f} {speedups[i]:<10.2f}x {efficiency[i]:<12.1f}%")

        # Calculate metrics
        max_speedup = np.max(speedups)
        scaling_efficiency = speedups[-1] / threads[-1] * 100

        print(f"\n{'Metric':<30} {'Value':<15}")
        print("-" * 50)
        print(f"{'Maximum Speedup':<30} {max_speedup:.2f}x")
        print(f"{'Speedup at 16 threads':<30} {speedups[-1]:.2f}x")
        print(f"{'Scaling Efficiency (16T)':<30} {scaling_efficiency:.1f}%")
        print(f"{'Average Efficiency':<30} {np.mean(efficiency):.1f}%")
        print(f"{'Baseline Time (1 thread)':<30} {baseline:.6f}s")

        # Identify bottleneck
        if speedups[-1] / threads[-1] < 0.6:
            print(f"{'Scaling Analysis':<30} BOUNDED (Amdahl's Law)")
        elif speedups[-1] / threads[-1] < 0.9:
            print(f"{'Scaling Analysis':<30} MODERATE (Good scaling)")
        else:
            print(f"{'Scaling Analysis':<30} EXCELLENT (Near-linear)")

        return data, efficiency

    except Exception as e:
        print(f"Error analyzing {filename}: {e}")
        return None, None

def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("LAB2 - Advanced Parallel Programming: Results Analysis")
    print("="*70)

    questions = [
        ("Question1/md_results.txt", "Q1 - Molecular Dynamics (Lennard-Jones)"),
        ("Question2/smithwaterman_results.txt", "Q2 - Smith-Waterman DNA Alignment"),
        ("Question3/heatsim_results.txt", "Q3 - Heat Diffusion Simulation"),
    ]

    all_efficiencies = {}

    for filepath, label in questions:
        full_path = Path(filepath)
        if full_path.exists():
            data, eff = analyze_results(str(full_path), label)
            if eff is not None:
                all_efficiencies[label] = eff
        else:
            print(f"\nWarning: {filepath} not found. Run the programs first:")
            print(f"  cd {filepath.split('/')[0]} && g++ -fopenmp -O3 -std=c++17 q*.cpp -o q")

    # Comparative analysis
    if len(all_efficiencies) > 1:
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS")
        print("="*70)

        print("\nScaling Efficiency Comparison:")
        print("-" * 70)
        for label, eff in all_efficiencies.items():
            print(f"{label:<45} {eff[-1]:>6.1f}% @ 16 threads")

        best = max(all_efficiencies.items(), key=lambda x: x[1][-1])
        print(f"\nBest Overall Scaling: {best[0]}")
        print(f"  - Maintains {best[1][-1]:.1f}% efficiency at 16 threads")

    print("\n" + "="*70)
    print("Analysis complete! Check plot_results.py for visualization.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
