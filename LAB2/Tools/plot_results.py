#!/usr/bin/env python3
"""
Plot generation script for LAB2 performance results
Generates speedup and efficiency plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results():
    """Generate speedup and efficiency plots"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LAB2: Parallel Programming Performance Analysis', fontsize=16, fontweight='bold')

    # Define problems
    problems = [
        {
            'file': 'Question1/md_results.txt',
            'label': 'Q1: Molecular Dynamics',
            'color': '#FF6B6B',
            'ax_speedup': axes[0, 0],
            'ax_efficiency': axes[1, 0],
        },
        {
            'file': 'Question2/smithwaterman_results.txt',
            'label': 'Q2: Smith-Waterman',
            'color': '#4ECDC4',
            'ax_speedup': axes[0, 1],
            'ax_efficiency': axes[1, 1],
        },
    ]

    all_threads = None
    all_speedups = []
    all_labels = []

    for problem in problems:
        filepath = Path(problem['file'])
        if not filepath.exists():
            print(f"Warning: {filepath} not found.")
            continue

        try:
            data = np.genfromtxt(filepath, delimiter='\t', skip_header=1, unpack=True)
            threads = data[0]
            times = data[1]
            speedups = data[2]

            if all_threads is None:
                all_threads = threads

            efficiency = speedups / threads * 100

            # Speedup plot
            problem['ax_speedup'].plot(threads, speedups, 'o-', color=problem['color'],
                                       linewidth=2, markersize=8, label='Measured')
            problem['ax_speedup'].plot(threads, threads, '--', color='gray', alpha=0.5,
                                      label='Ideal Linear')
            problem['ax_speedup'].set_xlabel('Number of Threads', fontsize=11)
            problem['ax_speedup'].set_ylabel('Speedup', fontsize=11)
            problem['ax_speedup'].set_title(problem['label'], fontsize=12, fontweight='bold')
            problem['ax_speedup'].grid(True, alpha=0.3)
            problem['ax_speedup'].legend()
            problem['ax_speedup'].set_xticks([1, 2, 4, 8, 12, 16])

            # Efficiency plot
            problem['ax_efficiency'].plot(threads, efficiency, 's-', color=problem['color'],
                                         linewidth=2, markersize=8)
            problem['ax_efficiency'].axhline(y=100, color='gray', linestyle='--', alpha=0.5,
                                            label='Ideal (100%)')
            problem['ax_efficiency'].axhline(y=75, color='orange', linestyle=':', alpha=0.5,
                                            label='Good (75%)')
            problem['ax_efficiency'].set_xlabel('Number of Threads', fontsize=11)
            problem['ax_efficiency'].set_ylabel('Efficiency (%)', fontsize=11)
            problem['ax_efficiency'].set_title(f'{problem["label"]} - Efficiency', fontsize=12, fontweight='bold')
            problem['ax_efficiency'].grid(True, alpha=0.3)
            problem['ax_efficiency'].set_ylim([0, 120])
            problem['ax_efficiency'].set_xticks([1, 2, 4, 8, 12, 16])
            problem['ax_efficiency'].legend()

            all_speedups.append(speedups)
            all_labels.append(problem['label'])

            print(f"Plotted: {problem['label']}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    plt.tight_layout()
    save_path = Path('Results/lab2_performance_analysis.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSpeedup & Efficiency plots saved: {save_path}")
    plt.close()

    # Comparison plot
    if len(all_speedups) > 1 and all_threads is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Comparative Performance Analysis', fontsize=14, fontweight='bold')

        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

        # Speedup comparison
        for i, (speedup, label) in enumerate(zip(all_speedups, all_labels)):
            ax1.plot(all_threads, speedup, 'o-', linewidth=2, markersize=8,
                    label=label, color=colors[i % len(colors)])

        ax1.plot(all_threads, all_threads, '--', color='gray', alpha=0.5, label='Ideal Linear', linewidth=2)
        ax1.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Speedup', fontsize=11, fontweight='bold')
        ax1.set_title('Speedup Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xticks([1, 2, 4, 8, 12, 16])

        # Efficiency comparison
        for i, (speedup, label) in enumerate(zip(all_speedups, all_labels)):
            efficiency = speedup / all_threads * 100
            ax2.plot(all_threads, efficiency, 's-', linewidth=2, markersize=8,
                    label=label, color=colors[i % len(colors)])

        ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')
        ax2.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Efficiency Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([1, 2, 4, 8, 12, 16])
        ax2.set_ylim([0, 120])
        ax2.legend(fontsize=10)

        plt.tight_layout()
        save_path = Path('Results/comparison.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
        plt.close()

    print("\n✅ All plots generated successfully!")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)  # Change to lab directory
    plot_results()
