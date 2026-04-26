#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Use a clean, professional aesthetic for academic/HPC plotting
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')  # Fallback for older matplotlib versions

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_execution_vs_q(df, out_dir):
    d = df[df['experiment'] == 'vary_q']
    if d.empty:
        return
    plt.figure(figsize=(9, 6))
    
    # Define a consistent color scheme
    colors = {'mos': '#1f77b4', 'omp_bruteforce': '#ff7f0e', 'cuda_bruteforce': '#2ca02c'}
    
    for method, g in d.groupby('method'):
        g2 = g.sort_values('q')
        plt.plot(g2['q'], g2['query_sec'], marker='o', linewidth=2, markersize=8, 
                 label=method, color=colors.get(method))
    
    # CRITICAL: Logarithmic scale for Y-axis to show the massive performance gap
    plt.yscale('log')
    
    plt.xlabel('Number of Queries (Q)', fontsize=12, fontweight='bold')
    plt.ylabel('Query Time (Seconds) [Log Scale]', fontsize=12, fontweight='bold')
    plt.title('Execution Time vs Number of Queries (Lower is Better)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    save_plot(os.path.join(out_dir, 'execution_time_vs_queries_log.png'))

def plot_relative_speedup_bar(df, out_dir):
    """New: Generates the Speedup Multiplier bar chart for the README."""
    d = df[df['experiment'] == 'vary_q']
    if d.empty:
        return
    
    # Grab the largest workload available
    max_q = d['q'].max()
    d_max = d[d['q'] == max_q]
    
    # Get Mo's algorithm baseline time
    mos_data = d_max[d_max['method'] == 'mos']
    if mos_data.empty:
        return
    mos_time = mos_data['query_sec'].values[0]
    
    methods = []
    speedups = []
    colors_list = []
    colors_map = {'mos': '#1f77b4', 'omp_bruteforce': '#ff7f0e', 'cuda_bruteforce': '#2ca02c'}
    
    for method in ['mos', 'omp_bruteforce', 'cuda_bruteforce']:
        m_data = d_max[d_max['method'] == method]
        if not m_data.empty:
            m_time = m_data['query_sec'].values[0]
            speedup = mos_time / m_time
            methods.append(method)
            speedups.append(speedup)
            colors_list.append(colors_map.get(method, '#333333'))
            
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, speedups, color=colors_list, edgecolor='black', linewidth=1.2)
    
    # Add text labels on top of the bars (e.g., "40.5x")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), 
                 f'{yval:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup Multiplier (Relative to mos)', fontsize=12, fontweight='bold')
    plt.title(f'Algorithm Speedup Comparison at Q={max_q}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_plot(os.path.join(out_dir, 'relative_speedup_bar.png'))

def plot_speedup_vs_threads(df, out_dir):
    d = df[(df['experiment'] == 'strong_scaling') & (df['method'] == 'omp_bruteforce')]
    if d.empty:
        return
    d = d.sort_values('threads')
    plt.figure(figsize=(8, 5))
    plt.plot(d['threads'], d['speedup'], marker='s', linewidth=2, color='#ff7f0e')
    plt.xlabel('Threads', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    plt.title('OpenMP Strong Scaling (Speedup vs Threads)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.5, linestyle='--')
    save_plot(os.path.join(out_dir, 'speedup_vs_threads.png'))

def plot_efficiency_vs_threads(df, out_dir):
    d = df[(df['experiment'] == 'strong_scaling') & (df['method'] == 'omp_bruteforce')]
    if d.empty:
        return
    d = d.sort_values('threads')
    plt.figure(figsize=(8, 5))
    plt.plot(d['threads'], d['efficiency'], marker='D', linewidth=2, color='#9467bd')
    plt.xlabel('Threads', fontsize=12, fontweight='bold')
    plt.ylabel('Parallel Efficiency', fontsize=12, fontweight='bold')
    plt.title('OpenMP Efficiency vs Threads (Hardware Bottlenecks)', fontsize=14, fontweight='bold')
    plt.axhline(y=1.0, color='r', linestyle=':', label='Perfect Efficiency (1.0)')
    plt.legend()
    plt.grid(True, alpha=0.5, linestyle='--')
    save_plot(os.path.join(out_dir, 'efficiency_vs_threads.png'))

def plot_query_size_performance(df, out_dir):
    d = df[df['experiment'] == 'vary_range']
    if d.empty:
        return
    order = {'small': 0, 'medium': 1, 'large': 2}
    d = d.assign(range_order=d['range_class'].map(order)).sort_values('range_order')

    plt.figure(figsize=(8, 5))
    for method, g in d.groupby('method'):
        g2 = g.sort_values('range_order')
        plt.plot(g2['range_class'], g2['query_sec'], marker='o', linewidth=2, label=method)
    
    plt.yscale('log')
    plt.xlabel('Range Class', fontsize=12, fontweight='bold')
    plt.ylabel('Query Time (Seconds) [Log Scale]', fontsize=12, fontweight='bold')
    plt.title('Query Size vs Performance', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, which='both', alpha=0.5, linestyle='--')
    save_plot(os.path.join(out_dir, 'query_size_vs_performance_log.png'))

def plot_throughput(df, out_dir):
    d = df[df['experiment'] == 'vary_q']
    if d.empty:
        return
    plt.figure(figsize=(9, 6))
    for method, g in d.groupby('method'):
        g2 = g.sort_values('q')
        plt.plot(g2['q'], g2['throughput_qps'], marker='^', linewidth=2, label=method)
    
    plt.yscale('log')
    plt.xlabel('Number of Queries (Q)', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (Queries / Second) [Log]', fontsize=12, fontweight='bold')
    plt.title('System Throughput Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, which='both', alpha=0.5, linestyle='--')
    save_plot(os.path.join(out_dir, 'throughput_comparison_log.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/data.csv')
    parser.add_argument('--out-dir', default='plots')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.input)

    plot_execution_vs_q(df, args.out_dir)
    plot_relative_speedup_bar(df, args.out_dir)
    plot_speedup_vs_threads(df, args.out_dir)
    plot_efficiency_vs_threads(df, args.out_dir)
    plot_query_size_performance(df, args.out_dir)
    plot_throughput(df, args.out_dir)

    print(f'Generated high-fidelity academic plots in {args.out_dir}')

if __name__ == '__main__':
    main()