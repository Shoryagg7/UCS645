set terminal png size 800,600
set output "daxpy_results.png"
set grid

# Plot Time vs Threads
set title "DAXPY Execution Time vs Threads"
set xlabel "Number of Threads"
set ylabel "Execution Time (s)"
plot "daxpy_speedup.txt" using 1:2 with linespoints lw 2 lc rgb "blue" title "Time"

# To plot Speedup, you can run a second plot:
set output "daxpy_speedup.png"
set title "DAXPY Speedup vs Threads"
set ylabel "Speedup"
plot "daxpy_speedup.txt" using 1:3 with linespoints lw 2 lc rgb "red" title "Speedup"
