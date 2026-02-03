set terminal pngcairo size 900,600
set output "speedup.png"

set title "Speedup vs Threads (Matrix Multiplication)"
set xlabel "Number of Threads"
set ylabel "Speedup"

set grid
set key top left
set xtics 1

plot \
    "timings.dat" using 1:4 with linespoints lw 2 pt 7 title "1D OpenMP", \
    "timings.dat" using 1:5 with linespoints lw 2 pt 5 title "2D OpenMP"
