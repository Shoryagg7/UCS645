set terminal pngcairo size 800,600
set output "speedup.png"

set title "OpenMP Speedup vs Number of Threads"
set xlabel "Number of Threads"
set ylabel "Speedup"

set grid
set key left top

plot "speedup.dat" using 1:2 with linespoints lw 2 pt 7 title "Measured Speedup", \
     x with lines lw 2 dt 2 title "Ideal Speedup"
