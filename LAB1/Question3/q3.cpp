#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;

int main() {
    long steps = 10000000;
    double step = 1.0 / steps;

    ofstream outfile("speedup.dat");

    double time1 = 0.0;

    for (int threads = 1; threads <= 16; ++threads) {
        double sum = 0.0;

        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(threads) reduction(+:sum)
        for (long i = 0; i < steps; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        double pi = step * sum;
        double end = omp_get_wtime();
        double elapsed = end - start;

        if (threads == 1)
            time1 = elapsed;

        double speedup = time1 / elapsed;

        cout << "Threads: " << threads
             << "  Pi: " << pi
             << "  Time: " << elapsed
             << "  Speedup: " << speedup << endl;

        outfile << threads << " " << speedup << endl;
    }

    outfile.close();
    return 0;
}
