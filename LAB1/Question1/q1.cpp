#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
using namespace std;
int main() {
    int N = 1 << 24;  // 16 million elements for noticeable speedup
    double a = 3.0;
    vector<double> X(N, 1.0), Y(N, 2.0);

    ofstream outfile("daxpy_speedup.txt");

    if (!outfile.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    } 

    outfile << "#Threads\tTime(s)\tSpeedup\n";
    double serial_time = 0.0;

    for (int threads = 1; threads <= 16; ++threads) {
        vector<double> x_copy = X; 
        double start = omp_get_wtime();
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; i++) {
            x_copy[i] = a * x_copy[i] + Y[i];
        }
        double end = omp_get_wtime();
        double elapsed = (end - start) * 1000.0; 
        if (threads == 1) serial_time = elapsed;
        double speedup = serial_time / elapsed;
        cout << "Threads: " << threads
             << " | Time: " << elapsed
             << " s | Speedup: " << speedup << endl;
        outfile << threads << "\t" << elapsed << "\t" << speedup << "\n";
    }

    outfile.close();
    cout << "Results saved to daxpy_speedup.txt\n";
    return 0;
}
