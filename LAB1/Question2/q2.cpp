#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
using namespace std;

using Matrix = vector<vector<double>>;

void multiply_parallel_1D(int n, const Matrix& A, const Matrix& B, Matrix& C, int threads) {
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

void multiply_parallel_2D(int n, const Matrix& A, const Matrix& B, Matrix& C, int threads) {
    #pragma omp parallel for collapse(2) num_threads(threads) schedule(static, 64)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

int main() {
    int n = 1000;
    Matrix A(n, vector<double>(n, 1.0));
    Matrix B(n, vector<double>(n, 2.0));
    Matrix C(n, vector<double>(n, 0.0));

    ofstream fout("timings.dat");
    fout << "#threads time1D time2D speedup1D speedup2D\n";

    double base1D = 0.0, base2D = 0.0;

    for (int threads = 1; threads <= 16; threads++) {
        // 1D
        for (auto &row : C) fill(row.begin(), row.end(), 0.0);
        double start = omp_get_wtime();
        multiply_parallel_1D(n, A, B, C, threads);
        double t1 = omp_get_wtime() - start;

        // 2D
        for (auto &row : C) fill(row.begin(), row.end(), 0.0);
        start = omp_get_wtime();
        multiply_parallel_2D(n, A, B, C, threads);
        double t2 = omp_get_wtime() - start;

        if (threads == 1) {
            base1D = t1;
            base2D = t2;
        }

        double s1 = base1D / t1;
        double s2 = base2D / t2;

        cout << "Threads " << threads
             << " | Time1D: " << t1
             << " | Time2D: " << t2
             << " | Speedup1D: " << s1
             << " | Speedup2D: " << s2 << "\n";

        fout << threads << " " << t1 << " " << t2 << " "
             << s1 << " " << s2 << "\n";
    }

    fout.close();
    return 0;
}
