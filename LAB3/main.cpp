#include "correlate.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <cstring>
#include <algorithm>

// =======================================================================
// PERFORMANCE MEASUREMENT & VERIFICATION UTILITIES
// =======================================================================

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

void generate_data(int ny, int nx, float* data) {
    for (int i = 0; i < ny * nx; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

bool verify_result(int ny, float* result1, float* result2, float tolerance = 1e-5f) {
    int errors = 0;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            float val1 = result1[i + j * ny];
            float val2 = result2[i + j * ny];
            float diff = std::fabs(val1 - val2);

            if (diff > tolerance && diff / (std::fabs(val1) + 1e-10f) > tolerance) {
                errors++;
                if (errors <= 5) {
                    std::cout << "Mismatch at [" << i << ", " << j << "]: "
                              << val1 << " vs " << val2 << std::endl;
                }
            }
        }
    }

    if (errors > 0) {
        std::cout << "Total errors: " << errors << std::endl;
        return false;
    }
    return true;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n\n"
              << "Options:\n"
              << "  -ny NUM          Number of vectors (rows)        [default: 1000]\n"
              << "  -nx NUM          Vector dimension (columns)      [default: 1000]\n"
              << "  -threads NUM     Number of OpenMP threads        [default: auto]\n"
              << "  -verify          Verify parallel correctness     [default: off]\n"
              << "  -all             Run all implementations          [default: off]\n"
              << "  -impl [1-4]      Run specific implementation     [default: all]\n"
              << "\nExample:\n"
              << "  " << program << " -ny 2000 -nx 5000 -threads 8 -all\n";
}

// =======================================================================
// MAIN PROGRAM
// =======================================================================
int main(int argc, char* argv[])
{
    int ny = 1000;
    int nx = 1000;
    int num_threads = 0;
    bool verify = false;
    bool run_all = false;
    int specific_impl = -1;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-ny") == 0 && i + 1 < argc) {
            ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-nx") == 0 && i + 1 < argc) {
            nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-verify") == 0) {
            verify = true;
        } else if (std::strcmp(argv[i], "-all") == 0) {
            run_all = true;
        } else if (std::strcmp(argv[i], "-impl") == 0 && i + 1 < argc) {
            specific_impl = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
    }

    std::cout << "========================================\n"
              << "  Correlation Coefficient Calculator\n"
              << "========================================\n\n"
              << "Configuration:\n"
              << "  Vectors (ny):        " << ny << "\n"
              << "  Dimension (nx):      " << nx << "\n"
              << "  Max threads:         " << omp_get_max_threads() << "\n"
              << "  Active threads:      " << num_threads << "\n"
              << "  Matrix size:         " << (ny * nx * sizeof(float) / (1024.0 * 1024.0)) << " MB\n"
              << "  Result size:         " << (ny * ny * sizeof(float) / (1024.0 * 1024.0)) << " MB\n\n";

    float* data = new float[ny * nx];
    float* result_seq = new float[ny * ny];
    float* result_par = new float[ny * ny];
    float* result_opt = new float[ny * ny];
    float* result_hopt = new float[ny * ny];

    generate_data(ny, nx, data);

    std::cout << "Generated random data.\n\n";
    std::cout << std::string(60, '=') << "\n"
              << "PERFORMANCE RESULTS\n"
              << std::string(60, '=') << "\n\n";

    Timer timer;
    double time_seq = 0.0;

    // ===== IMPLEMENTATION 1: SEQUENTIAL =====
    if (run_all || specific_impl == 1 || specific_impl == -1) {
        std::cout << "[1] Sequential Baseline:\n";
        timer.start();
        correlate_sequential(ny, nx, data, result_seq);
        time_seq = timer.elapsed();
        std::cout << "    Time: " << std::fixed << std::setprecision(4) << time_seq << " seconds\n"
                  << "    GFLOPS: " << (2.0 * ny * ny * nx / 1e9 / time_seq) << "\n\n";
    }

    // ===== IMPLEMENTATION 2: PARALLEL =====
    if (run_all || specific_impl == 2 || specific_impl == -1) {
        std::cout << "[2] Parallel (OpenMP):\n";
        omp_set_num_threads(num_threads);
        timer.start();
        correlate_parallel(ny, nx, data, result_par);
        double time_par = timer.elapsed();
        double speedup = (time_seq > 0) ? time_seq / time_par : 1.0;
        std::cout << "    Time: " << std::fixed << std::setprecision(4) << time_par << " seconds\n"
                  << "    GFLOPS: " << (2.0 * ny * ny * nx / 1e9 / time_par) << "\n";
        if (speedup > 1.0) {
            std::cout << "    Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        std::cout << "\n";

        if (verify && time_seq > 0) {
            std::cout << "    Verification vs Sequential: ";
            if (verify_result(ny, result_seq, result_par)) {
                std::cout << "✓ PASS\n\n";
            } else {
                std::cout << "✗ FAIL\n\n";
            }
        }
    }

    // ===== IMPLEMENTATION 3: OPTIMIZED =====
    if (run_all || specific_impl == 3 || specific_impl == -1) {
        std::cout << "[3] Optimized (SIMD + Blocking):\n";
        omp_set_num_threads(num_threads);
        timer.start();
        correlate_optimized(ny, nx, data, result_opt);
        double time_opt = timer.elapsed();
        double speedup = (time_seq > 0) ? time_seq / time_opt : 1.0;
        std::cout << "    Time: " << std::fixed << std::setprecision(4) << time_opt << " seconds\n"
                  << "    GFLOPS: " << (2.0 * ny * ny * nx / 1e9 / time_opt) << "\n";
        if (speedup > 1.0) {
            std::cout << "    Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        std::cout << "\n";

        if (verify && time_seq > 0) {
            std::cout << "    Verification vs Sequential: ";
            if (verify_result(ny, result_seq, result_opt)) {
                std::cout << "✓ PASS\n\n";
            } else {
                std::cout << "✗ FAIL\n\n";
            }
        }
    }

    // ===== IMPLEMENTATION 4: HIGHLY OPTIMIZED =====
    if (run_all || specific_impl == 4 || specific_impl == -1) {
        std::cout << "[4] Highly Optimized (Aligned Memory + Advanced SIMD):\n";
        omp_set_num_threads(num_threads);
        timer.start();
        correlate_highly_optimized(ny, nx, data, result_hopt);
        double time_hopt = timer.elapsed();
        double speedup = (time_seq > 0) ? time_seq / time_hopt : 1.0;
        std::cout << "    Time: " << std::fixed << std::setprecision(4) << time_hopt << " seconds\n"
                  << "    GFLOPS: " << (2.0 * ny * ny * nx / 1e9 / time_hopt) << "\n";
        if (speedup > 1.0) {
            std::cout << "    Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        std::cout << "\n";

        if (verify && time_seq > 0) {
            std::cout << "    Verification vs Sequential: ";
            if (verify_result(ny, result_seq, result_hopt)) {
                std::cout << "✓ PASS\n\n";
            } else {
                std::cout << "✗ FAIL\n\n";
            }
        }
    }

    // ===== SUMMARY =====
    if (run_all) {
        std::cout << std::string(60, '=') << "\n";
        std::cout << "SUMMARY:\n\n";

        double time_seq_b = 0, time_par_b = 0, time_opt_b = 0, time_hopt_b = 0;

        timer.start();
        correlate_sequential(ny, nx, data, result_seq);
        time_seq_b = timer.elapsed();

        timer.start();
        correlate_parallel(ny, nx, data, result_par);
        time_par_b = timer.elapsed();

        timer.start();
        correlate_optimized(ny, nx, data, result_opt);
        time_opt_b = timer.elapsed();

        timer.start();
        correlate_highly_optimized(ny, nx, data, result_hopt);
        time_hopt_b = timer.elapsed();

        std::cout << std::left << std::setw(25) << "Implementation"
                  << std::right << std::setw(12) << "Time (s)"
                  << std::setw(12) << "Speedup"
                  << std::setw(12) << "GFLOPS\n";
        std::cout << std::string(60, '-') << "\n";

        std::cout << std::left << std::setw(25) << "Sequential"
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << time_seq_b
                  << std::setw(12) << "1.0x"
                  << std::setw(12) << std::setprecision(2) << (2.0 * ny * ny * nx / 1e9 / time_seq_b) << "\n";

        std::cout << std::left << std::setw(25) << "Parallel (OpenMP)"
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << time_par_b
                  << std::setw(12) << (time_seq_b / time_par_b) << "x"
                  << std::setw(12) << std::setprecision(2) << (2.0 * ny * ny * nx / 1e9 / time_par_b) << "\n";

        std::cout << std::left << std::setw(25) << "Optimized (SIMD+Block)"
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << time_opt_b
                  << std::setw(12) << (time_seq_b / time_opt_b) << "x"
                  << std::setw(12) << std::setprecision(2) << (2.0 * ny * ny * nx / 1e9 / time_opt_b) << "\n";

        std::cout << std::left << std::setw(25) << "Highly Optimized"
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << time_hopt_b
                  << std::setw(12) << (time_seq_b / time_hopt_b) << "x"
                  << std::setw(12) << std::setprecision(2) << (2.0 * ny * ny * nx / 1e9 / time_hopt_b) << "\n";

        double best_time = std::min({time_seq_b, time_par_b, time_opt_b, time_hopt_b});
        std::cout << "\nBest performance: ";
        if (best_time == time_hopt_b) {
            std::cout << "Highly Optimized (" << (time_seq_b / time_hopt_b) << "x speedup)\n";
        } else if (best_time == time_opt_b) {
            std::cout << "Optimized (" << (time_seq_b / time_opt_b) << "x speedup)\n";
        } else if (best_time == time_par_b) {
            std::cout << "Parallel (" << (time_seq_b / time_par_b) << "x speedup)\n";
        } else {
            std::cout << "Sequential (baseline)\n";
        }
    }

    std::cout << "\n";

    delete[] data;
    delete[] result_seq;
    delete[] result_par;
    delete[] result_opt;
    delete[] result_hopt;

    return 0;
}
