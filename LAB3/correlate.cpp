#include "correlate.h"
#include <cmath>
#include <cstring>
#include <omp.h>

// ======================================================================
// IMPLEMENTATION 1: SEQUENTIAL BASELINE
// ======================================================================
void correlate_sequential(int ny, int nx, const float* data, float* result)
{
    double* means = new double[ny];
    double* stddevs = new double[ny];

    // Calculate mean and stddev for each row
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int j = 0; j < nx; j++) {
            float val = data[i * nx + j];
            sum += val;
            sum_sq += val * val;
        }

        means[i] = sum / nx;
        double variance = (sum_sq / nx) - (means[i] * means[i]);
        stddevs[i] = std::sqrt(std::max(0.0, variance));
    }

    // Calculate correlation coefficients
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double covariance = 0.0;

            for (int k = 0; k < nx; k++) {
                double xi = data[i * nx + k] - means[i];
                double yi = data[j * nx + k] - means[j];
                covariance += xi * yi;
            }

            covariance /= nx;

            double denom = stddevs[i] * stddevs[j];
            if (denom > 1e-10) {
                result[i + j * ny] = (float)(covariance / denom);
            } else {
                result[i + j * ny] = 0.0f;
            }
        }
    }

    delete[] means;
    delete[] stddevs;
}

// ======================================================================
// IMPLEMENTATION 2: PARALLEL WITH OpenMP
// ======================================================================
void correlate_parallel(int ny, int nx, const float* data, float* result)
{
    double* means = new double[ny];
    double* stddevs = new double[ny];

    // Parallel mean calculation
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int j = 0; j < nx; j++) {
            float val = data[i * nx + j];
            sum += val;
            sum_sq += val * val;
        }

        means[i] = sum / nx;
        double variance = (sum_sq / nx) - (means[i] * means[i]);
        stddevs[i] = std::sqrt(std::max(0.0, variance));
    }

    // Parallel correlation calculation
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double covariance = 0.0;

            for (int k = 0; k < nx; k++) {
                double xi = data[i * nx + k] - means[i];
                double yi = data[j * nx + k] - means[j];
                covariance += xi * yi;
            }

            covariance /= nx;

            double denom = stddevs[i] * stddevs[j];
            if (denom > 1e-10) {
                result[i + j * ny] = (float)(covariance / denom);
            } else {
                result[i + j * ny] = 0.0f;
            }
        }
    }

    delete[] means;
    delete[] stddevs;
}

// ======================================================================
// IMPLEMENTATION 3: OPTIMIZED WITH SIMD & LOOP UNROLLING
// ======================================================================
void correlate_optimized(int ny, int nx, const float* data, float* result)
{
    double* means = new double[ny];
    double* stddevs = new double[ny];

    // Parallel mean calculation with SIMD
    #pragma omp parallel for simd
    for (int i = 0; i < ny; i++) {
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int j = 0; j < nx; j++) {
            float val = data[i * nx + j];
            sum += val;
            sum_sq += val * val;
        }

        means[i] = sum / nx;
        double variance = (sum_sq / nx) - (means[i] * means[i]);
        stddevs[i] = std::sqrt(std::max(0.0, variance));
    }

    // Optimized correlation with loop unrolling
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double covariance = 0.0;

            // Unrolled inner loop (4x unrolling)
            int k = 0;
            for (; k + 4 <= nx; k += 4) {
                double xi0 = data[i * nx + k] - means[i];
                double yi0 = data[j * nx + k] - means[j];
                double xi1 = data[i * nx + k + 1] - means[i];
                double yi1 = data[j * nx + k + 1] - means[j];
                double xi2 = data[i * nx + k + 2] - means[i];
                double yi2 = data[j * nx + k + 2] - means[j];
                double xi3 = data[i * nx + k + 3] - means[i];
                double yi3 = data[j * nx + k + 3] - means[j];

                covariance += xi0 * yi0 + xi1 * yi1 + xi2 * yi2 + xi3 * yi3;
            }

            // Handle remainder
            for (; k < nx; k++) {
                double xi = data[i * nx + k] - means[i];
                double yi = data[j * nx + k] - means[j];
                covariance += xi * yi;
            }

            covariance /= nx;

            double denom = stddevs[i] * stddevs[j];
            if (denom > 1e-10) {
                result[i + j * ny] = (float)(covariance / denom);
            } else {
                result[i + j * ny] = 0.0f;
            }
        }
    }

    delete[] means;
    delete[] stddevs;
}

// ======================================================================
// IMPLEMENTATION 4: HIGHLY OPTIMIZED (SIMD + ALIGNED MEMORY)
// ======================================================================
void correlate_highly_optimized(int ny, int nx, const float* data, float* result)
{
    double* means = new double[ny];
    double* stddevs = new double[ny];

    // Phase 1: Vectorized mean and stddev calculation
    #pragma omp parallel for simd
    for (int i = 0; i < ny; i++) {
        double sum = 0.0;
        double sum_sq = 0.0;

        #pragma omp simd reduction(+:sum, sum_sq)
        for (int j = 0; j < nx; j++) {
            float val = data[i * nx + j];
            sum += val;
            sum_sq += val * val;
        }

        means[i] = sum / nx;
        double variance = (sum_sq / nx) - (means[i] * means[i]);
        stddevs[i] = std::sqrt(std::max(0.0, variance));
    }

    // Phase 2: Optimized correlation with SIMD inner loop
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double covariance = 0.0;

            // SIMD-enabled inner loop with reduction
            #pragma omp simd reduction(+:covariance)
            for (int k = 0; k < nx; k++) {
                double xi = data[i * nx + k] - means[i];
                double yi = data[j * nx + k] - means[j];
                covariance += xi * yi;
            }

            covariance /= nx;

            double denom = stddevs[i] * stddevs[j];
            if (denom > 1e-10) {
                result[i + j * ny] = (float)(covariance / denom);
            } else {
                result[i + j * ny] = 0.0f;
            }
        }
    }

    delete[] means;
    delete[] stddevs;
}
