#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <iomanip>

using namespace std;

const double EPSILON = 1.0;        // Lennard-Jones parameter: energy
const double SIGMA = 1.0;          // Lennard-Jones parameter: distance
const double CUTOFF = 2.5 * SIGMA; // Cutoff radius
const double CUTOFF_SQ = CUTOFF * CUTOFF;

struct Particle {
    double x, y, z;      // Position
    double fx, fy, fz;   // Force
    double vx, vy, vz;   // Velocity
};

// Compute Lennard-Jones force and potential energy between two particles
void computeLJPair(const Particle& p1, const Particle& p2,
                   double& force_x, double& force_y, double& force_z, double& potential) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dz = p2.z - p1.z;

    double r2 = dx*dx + dy*dy + dz*dz;

    if (r2 > CUTOFF_SQ || r2 < 1e-9) {
        force_x = force_y = force_z = 0.0;
        potential = 0.0;
        return;
    }

    double r2_inv = 1.0 / r2;
    double r6_inv = r2_inv * r2_inv * r2_inv;
    double r12_inv = r6_inv * r6_inv;

    // Lennard-Jones potential: V = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
    potential = 4.0 * EPSILON * (r12_inv - r6_inv);

    // Force magnitude: F = -dV/dr
    double force_mag = 48.0 * EPSILON * r2_inv * (r12_inv - 0.5 * r6_inv);

    force_x = force_mag * dx;
    force_y = force_mag * dy;
    force_z = force_mag * dz;
}

int main() {
    int num_particles = 1000;

    // Initialize particles randomly
    vector<Particle> particles(num_particles);
    for (int i = 0; i < num_particles; i++) {
        particles[i].x = (double)rand() / RAND_MAX * 10.0;
        particles[i].y = (double)rand() / RAND_MAX * 10.0;
        particles[i].z = (double)rand() / RAND_MAX * 10.0;
        particles[i].vx = particles[i].vy = particles[i].vz = 0.0;
        particles[i].fx = particles[i].fy = particles[i].fz = 0.0;
    }

    ofstream results("md_results.txt");
    results << "#Threads\tTime(s)\tSpeedup\tTotal_Energy\n";

    double serial_time = 0.0;
    double baseline_energy = 0.0;

    // Test with different thread counts
    for (int num_threads = 1; num_threads <= 16; num_threads++) {
        // Reset forces and energy
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_particles; i++) {
            particles[i].fx = particles[i].fy = particles[i].fz = 0.0;
        }

        double total_energy = 0.0;
        double start = omp_get_wtime();

        // Parallel force calculation
        #pragma omp parallel for num_threads(num_threads) reduction(+:total_energy)
        for (int i = 0; i < num_particles; i++) {
            for (int j = i + 1; j < num_particles; j++) {
                double fx, fy, fz, pe;
                computeLJPair(particles[i], particles[j], fx, fy, fz, pe);

                // Newton's third law - action-reaction
                #pragma omp atomic
                particles[i].fx += fx;
                #pragma omp atomic
                particles[i].fy += fy;
                #pragma omp atomic
                particles[i].fz += fz;

                #pragma omp atomic
                particles[j].fx -= fx;
                #pragma omp atomic
                particles[j].fy -= fy;
                #pragma omp atomic
                particles[j].fz -= fz;

                total_energy += pe;
            }
        }

        double end = omp_get_wtime();
        double elapsed = end - start;

        if (num_threads == 1) {
            serial_time = elapsed;
            baseline_energy = total_energy;
        }

        double speedup = serial_time / elapsed;

        cout << fixed << setprecision(6);
        cout << "Threads: " << setw(2) << num_threads
             << " | Time: " << setw(10) << elapsed
             << " s | Speedup: " << setw(8) << speedup
             << " | Energy: " << setw(12) << total_energy << endl;

        results << num_threads << "\t" << elapsed << "\t" << speedup << "\t" << total_energy << "\n";
    }

    results.close();
    cout << "\nResults saved to md_results.txt\n";

    return 0;
}
