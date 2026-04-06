#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <cmath>

using namespace std;

class HeatDiffusion {
public:
    int grid_size;
    double dt;           // Time step
    double dx;           // Spatial step
    double alpha;        // Thermal diffusivity
    vector<vector<double>> temperature;
    vector<vector<double>> next_temperature;

    HeatDiffusion(int size, double thermal_alpha = 0.1)
        : grid_size(size), alpha(thermal_alpha) {
        dx = 1.0 / (grid_size - 1);
        dt = 0.001;  // Small time step for stability

        temperature.assign(grid_size, vector<double>(grid_size, 0.0));
        next_temperature.assign(grid_size, vector<double>(grid_size, 0.0));

        // Initialize with heat source in the center
        temperature[grid_size/2][grid_size/2] = 100.0;

        // Set boundary conditions (edges at 0 temperature)
        for (int i = 0; i < grid_size; i++) {
            temperature[0][i] = 0.0;
            temperature[grid_size-1][i] = 0.0;
            temperature[i][0] = 0.0;
            temperature[i][grid_size-1] = 0.0;
        }
    }

    // Serial implementation
    void stepSerial() {
        double r = alpha * dt / (dx * dx);  // Stability parameter

        for (int i = 1; i < grid_size - 1; i++) {
            for (int j = 1; j < grid_size - 1; j++) {
                next_temperature[i][j] = temperature[i][j] +
                    r * (temperature[i+1][j] + temperature[i-1][j] +
                         temperature[i][j+1] + temperature[i][j-1] -
                         4.0 * temperature[i][j]);
            }
        }

        // Boundary conditions
        for (int i = 0; i < grid_size; i++) {
            next_temperature[0][i] = 0.0;
            next_temperature[grid_size-1][i] = 0.0;
            next_temperature[i][0] = 0.0;
            next_temperature[i][grid_size-1] = 0.0;
        }

        swap(temperature, next_temperature);
    }

    // Parallel implementation with different scheduling strategies
    void stepParallel(int num_threads, const string& schedule_type = "static", int chunk_size = 1) {
        double r = alpha * dt / (dx * dx);

        const char* schedule_pragma;
        if (schedule_type == "static") {
            schedule_pragma = "schedule(static)";
        } else if (schedule_type == "dynamic") {
            schedule_pragma = "schedule(dynamic, chunk)";
        } else {
            schedule_pragma = "schedule(guided)";
        }

        // Main computation with proper scheduling
        if (schedule_type == "static") {
            #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
            for (int i = 1; i < grid_size - 1; i++) {
                for (int j = 1; j < grid_size - 1; j++) {
                    next_temperature[i][j] = temperature[i][j] +
                        r * (temperature[i+1][j] + temperature[i-1][j] +
                             temperature[i][j+1] + temperature[i][j-1] -
                             4.0 * temperature[i][j]);
                }
            }
        } else if (schedule_type == "dynamic") {
            #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(dynamic, chunk_size)
            for (int i = 1; i < grid_size - 1; i++) {
                for (int j = 1; j < grid_size - 1; j++) {
                    next_temperature[i][j] = temperature[i][j] +
                        r * (temperature[i+1][j] + temperature[i-1][j] +
                             temperature[i][j+1] + temperature[i][j-1] -
                             4.0 * temperature[i][j]);
                }
            }
        } else {
            #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(guided)
            for (int i = 1; i < grid_size - 1; i++) {
                for (int j = 1; j < grid_size - 1; j++) {
                    next_temperature[i][j] = temperature[i][j] +
                        r * (temperature[i+1][j] + temperature[i-1][j] +
                             temperature[i][j+1] + temperature[i][j-1] -
                             4.0 * temperature[i][j]);
                }
            }
        }

        // Boundary conditions
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < grid_size; i++) {
            next_temperature[0][i] = 0.0;
            next_temperature[grid_size-1][i] = 0.0;
            next_temperature[i][0] = 0.0;
            next_temperature[i][grid_size-1] = 0.0;
        }

        swap(temperature, next_temperature);
    }

    double getTotalHeat() {
        double total = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:total)
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                total += temperature[i][j];
            }
        }
        return total;
    }

    double getMaxTemperature() {
        double max_temp = 0.0;
        #pragma omp parallel for collapse(2) reduction(max:max_temp)
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                max_temp = max(max_temp, temperature[i][j]);
            }
        }
        return max_temp;
    }
};

int main() {
    int grid_size = 500;   // 500x500 grid
    int time_steps = 1000;  // Number of time steps to simulate

    cout << "\nHeat Diffusion Simulation (2D Finite Difference Method)\n";
    cout << "=" << string(60, '=') << "\n";
    cout << "Grid size: " << grid_size << " x " << grid_size << "\n";
    cout << "Time steps: " << time_steps << "\n";
    cout << "Total cells: " << (grid_size * grid_size) << "\n\n";

    ofstream results("heatsim_results.txt");
    results << "#Threads\tSchedule\tTime(s)\tSpeedup\tTotal_Heat\tMax_Temp\n";

    double serial_time = 0.0;
    double baseline_heat = 0.0;
    double baseline_max = 0.0;

    // Test different scheduling strategies
    vector<string> schedules = {"static", "dynamic", "guided"};

    for (const auto& schedule_type : schedules) {
        cout << "Schedule: " << schedule_type << "\n";

        for (int num_threads = 1; num_threads <= 16; num_threads++) {
            HeatDiffusion simu(grid_size);

            double start = omp_get_wtime();

            // Run simulation
            for (int step = 0; step < time_steps; step++) {
                if (num_threads == 1) {
                    simu.stepSerial();
                } else {
                    simu.stepParallel(num_threads, schedule_type);
                }
            }

            double end = omp_get_wtime();
            double elapsed = end - start;

            double total_heat = simu.getTotalHeat();
            double max_temp = simu.getMaxTemperature();

            if (num_threads == 1) {
                serial_time = elapsed;
                baseline_heat = total_heat;
                baseline_max = max_temp;
            }

            double speedup = serial_time / elapsed;

            cout << fixed << setprecision(6);
            cout << "  Threads: " << setw(2) << num_threads
                 << " | Time: " << setw(10) << elapsed
                 << " s | Speedup: " << setw(8) << speedup << endl;

            results << num_threads << "\t" << schedule_type << "\t"
                   << elapsed << "\t" << speedup << "\t"
                   << total_heat << "\t" << max_temp << "\n";
        }
        cout << "\n";
    }

    results.close();
    cout << "Results saved to heatsim_results.txt\n";

    return 0;
}
