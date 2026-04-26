#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// ... [Metrics and Record structs remain exactly the same] ...
struct Metrics {
    std::string method;
    double compress_sec = 0.0;
    double query_sec = 0.0;
    double total_sec = 0.0;
    double avg_query_us = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    bool ok = false;
    std::string note;
};

struct Record {
    std::string experiment;
    std::string method;
    int threads = 1;
    int n = 0;
    int q = 0;
    std::string range_class;
    int max_len = 0;
    double total_sec = 0.0;
    double query_sec = 0.0;
    double avg_query_us = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double throughput_qps = 0.0;
    double memory_mb = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    std::string notes;
};

static std::string run_command_capture(const std::string& cmd) {
    std::array<char, 4096> buf{};
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
        out += buf.data();
    }
    pclose(pipe);
    return out;
}

static std::map<std::string, std::string> parse_kv_line(const std::string& line) {
    std::map<std::string, std::string> m;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto pos = tok.find('=');
        if (pos == std::string::npos) continue;
        m[tok.substr(0, pos)] = tok.substr(pos + 1);
    }
    return m;
}

static std::optional<Metrics> parse_metrics_from_output(const std::string& output) {
    std::stringstream ss(output);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.rfind("method=", 0) == 0) {
            auto kv = parse_kv_line(line);
            Metrics x;
            x.method = kv["method"];
            if (kv.count("compress_sec")) x.compress_sec = std::stod(kv["compress_sec"]);
            if (kv.count("query_sec")) x.query_sec = std::stod(kv["query_sec"]);
            if (kv.count("total_ms")) x.total_sec = std::stod(kv["total_ms"]) / 1000.0;
            if (kv.count("avg_query_us")) x.avg_query_us = std::stod(kv["avg_query_us"]);
            if (kv.count("h2d_ms")) x.h2d_ms = std::stod(kv["h2d_ms"]);
            if (kv.count("kernel_ms")) x.kernel_ms = std::stod(kv["kernel_ms"]);
            if (kv.count("d2h_ms")) x.d2h_ms = std::stod(kv["d2h_ms"]);
            if (x.total_sec == 0.0) x.total_sec = x.compress_sec + x.query_sec;
            if (x.query_sec == 0.0 && kv.count("total_ms")) x.query_sec = x.total_sec;
            x.ok = true;
            return x;
        }
    }
    return std::nullopt;
}

static double estimate_memory_mb(const std::string& method, int n, int q, int threads, int value_range) {
    const double bytes_per_mb = 1024.0 * 1024.0;
    int sigma = std::min(n, value_range);

    size_t arr_bytes = static_cast<size_t>(n) * sizeof(int);
    size_t query_bytes = static_cast<size_t>(q) * sizeof(int) * 3;
    size_t ans_bytes = static_cast<size_t>(q) * sizeof(int);
    size_t freq_bytes = static_cast<size_t>(sigma) * sizeof(int);

    if (method == "mos") {
        size_t total = arr_bytes * 2 + query_bytes + ans_bytes + freq_bytes;
        return static_cast<double>(total) / bytes_per_mb;
    }
    if (method == "omp_bruteforce") {
        size_t seen_all_threads = static_cast<size_t>(threads) * freq_bytes;
        size_t total = arr_bytes * 2 + query_bytes + ans_bytes + seen_all_threads;
        return static_cast<double>(total) / bytes_per_mb;
    }
    if (method == "cuda_bruteforce") {
        size_t host = arr_bytes + query_bytes + ans_bytes;
        size_t device = host;
        return static_cast<double>(host + device) / bytes_per_mb;
    }
    return 0.0;
}

static std::string range_name_for_max_len(int max_len) {
    if (max_len <= 128) return "small";
    if (max_len <= 4096) return "medium";
    return "large";
}

static std::optional<Metrics> run_method(
    const std::string& method,
    int n,
    int q,
    int value_range,
    int max_len,
    int threads,
    uint64_t seed
) {
    std::ostringstream cmd;
    if (method == "mos") {
        cmd << "./mos --n " << n << " --q " << q << " --value-range " << value_range
            << " --max-len " << max_len << " --seed " << seed;
    } else if (method == "omp_bruteforce") {
        cmd << "./omp_bruteforce --n " << n << " --q " << q << " --value-range " << value_range
            << " --max-len " << max_len << " --threads " << threads << " --seed " << seed;
    } else if (method == "cuda_bruteforce") {
        if (!std::filesystem::exists("./cuda_kernel")) return std::nullopt;
        cmd << "./cuda_kernel --n " << n << " --q " << q << " --value-range " << value_range
            << " --max-len " << max_len << " --seed " << seed;
    } else {
        return std::nullopt;
    }

    auto output = run_command_capture(cmd.str());
    auto m = parse_metrics_from_output(output);
    return m;
}

static void write_csv(const std::string& path, const std::vector<Record>& rows) {
    std::ofstream out(path);
    out << "experiment,method,threads,n,q,range_class,max_len,total_sec,query_sec,avg_query_us,speedup,efficiency,throughput_qps,memory_mb,h2d_ms,kernel_ms,d2h_ms,notes\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.experiment << ','
            << r.method << ','
            << r.threads << ','
            << r.n << ','
            << r.q << ','
            << r.range_class << ','
            << r.max_len << ','
            << r.total_sec << ','
            << r.query_sec << ','
            << r.avg_query_us << ','
            << r.speedup << ','
            << r.efficiency << ','
            << r.throughput_qps << ','
            << r.memory_mb << ','
            << r.h2d_ms << ','
            << r.kernel_ms << ','
            << r.d2h_ms << ','
            << r.notes << '\n';
    }
}

int main(int argc, char** argv) {
    int n = 1'000'000;
    int value_range = 1'000'000;
    uint64_t seed = 42;
    bool quick = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--value-range" && i + 1 < argc) value_range = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        else if (arg == "--quick") quick = true;
    }

    std::filesystem::create_directories("results");

    std::vector<int> q_values = quick ? std::vector<int>{5000, 10000, 20000} : std::vector<int>{10000, 30000, 60000, 100000};
    std::vector<int> range_lens = quick ? std::vector<int>{64, 512, 4096} : std::vector<int>{128, 2048, 32768};

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    std::vector<int> thread_values = {1, 2, 4, 8, 16, 32};
    thread_values.erase(
        std::remove_if(thread_values.begin(), thread_values.end(), [max_threads](int t) { return t > max_threads; }),
        thread_values.end()
    );
    if (thread_values.empty()) thread_values.push_back(1);

    std::vector<Record> rows;

    // A) Vary number of queries at medium range.
    for (int q : q_values) {
        int max_len = quick ? 512 : 2048;
        
        // FIX: Let OMP use max_threads, and remove the 1024 cap for CUDA
        auto mos = run_method("mos", n, q, value_range, max_len, 1, seed);
        auto omp_max = run_method("omp_bruteforce", n, q, value_range, max_len, max_threads, seed);
        auto cuda = run_method("cuda_bruteforce", n, q, value_range, max_len, 1, seed);

        if (mos) {
            rows.push_back({"vary_q", "mos", 1, n, q, "medium", max_len, mos->total_sec, mos->query_sec, mos->avg_query_us,
                1.0, 1.0, q / std::max(mos->query_sec, 1e-12), estimate_memory_mb("mos", n, q, 1, value_range),
                0.0, 0.0, 0.0, "cache-friendly query ordering"});
        }

        if (omp_max && mos) {
            double sp = mos->query_sec / std::max(omp_max->query_sec, 1e-12);
            rows.push_back({"vary_q", "omp_bruteforce", max_threads, n, q, "medium", max_len, omp_max->total_sec, omp_max->query_sec, omp_max->avg_query_us,
                sp, sp, q / std::max(omp_max->query_sec, 1e-12), estimate_memory_mb("omp_bruteforce", n, q, max_threads, value_range),
                0.0, 0.0, 0.0, "multithreaded CPU brute-force"});
        }

        if (cuda && mos) {
            double sp = mos->query_sec / std::max(cuda->query_sec, 1e-12);
            rows.push_back({"vary_q", "cuda_bruteforce", 1, n, q, "medium", max_len, cuda->total_sec, cuda->query_sec,
                cuda->avg_query_us, sp, sp, q / std::max(cuda->query_sec, 1e-12), estimate_memory_mb("cuda_bruteforce", n, q, 1, value_range),
                cuda->h2d_ms, cuda->kernel_ms, cuda->d2h_ms, "one GPU thread per query"});
        }
    }

    // B) Vary range size for fixed Q.
    int fixed_q = quick ? 15000 : 50000;
    for (int max_len : range_lens) {
        std::string rc = range_name_for_max_len(max_len);

        // FIX: Let OMP use max_threads, and remove the 1024 cap for CUDA
        auto mos = run_method("mos", n, fixed_q, value_range, max_len, 1, seed + max_len);
        auto omp_max = run_method("omp_bruteforce", n, fixed_q, value_range, max_len, max_threads, seed + max_len);
        auto cuda = run_method("cuda_bruteforce", n, fixed_q, value_range, max_len, 1, seed + max_len);

        if (mos) {
            rows.push_back({"vary_range", "mos", 1, n, fixed_q, rc, max_len, mos->total_sec, mos->query_sec, mos->avg_query_us,
                1.0, 1.0, fixed_q / std::max(mos->query_sec, 1e-12), estimate_memory_mb("mos", n, fixed_q, 1, value_range),
                0.0, 0.0, 0.0, "high locality from Mo ordering"});
        }
        if (omp_max && mos) {
            double sp = mos->query_sec / std::max(omp_max->query_sec, 1e-12);
            rows.push_back({"vary_range", "omp_bruteforce", max_threads, n, fixed_q, rc, max_len, omp_max->total_sec, omp_max->query_sec, omp_max->avg_query_us,
                sp, sp, fixed_q / std::max(omp_max->query_sec, 1e-12), estimate_memory_mb("omp_bruteforce", n, fixed_q, max_threads, value_range),
                0.0, 0.0, 0.0, "multithreaded brute force"});
        }
        if (cuda && mos) {
            double sp = mos->query_sec / std::max(cuda->query_sec, 1e-12);
            rows.push_back({"vary_range", "cuda_bruteforce", 1, n, fixed_q, rc, max_len, cuda->total_sec, cuda->query_sec,
                cuda->avg_query_us, sp, sp, fixed_q / std::max(cuda->query_sec, 1e-12), estimate_memory_mb("cuda_bruteforce", n, fixed_q, 1, value_range),
                cuda->h2d_ms, cuda->kernel_ms, cuda->d2h_ms, "GPU transfer+kernel breakdown"});
        }
    }

    // C) Strong scaling for OpenMP (fixed Q).
    int strong_q = quick ? 20000 : 100000;
    int strong_len = quick ? 512 : 2048;
    auto omp_baseline = run_method("omp_bruteforce", n, strong_q, value_range, strong_len, 1, seed + 111);
    for (int t : thread_values) {
        auto ompt = run_method("omp_bruteforce", n, strong_q, value_range, strong_len, t, seed + 111);
        if (!ompt || !omp_baseline) continue;
        double sp = omp_baseline->query_sec / std::max(ompt->query_sec, 1e-12);
        double eff = sp / std::max(1, t);
        rows.push_back({"strong_scaling", "omp_bruteforce", t, n, strong_q, "medium", strong_len,
            ompt->total_sec, ompt->query_sec, ompt->avg_query_us, sp, eff,
            strong_q / std::max(ompt->query_sec, 1e-12), estimate_memory_mb("omp_bruteforce", n, strong_q, t, value_range),
            0.0, 0.0, 0.0, "speedup against t=1"});
    }

    // D) Weak scaling for OpenMP (Q grows with threads).
    int weak_base_q = quick ? 4000 : 10000;
    int weak_len = quick ? 512 : 2048;
    for (int t : thread_values) {
        int weak_q = weak_base_q * t;
        auto ompt = run_method("omp_bruteforce", n, weak_q, value_range, weak_len, t, seed + 222 + t);
        if (!ompt) continue;
        rows.push_back({"weak_scaling", "omp_bruteforce", t, n, weak_q, "medium", weak_len,
            ompt->total_sec, ompt->query_sec, ompt->avg_query_us, 0.0, 0.0,
            weak_q / std::max(ompt->query_sec, 1e-12), estimate_memory_mb("omp_bruteforce", n, weak_q, t, value_range),
            0.0, 0.0, 0.0, "Q proportional to thread count"});
    }

    write_csv("results/data.csv", rows);

    std::cout << "Wrote " << rows.size() << " rows to results/data.csv\n";
    std::cout << "Note: cache metrics can be collected externally via perf, e.g.:\n";
    std::cout << "perf stat -e L1-dcache-load-misses,cycles,instructions ./mos --n " << n << " --q " << (quick ? 20000 : 100000) << " --max-len 2048\n";

    return 0;
}