#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Query {
    int l;
    int r;
    int idx;
};

static std::vector<Query> generate_queries(int n, int q, int max_len, uint64_t seed, bool skewed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> left_dist(0, n - 1);
    
    // Uniform distribution vs Skewed (Normal) distribution for query lengths
    std::uniform_int_distribution<int> uni_len_dist(1, std::max(1, max_len));
    std::normal_distribution<double> norm_len_dist(max_len / 2.0, max_len / 6.0); 

    std::vector<Query> queries;
    queries.reserve(q);
    for (int i = 0; i < q; ++i) {
        int l = left_dist(rng);
        
        int len;
        if (skewed) {
            // Force normal distribution into bounds
            do {
                len = static_cast<int>(std::round(norm_len_dist(rng)));
            } while (len < 1 || len > max_len);
        } else {
            len = uni_len_dist(rng);
        }

        int r = std::min(n - 1, l + len - 1);
        queries.push_back({l, r, i});
    }
    return queries;
}

// --- CSV Output (For Debugging / Human Verification) ---
static void write_array_csv(const std::string& path, const std::vector<int>& arr) {
    std::ofstream out(path);
    out << "idx,value\n";
    for (size_t i = 0; i < arr.size(); ++i) {
        out << i << ',' << arr[i] << '\n';
    }
}

static void write_queries_csv(const std::string& path, const std::vector<Query>& qs) {
    std::ofstream out(path);
    out << "idx,l,r,length\n";
    for (const auto& q : qs) {
        out << q.idx << ',' << q.l << ',' << q.r << ',' << (q.r - q.l + 1) << '\n';
    }
}

// --- Binary Output (For High-Speed Benchmark Loading) ---
static void write_array_binary(const std::string& path, const std::vector<int>& arr) {
    std::ofstream out(path, std::ios::binary);
    int size = arr.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(int));
    out.write(reinterpret_cast<const char*>(arr.data()), size * sizeof(int));
}

static void write_queries_binary(const std::string& path, const std::vector<Query>& qs) {
    std::ofstream out(path, std::ios::binary);
    int size = qs.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(int));
    out.write(reinterpret_cast<const char*>(qs.data()), size * sizeof(Query));
}

int main(int argc, char** argv) {
    int n = 1'000'000;
    int q = 100'000;
    int value_range = 1'000'000;
    uint64_t seed = 42;
    std::string out_dir = "results";
    bool skewed = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--q" && i + 1 < argc) q = std::stoi(argv[++i]);
        else if (arg == "--value-range" && i + 1 < argc) value_range = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        else if (arg == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (arg == "--skewed") skewed = true;
    }

    std::filesystem::create_directories(out_dir);

    std::mt19937_64 rng(seed);
    
    // Apply skew to array values if requested
    std::vector<int> arr(n);
    if (skewed) {
        std::normal_distribution<double> val_dist(value_range / 2.0, value_range / 6.0);
        for (int i = 0; i < n; ++i) {
            int val;
            do {
                val = static_cast<int>(std::round(val_dist(rng)));
            } while (val < 0 || val >= value_range);
            arr[i] = val;
        }
    } else {
        std::uniform_int_distribution<int> val_dist(0, value_range - 1);
        for (int i = 0; i < n; ++i) {
            arr[i] = val_dist(rng);
        }
    }

    auto small_q = generate_queries(n, q, 128, seed + 1, skewed);
    auto medium_q = generate_queries(n, q, 2048, seed + 2, skewed);
    auto large_q = generate_queries(n, q, 32768, seed + 3, skewed);

    // Write CSVs
    write_array_csv(out_dir + "/array.csv", arr);
    write_queries_csv(out_dir + "/queries_small.csv", small_q);
    write_queries_csv(out_dir + "/queries_medium.csv", medium_q);
    write_queries_csv(out_dir + "/queries_large.csv", large_q);

    // Write Binaries
    write_array_binary(out_dir + "/array.bin", arr);
    write_queries_binary(out_dir + "/queries_small.bin", small_q);
    write_queries_binary(out_dir + "/queries_medium.bin", medium_q);
    write_queries_binary(out_dir + "/queries_large.bin", large_q);

    std::ofstream meta(out_dir + "/input_meta.txt");
    meta << "seed=" << seed << "\n";
    meta << "n=" << n << "\n";
    meta << "q=" << q << "\n";
    meta << "value_range=" << value_range << "\n";
    meta << "skewed=" << (skewed ? "true" : "false") << "\n";
    meta << "small_max_len=128\n";
    meta << "medium_max_len=2048\n";
    meta << "large_max_len=32768\n";

    std::cout << "Generated input datasets in " << out_dir << "\n";
    std::cout << "-> Included ultra-fast .bin files for C++/CUDA reading.\n";
    return 0;
}