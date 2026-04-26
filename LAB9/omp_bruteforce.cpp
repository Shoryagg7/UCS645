#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

struct Query {
    int l;
    int r;
    int idx;
};

static std::vector<int> compress_values(const std::vector<int>& a) {
    std::vector<int> vals = a;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    std::vector<int> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<int>(std::lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
    }
    return out;
}

// Precompute the index of the previous occurrence for each element
static std::vector<int> compute_prev_array(const std::vector<int>& arr_comp) {
    int sigma = 0;
    if (!arr_comp.empty()) {
        sigma = *std::max_element(arr_comp.begin(), arr_comp.end()) + 1;
    }
    
    std::vector<int> prev(arr_comp.size());
    std::vector<int> last_seen(sigma, -1);
    
    for (size_t i = 0; i < arr_comp.size(); ++i) {
        prev[i] = last_seen[arr_comp[i]];
        last_seen[arr_comp[i]] = static_cast<int>(i);
    }
    return prev;
}

static std::vector<Query> generate_queries(int n, int q, int max_len, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> left_dist(0, n - 1);
    std::uniform_int_distribution<int> len_dist(1, std::max(1, max_len));

    std::vector<Query> queries;
    queries.reserve(q);
    for (int i = 0; i < q; ++i) {
        int l = left_dist(rng);
        int len = len_dist(rng);
        int r = std::min(n - 1, l + len - 1);
        queries.push_back({l, r, i});
    }
    return queries;
}

// -------------------------------------------------------------------------
// ENHANCED CPU ALGORITHM: Sequential Memory Access + SIMD Vectorization
// -------------------------------------------------------------------------
static std::vector<int> omp_bruteforce_distinct(
    const std::vector<int>& prev,
    const std::vector<Query>& queries,
    int threads
) {
    const int q = static_cast<int>(queries.size());
    std::vector<int> ans(q, 0);

    omp_set_num_threads(threads);

    // We no longer need to allocate a massive thread-local `seen` array!
#pragma omp parallel for schedule(dynamic, 32)
    for (int qi = 0; qi < q; ++qi) {
        int cnt = 0;
        int l = queries[qi].l;
        int r = queries[qi].r;

        // Force the compiler to use AVX/AVX2 SIMD instructions 
#pragma omp simd reduction(+:cnt)
        for (int i = l; i <= r; ++i) {
            // Because memory access is sequential, the hardware prefetcher
            // guarantees L1 cache hits almost 100% of the time.
            if (prev[i] < l) {
                cnt++;
            }
        }
        ans[queries[qi].idx] = cnt;
    }

    return ans;
}

static std::vector<int> serial_bruteforce_distinct(const std::vector<int>& arr_comp, const std::vector<Query>& queries) {
    const int sigma = *std::max_element(arr_comp.begin(), arr_comp.end()) + 1;
    std::vector<int> ans(queries.size(), 0);
    std::vector<int> seen(sigma, -1);

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        int cnt = 0;
        for (int i = queries[qi].l; i <= queries[qi].r; ++i) {
            int v = arr_comp[i];
            if (seen[v] != static_cast<int>(qi)) {
                seen[v] = static_cast<int>(qi);
                ++cnt;
            }
        }
        ans[queries[qi].idx] = cnt;
    }
    return ans;
}

int main(int argc, char** argv) {
    int n = 1'000'000;
    int q = 100'000;
    int value_range = 1'000'000;
    int max_len = 10'000;
    int threads = omp_get_max_threads();
    uint64_t seed = 42;
    bool verify = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--q" && i + 1 < argc) q = std::stoi(argv[++i]);
        else if (arg == "--value-range" && i + 1 < argc) value_range = std::stoi(argv[++i]);
        else if (arg == "--max-len" && i + 1 < argc) max_len = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        else if (arg == "--verify") verify = true;
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> val_dist(0, value_range - 1);

    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) arr[i] = val_dist(rng);

    auto queries = generate_queries(n, q, max_len, seed + 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Preparation Phase
    auto arr_comp = compress_values(arr);
    auto prev_array = compute_prev_array(arr_comp); // Added mathematical pre-computation
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // We now pass `prev_array` instead of `arr_comp`
    auto ans = omp_bruteforce_distinct(prev_array, queries, threads);
    
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> compress_s = t1 - t0;
    std::chrono::duration<double> query_s = t2 - t1;

    std::cout << "method=omp_bruteforce"
              << ",threads=" << threads
              << ",n=" << n
              << ",q=" << q
              << ",compress_sec=" << compress_s.count()
              << ",query_sec=" << query_s.count()
              << ",avg_query_us=" << (query_s.count() * 1e6 / std::max(1, q))
              << "\n";

    if (verify && q <= 5000 && n <= 200000) {
        auto chk = serial_bruteforce_distinct(arr_comp, queries);
        bool ok = (chk == ans);
        std::cout << "verify=" << (ok ? "PASS" : "FAIL") << "\n";
        if (!ok) return 2;
    }

    uint64_t checksum = 0;
    for (int x : ans) checksum = checksum * 11400714819323198485ULL + static_cast<uint64_t>(x + 7);
    std::cout << "checksum=" << checksum << "\n";

    return 0;
}