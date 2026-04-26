#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct Query {
    int l;
    int r;
    int idx;
};

struct MoQuery {
    int l;
    int r;
    int idx;
    int64_t hilbert_ord;
};

// -------------------------------------------------------------------------
// HILBERT CURVE MAPPING
// Converts a 2D coordinate (L, R) into a 1D position on the Hilbert Curve.
// Using 21 bits supports arrays up to size 2,097,152 (2^21).
// -------------------------------------------------------------------------
inline int64_t get_hilbert_order(int x, int y) {
    int64_t d = 0;
    for (int s = 1 << 21; s > 0; s >>= 1) {
        int rx = (x & s) > 0;
        int ry = (y & s) > 0;
        d += static_cast<int64_t>(s) * s * ((3 * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = (1 << 22) - 1 - x;
                y = (1 << 22) - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

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
// ENHANCED MO'S ALGORITHM
// -------------------------------------------------------------------------
static std::vector<int> mos_distinct(const std::vector<int>& arr_comp, const std::vector<Query>& queries) {
    const int q = static_cast<int>(queries.size());
    if (q == 0) return {};
    
    // Safety check to prevent segfaults on empty arrays
    int sigma = 0;
    if (!arr_comp.empty()) {
        sigma = *std::max_element(arr_comp.begin(), arr_comp.end()) + 1;
    }

    std::vector<MoQuery> mo_queries;
    mo_queries.reserve(q);
    
    // Calculate the Hilbert curve order for every query
    for (const auto& qu : queries) {
        mo_queries.push_back({qu.l, qu.r, qu.idx, get_hilbert_order(qu.l, qu.r)});
    }

    // Sort by 1D Hilbert Curve distance. 
    std::sort(mo_queries.begin(), mo_queries.end(), [](const MoQuery& a, const MoQuery& b) {
        return a.hilbert_ord < b.hilbert_ord;
    });

    std::vector<int> freq(sigma, 0);
    std::vector<int> ans(q, 0);

    // Initialize pointers. cur_l must be strictly > cur_r to represent an empty range.
    int cur_l = 0;
    int cur_r = -1;
    int distinct = 0;

    // FIX: Removed the invalid 'inline' keyword. 
    // Lambdas are implicitly inline in C++, the compiler will optimize this natively.
    auto add = [&](int pos) {
        if (freq[arr_comp[pos]]++ == 0) {
            ++distinct;
        }
    };

    auto remove = [&](int pos) {
        if (--freq[arr_comp[pos]] == 0) {
            --distinct;
        }
    };

    for (const auto& qu : mo_queries) {
        // ALWAYS expand the range first to avoid invalid negative range bounds
        while (cur_l > qu.l) add(--cur_l);
        while (cur_r < qu.r) add(++cur_r);
        
        // THEN shrink the range
        while (cur_l < qu.l) remove(cur_l++);
        while (cur_r > qu.r) remove(cur_r--);
        
        ans[qu.idx] = distinct;
    }

    return ans;
}

static std::vector<int> brute_distinct_check(const std::vector<int>& arr_comp, const std::vector<Query>& queries) {
    int sigma = 0;
    if (!arr_comp.empty()) {
        sigma = *std::max_element(arr_comp.begin(), arr_comp.end()) + 1;
    }
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
    uint64_t seed = 42;
    bool verify = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--q" && i + 1 < argc) q = std::stoi(argv[++i]);
        else if (arg == "--value-range" && i + 1 < argc) value_range = std::stoi(argv[++i]);
        else if (arg == "--max-len" && i + 1 < argc) max_len = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        else if (arg == "--verify") verify = true;
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> val_dist(0, value_range - 1);

    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) arr[i] = val_dist(rng);

    auto queries = generate_queries(n, q, max_len, seed + 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto arr_comp = compress_values(arr);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    auto ans = mos_distinct(arr_comp, queries);
    
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> compress_s = t1 - t0;
    std::chrono::duration<double> mo_s = t2 - t1;

    std::cout << "method=mos"
              << ",n=" << n
              << ",q=" << q
              << ",compress_sec=" << compress_s.count()
              << ",query_sec=" << mo_s.count()
              << ",avg_query_us=" << (mo_s.count() * 1e6 / std::max(1, q))
              << "\n";

    if (verify && q <= 5000 && n <= 200000) {
        auto chk = brute_distinct_check(arr_comp, queries);
        bool ok = (chk == ans);
        std::cout << "verify=" << (ok ? "PASS" : "FAIL") << "\n";
        if (!ok) return 2;
    }

    uint64_t checksum = 0;
    for (int x : ans) checksum = checksum * 1315423911ULL + static_cast<uint64_t>(x + 1);
    std::cout << "checksum=" << checksum << "\n";

    return 0;
}