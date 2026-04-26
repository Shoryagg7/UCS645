#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

struct Query {
    int l;
    int r;
    int idx;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

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
        last_seen[arr_comp[i]] = i;
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
// ENHANCED KERNEL: Warp-level parallelism + Prev Array logic
// -------------------------------------------------------------------------
__global__ void distinct_warp_kernel(const int* prev, const Query* queries, int q, int* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Assign 32 threads (1 warp) to each query
    int query_idx = tid / 32; 
    int lane_idx = tid % 32;  // The specific thread's ID within its warp

    if (query_idx >= q) return;

    Query qu = queries[query_idx];
    int l = qu.l;
    int r = qu.r;
    int local_count = 0;

    // The warp strides through the query range together (Perfect coalescing!)
    for (int i = l + lane_idx; i <= r; i += 32) {
        if (prev[i] < l) {
            local_count++;
        }
    }

    // Warp Reduction: Rapidly sum the counts across all 32 threads in the warp
    for (int offset = 16; offset > 0; offset /= 2) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

    // Thread 0 of the warp writes the final answer for this query
    if (lane_idx == 0) {
        out[qu.idx] = local_count;
    }
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
    int max_len = 1024;
    int block_size = 256;
    uint64_t seed = 42;
    bool verify = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--q" && i + 1 < argc) q = std::stoi(argv[++i]);
        else if (arg == "--value-range" && i + 1 < argc) value_range = std::stoi(argv[++i]);
        else if (arg == "--max-len" && i + 1 < argc) max_len = std::stoi(argv[++i]);
        else if (arg == "--block-size" && i + 1 < argc) block_size = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        else if (arg == "--verify") verify = true;
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> val_dist(0, value_range - 1);

    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) arr[i] = val_dist(rng);
    auto queries = generate_queries(n, q, max_len, seed + 1);
    auto arr_comp = compress_values(arr);
    
    // Compute the mathematical trick on the CPU first
    auto prev_array = compute_prev_array(arr_comp);

    int *d_prev = nullptr, *d_out = nullptr;
    Query* d_queries = nullptr;
    std::vector<int> out(q, 0);

    size_t prev_bytes = sizeof(int) * prev_array.size();
    size_t q_bytes = sizeof(Query) * queries.size();
    size_t out_bytes = sizeof(int) * out.size();

    cudaEvent_t h2d_start, h2d_end, k_start, k_end, d2h_start, d2h_end;
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_end));
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_end));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_end));

    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMalloc(&d_prev, prev_bytes));
    CUDA_CHECK(cudaMalloc(&d_queries, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    
    // We only need to copy the `prev` array now, not the original array!
    CUDA_CHECK(cudaMemcpy(d_prev, prev_array.data(), prev_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_end));
    CUDA_CHECK(cudaEventSynchronize(h2d_end));

    // We need 32 threads for EVERY query
    int total_threads = q * 32; 
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    CUDA_CHECK(cudaEventRecord(k_start));
    distinct_warp_kernel<<<grid_size, block_size>>>(d_prev, d_queries, q, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_end));
    CUDA_CHECK(cudaEventSynchronize(k_end));

    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_end));
    CUDA_CHECK(cudaEventSynchronize(d2h_end));

    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k_start, k_end));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_end));

    std::cout << "method=cuda_bruteforce"
              << ",n=" << n
              << ",q=" << q
              << ",h2d_ms=" << h2d_ms
              << ",kernel_ms=" << kernel_ms
              << ",d2h_ms=" << d2h_ms
              << ",total_ms=" << (h2d_ms + kernel_ms + d2h_ms)
              << ",avg_query_us=" << ((h2d_ms + kernel_ms + d2h_ms) * 1e3 / std::max(1, q))
              << "\n";

    if (verify && q <= 2000 && n <= 200000) {
        auto chk = serial_bruteforce_distinct(arr_comp, queries);
        bool ok = (chk == out);
        std::cout << "verify=" << (ok ? "PASS" : "FAIL") << "\n";
        if (!ok) return 2;
    }

    uint64_t checksum = 0;
    for (int x : out) checksum = checksum * 1469598103934665603ULL + static_cast<uint64_t>(x + 11);
    std::cout << "checksum=" << checksum << "\n";

    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_out));

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_end));
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_end));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_end));

    return 0;
}