// stream_bw.cpp
// Measures sequential read bandwidth (GB/s) over a 1GB-hugepage region.
// Each thread scans its own contiguous sub-region sequentially (cache-line stride),
// wrapping around until iters_per_thread accesses are done.
// Companion to randread_bw.cpp — same harness, sequential vs random access pattern.
//
// Usage: ./stream_bw [ncores] [iters_per_thread] [hugepages_1gb]
//        e.g.: ./stream_bw 16 100000000 4   (default: ncores=16, iters=100000000, hugepages=4)

#include <algorithm>
#include <barrier>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <sys/mman.h>
#include "bw_width.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

static constexpr uint64_t GB = 1ULL << 30;

// Thread 0 stores the measurement start time immediately after the barrier.
static std::atomic<std::chrono::steady_clock::time_point> g_t_start;

struct ThreadArg {
    int      tid;
    int      ncores;
    uint8_t *base;
    int64_t  iters_per_thread;
    uint64_t ncl;        // total number of 64B cachelines in region
    uint64_t sink_out;
};

static void thread_func(ThreadArg &a, std::barrier<> &bar) {
    // CPU affinity is set externally by numactl -C / -m before process launch.

    // Divide the region into contiguous per-thread sub-ranges [cl_begin, cl_end).
    uint64_t cl_per_thread = a.ncl / static_cast<uint64_t>(a.ncores);
    uint64_t cl_begin      = static_cast<uint64_t>(a.tid) * cl_per_thread;
    uint64_t cl_end        = (a.tid == a.ncores - 1) ? a.ncl : cl_begin + cl_per_thread;

    // Wait until all threads are ready, then start simultaneously.
    bar.arrive_and_wait();

    if (a.tid == 0)
        g_t_start.store(std::chrono::steady_clock::now(),
                        std::memory_order_release);

    // --- Measurement loop ---
    // Outer loop handles wrap-around; inner loop is a branchless sequential scan
    // so the compiler can freely vectorize and unroll it.
    // One load per 64B cacheline regardless of SIMD width: touching any byte of
    // a line causes the full 64B DRAM transaction, and main() accounts each
    // iteration as 64B.  Multiple sub-line loads would inflate instruction count
    // without changing DRAM traffic and risk making the loop core-bound.
#if BW_SIMD_WIDTH == 512
    __m512i sink = _mm512_setzero_si512();
#elif BW_SIMD_WIDTH == 256
    __m256i sink = _mm256_setzero_si256();
#else
    uint64_t sink = 0;
#endif
    int64_t remaining = a.iters_per_thread;

    while (remaining > 0) {
        uint64_t run = std::min<uint64_t>(cl_end - cl_begin,
                                          static_cast<uint64_t>(remaining));
        const uint8_t *p = a.base + cl_begin * 64;
        for (uint64_t i = 0; i < run; i++) {
#if BW_SIMD_WIDTH == 512
            __m512i v = _mm512_load_si512(reinterpret_cast<const __m512i *>(p));
            sink = _mm512_xor_si512(sink, v);  // accumulate to prevent DCE
#elif BW_SIMD_WIDTH == 256
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(p));
            sink = _mm256_xor_si256(sink, v);  // accumulate to prevent DCE
#else
            uint64_t v;
            std::memcpy(&v, p, sizeof(v));
            sink ^= v;                         // accumulate to prevent DCE
#endif
            p += 64;
        }
        remaining -= static_cast<int64_t>(run);
    }

    // Reduce sink to 64-bit for the caller.
#if BW_SIMD_WIDTH == 512
    alignas(64) uint64_t buf[8];
    _mm512_store_si512(reinterpret_cast<__m512i *>(buf), sink);
    uint64_t s = 0;
    for (int i = 0; i < 8; i++) s ^= buf[i];
    a.sink_out = s;
#elif BW_SIMD_WIDTH == 256
    alignas(32) uint64_t buf[4];
    _mm256_store_si256(reinterpret_cast<__m256i *>(buf), sink);
    uint64_t s = 0;
    for (int i = 0; i < 4; i++) s ^= buf[i];
    a.sink_out = s;
#else
    a.sink_out = sink;
#endif
}

int main(int argc, char *argv[]) {
    // Probe mode: print the compiled SIMD width and exit (used by sweep_bw.py).
    if (argc > 1 && std::strcmp(argv[1], "--simd") == 0) {
        std::printf("%s\n", BW_SIMD_WIDTH_STR);
        return 0;
    }

    int     ncores           = (argc > 1) ? std::atoi(argv[1]) : 16;
    int64_t iters_per_thread = (argc > 2) ? std::atoll(argv[2]) : 100'000'000LL;
    int     hugepages_1gb    = (argc > 3) ? std::atoi(argv[3]) : 4;

    if (ncores < 1) {
        std::fprintf(stderr, "error: ncores must be >= 1\n");
        return 1;
    }
    if (hugepages_1gb < 1) {
        std::fprintf(stderr, "error: hugepages_1gb must be >= 1\n");
        return 1;
    }

    uint64_t total = static_cast<uint64_t>(hugepages_1gb) * GB;
    uint64_t ncl   = total / 64;   // number of 64B cachelines

    std::printf("ncores=%d  iters/thread=%lld  region=%d GB  pattern=sequential  simd=%s\n",
        ncores, (long long)iters_per_thread, hugepages_1gb, BW_SIMD_WIDTH_STR);

    // Allocate hugepages_1gb × 1GB hugepages.
    void *base_ptr = mmap(nullptr, total,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                          -1, 0);
    if (base_ptr == MAP_FAILED) {
        std::perror("mmap 1GB hugepages failed");
        std::fprintf(stderr,
            "hint: check 'grep HugePages /proc/meminfo'  (need HugePages_Free >= %d)\n",
            hugepages_1gb);
        return 1;
    }

    // Warm up: position-dependent fill commits physical pages and gives a
    // meaningful checksum (uniform memset would always yield checksum=0).
    std::printf("Warming up %d GB ... ", hugepages_1gb);
    std::fflush(stdout);
    uint64_t *p = static_cast<uint64_t *>(base_ptr);
    for (uint64_t k = 0; k < total / 8; k++)
        p[k] = k * 0x9E3779B97F4A7C15ULL;
    std::printf("done\n");

    // Per-thread arguments.
    std::vector<ThreadArg> args(ncores);
    for (int i = 0; i < ncores; i++)
        args[i] = { i, ncores, static_cast<uint8_t *>(base_ptr), iters_per_thread, ncl, 0 };

    std::barrier<> bar(ncores);

    // Launch threads.
    std::vector<std::jthread> threads;
    threads.reserve(ncores);
    for (int i = 0; i < ncores; i++)
        threads.emplace_back([&args, &bar, i] { thread_func(args[i], bar); });

    threads.clear();  // jthread destructor joins each thread

    auto t_end   = std::chrono::steady_clock::now();
    auto t_start = g_t_start.load(std::memory_order_acquire);
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // XOR all sinks so the compiler cannot elide the loads.
    uint64_t total_sink = 0;
    for (int i = 0; i < ncores; i++) total_sink ^= args[i].sink_out;

    double total_bytes = static_cast<double>(ncores) * static_cast<double>(iters_per_thread) * 64.0;
    double total_iters = static_cast<double>(ncores) * static_cast<double>(iters_per_thread);
    double bw_GBs  = total_bytes / elapsed / 1e9;
    double bw_GiBs = total_bytes / elapsed / static_cast<double>(1ULL << 30);
    double gups    = total_iters / elapsed / 1e9;

    std::printf("\n=== Results ===\n");
    std::printf("Elapsed        : %.3f s\n", elapsed);
    std::printf("Total accesses : %.3e  (%.3e bytes)\n", total_iters, total_bytes);
    std::printf("Bandwidth      : %.3f GB/s  (%.3f GiB/s)\n", bw_GBs, bw_GiBs);
    std::printf("GUPS           : %.4f\n", gups);
    std::printf("Checksum       : %016llx\n", (unsigned long long)total_sink);

    munmap(base_ptr, total);
    return 0;
}
