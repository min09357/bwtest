// randread_bw.cpp
// Measures random-access read bandwidth (GB/s) over a 1GB-hugepage region.
// Uses HPCC RandomAccess POLY LFSR addressing with UNROLL independent streams
// per thread to maximise MLP (multiple outstanding cacheline loads).
//
// Usage: ./randread_bw [ncores] [iters_per_thread] [hugepages_1gb] [lines_per_access] [mode]
//        e.g.: ./randread_bw 16 100000000 4 4 0   (default: ncores=16, iters=100000000, hugepages=4, lines=1, mode=0)
// Note: hugepages_1gb must be a power of 2 (LFSR mask addressing requirement).
//       lines_per_access ∈ {1,2,4,8,16}; each random access fetches that many cachelines
//       starting from one randomly chosen address; iters_per_thread counts total 64B
//       cachelines fetched per thread; address-generation count = iters_per_thread / lines_per_access.
//       mode selects how those lines_per_access cachelines are placed (default 0):
//         0 = consecutive : addr, addr+64, addr+128, ... (block-aligned to lines_per_access*64B)
//         1 = samebank    : addr XOR'd with access_masks.h's per-column-bit masks, so all
//                           lines land in the same channel/rank/bank-group/bank/row at
//                           adjacent columns (row-buffer-hit pattern). The DRAM mapping is
//                           picked at runtime via the ACCESS_MAP env var (unset -> the
//                           compiled-in default ACCESS_DEFAULT_MAP == config.ADDR_MAP); all
//                           registered maps are in access_masks.h (gen_access_masks.py).
//                           Requires lines_per_access <= 2**num_col_bits for the chosen map.

#include <barrier>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <immintrin.h>
#include <sys/mman.h>
#include "bw_width.h"
#include "access_masks.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

// HPCC RandomAccess LFSR constants  (from RandomAccess.h)
static constexpr uint64_t POLY   = 0x0000000000000007ULL;
static constexpr int64_t  PERIOD = 1317624576693539401LL;

static constexpr uint64_t GB = 1ULL << 30;

static constexpr int UNROLL = 16;                 // independent RNG streams per thread

// Skip-ahead: returns the n-th value of the HPCC POLY LFSR sequence.
// Copied from hpcc/RandomAccess/utility.c; adapted to use uint64_t/int64_t.
static uint64_t hpcc_starts(int64_t n) {
    uint64_t m2[64], temp, ran;

    while (n < 0)      n += PERIOD;
    while (n > PERIOD) n -= PERIOD;
    if (n == 0) return 0x1ULL;

    temp = 0x1ULL;
    for (int i = 0; i < 64; i++) {
        m2[i] = temp;
        temp = (temp << 1) ^ (static_cast<int64_t>(temp) < 0 ? POLY : 0ULL);
        temp = (temp << 1) ^ (static_cast<int64_t>(temp) < 0 ? POLY : 0ULL);
    }

    int i;
    for (i = 62; i >= 0; i--)
        if ((n >> i) & 1) break;

    ran = 0x2ULL;
    while (i > 0) {
        temp = 0;
        for (int j = 0; j < 64; j++)
            if ((ran >> j) & 1) temp ^= m2[j];
        ran = temp;
        i--;
        if ((n >> i) & 1)
            ran = (ran << 1) ^ (static_cast<int64_t>(ran) < 0 ? POLY : 0ULL);
    }
    return ran;
}

// Thread 0 stores the measurement start time immediately after the barrier.
static std::atomic<std::chrono::steady_clock::time_point> g_t_start;

struct ThreadArg {
    int      tid;
    uint8_t *base;
    int64_t  L;          // address-generation iterations per stream (= iters_per_thread / UNROLL / LINES)
    uint64_t mask;       // block index mask: MODE 0 -> nblk-1 (nblk = ncl/LINES); MODE 1 -> ncl-1
    const uint64_t *col_masks;  // MODE 1 only: col_masks[d] steps the column by d (same bank/row); nullptr for MODE 0
    uint64_t sink_out;   // accumulated read data; returned to main to prevent DCE
};

// LINES is a compile-time constant; the inner k-loop is fully unrolled by -O3,
// leaving no loop-branch overhead in the hot path.
// MODE selects the access pattern: 0 = consecutive cachelines, 1 = same-bank/row
// adjacent columns via XOR masks (access_masks.h). Templating on MODE keeps the
// unused branch out of the compiled hot path entirely.
template <int LINES, int MODE>
static void thread_func(ThreadArg &a, std::barrier<> &bar) {
    // CPU affinity is set externally by numactl -C / -m before process launch.

    // Initialise UNROLL independent LFSR streams via skip-ahead.
    // Global stream g = tid*UNROLL + u; each advances L steps → start at g*L.
    uint64_t ran[UNROLL];
    for (int u = 0; u < UNROLL; u++)
        ran[u] = hpcc_starts(static_cast<int64_t>(a.tid * UNROLL + u) * a.L);

    // Wait until all threads are ready, then start simultaneously.
    bar.arrive_and_wait();

    if (a.tid == 0)
        g_t_start.store(std::chrono::steady_clock::now(),
                        std::memory_order_release);

    // --- Measurement loop ---
    const uint8_t  *base      = a.base;
    const int64_t   L         = a.L;
    const uint64_t  mask      = a.mask;
    const uint64_t *col_masks = a.col_masks;   // MODE 1 only

#if BW_SIMD_WIDTH == 512
    __m512i sink = _mm512_setzero_si512();
#elif BW_SIMD_WIDTH == 256
    __m256i sink = _mm256_setzero_si256();
#else
    uint64_t sink = 0;
#endif

    for (int64_t n = 0; n < L; n++) {
        for (int u = 0; u < UNROLL; u++) {
            // Galois LFSR 1 step.  Logic: ran = (ran<<1) ^ (ran<0 ? POLY : 0)
            // branchless: bit63 → mask (0 or all-ones) → select POLY  (no branch/cmov)
            uint64_t mv = static_cast<uint64_t>(0) - (ran[u] >> 63);
            ran[u] = (ran[u] << 1) ^ (mv & POLY);

            size_t blk = ran[u] & mask;
            size_t off;
            if constexpr (MODE == 0) {
                off = blk * (static_cast<size_t>(LINES) * 64);  // block index in [0, nblk)
            } else {
                off = blk * 64;  // blk is a cacheline index in [0, ncl)
            }

            for (int k = 0; k < LINES; k++) {
                size_t off_k;
                if constexpr (MODE == 0) {
                    off_k = off + static_cast<size_t>(k) * 64;       // consecutive
                } else {
                    off_k = off ^ col_masks[k];                      // same bank/row, adjacent column
                }
                const uint8_t *p = base + off_k;
#if BW_SIMD_WIDTH == 512
                __m512i v = _mm512_load_si512(
                    reinterpret_cast<const __m512i *>(p));
                sink = _mm512_xor_si512(sink, v);
#elif BW_SIMD_WIDTH == 256
                __m256i v = _mm256_load_si256(
                    reinterpret_cast<const __m256i *>(p));
                sink = _mm256_xor_si256(sink, v);
#else
                uint64_t v;
                std::memcpy(&v, p, sizeof(v));
                sink ^= v;
#endif
            }
        }
    }

    // Reduce sink to 64-bit for the caller
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

template <int LINES, int MODE>
static void run_measurement(std::vector<ThreadArg> &args, std::barrier<> &bar, int ncores) {
    std::vector<std::jthread> threads;
    threads.reserve(ncores);
    for (int i = 0; i < ncores; i++)
        threads.emplace_back([&args, &bar, i] { thread_func<LINES, MODE>(args[i], bar); });
    threads.clear();  // jthread destructor joins each thread
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
    int     lines_per_access = (argc > 4) ? std::atoi(argv[4]) : 1;
    int     mode             = (argc > 5) ? std::atoi(argv[5]) : 0;

    if (ncores < 1) {
        std::fprintf(stderr, "error: ncores must be >= 1\n");
        return 1;
    }
    if (hugepages_1gb < 1) {
        std::fprintf(stderr, "error: hugepages_1gb must be >= 1\n");
        return 1;
    }
    // LFSR mask addressing requires NCL (= hugepages_1gb * GB / 64) to be a power of 2,
    // which holds iff hugepages_1gb is a power of 2.
    if ((hugepages_1gb & (hugepages_1gb - 1)) != 0) {
        std::fprintf(stderr,
            "error: hugepages_1gb=%d must be a power of 2 (1,2,4,8,...) for LFSR mask addressing\n",
            hugepages_1gb);
        return 1;
    }
    if (lines_per_access != 1 && lines_per_access != 2 && lines_per_access != 4 &&
        lines_per_access != 8 && lines_per_access != 16) {
        std::fprintf(stderr,
            "error: lines_per_access=%d must be one of {1,2,4,8,16}\n",
            lines_per_access);
        return 1;
    }
    if (mode != 0 && mode != 1) {
        std::fprintf(stderr, "error: mode=%d must be 0 (consecutive) or 1 (samebank)\n", mode);
        return 1;
    }

    // MODE 1 selects a DRAM mapping (compiled into access_masks.h for every
    // registered system) at runtime via the ACCESS_MAP env var; unset -> default.
    const AccessMap *sel = nullptr;
    if (mode == 1) {
        const char *want = std::getenv("ACCESS_MAP");
        if (!want || !*want) want = ACCESS_DEFAULT_MAP;
        for (int i = 0; i < ACCESS_NUM_MAPS; i++)
            if (std::strcmp(ACCESS_MAPS[i].name, want) == 0) { sel = &ACCESS_MAPS[i]; break; }
        if (!sel) {
            std::fprintf(stderr, "error: ACCESS_MAP=%s unknown; available:", want);
            for (int i = 0; i < ACCESS_NUM_MAPS; i++)
                std::fprintf(stderr, " %s", ACCESS_MAPS[i].name);
            std::fprintf(stderr, "\n");
            return 1;
        }
        if (lines_per_access > (1 << sel->num_col_bits)) {
            std::fprintf(stderr,
                "error: lines_per_access=%d exceeds max %d supported by ADDR_MAP=%s "
                "(num_col_bits=%d); pick a mapping with more column bits (ACCESS_MAP=...) "
                "or reduce lines_per_access\n",
                lines_per_access, 1 << sel->num_col_bits, sel->name, sel->num_col_bits);
            return 1;
        }
    }

    uint64_t total = static_cast<uint64_t>(hugepages_1gb) * GB;
    uint64_t ncl   = total / 64;    // number of cachelines
    uint64_t nblk  = ncl / static_cast<uint64_t>(lines_per_access);  // aligned blocks (MODE 0)
    uint64_t mask  = (mode == 0) ? (nblk - 1) : (ncl - 1);

    // MODE 1: precompute the XOR mask that steps the column by d, for every
    // d in [0, lines_per_access). col_masks[0] is always 0 (no shift).
    uint64_t col_masks[16] = {0};
    if (mode == 1) {
        for (int d = 0; d < lines_per_access; d++) {
            uint64_t m = 0;
            for (int b = 0; b < sel->num_col_bits; b++)
                if ((d >> b) & 1) m ^= sel->col_step_mask[b];
            col_masks[d] = m;
        }
    }

    // Round to UNROLL * lines_per_access so each stream does L complete block accesses.
    int64_t round = static_cast<int64_t>(UNROLL) * lines_per_access;
    iters_per_thread = (iters_per_thread / round) * round;
    int64_t L = iters_per_thread / UNROLL / lines_per_access;  // address-gen iters per stream

    std::printf("ncores=%d  iters/thread=%lld  streams=%d  region=%d GB  lines/access=%d (%d B)  "
        "mode=%d (%s)%s%s  simd=%s\n",
        ncores, (long long)iters_per_thread,
        ncores * UNROLL, hugepages_1gb, lines_per_access, lines_per_access * 64,
        mode, mode == 0 ? "consecutive" : "samebank",
        mode == 1 ? "  addr_map=" : "", mode == 1 ? sel->name : "",
        BW_SIMD_WIDTH_STR);

    // Allocate hugepages_1gb × 1GB hugepages
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

    // Warm up: sequential write to commit physical pages before read-only measurement
    std::printf("Warming up %d GB ... ", hugepages_1gb);
    std::fflush(stdout);
    std::memset(base_ptr, 0x5A, total);
    std::printf("done\n");

    // Per-thread arguments
    std::vector<ThreadArg> args(ncores);
    for (int i = 0; i < ncores; i++)
        args[i] = { i, static_cast<uint8_t *>(base_ptr), L, mask, col_masks, 0 };

    std::barrier<> bar(ncores);

#define BW_DISPATCH(N) do {                                        \
        if (mode == 0) run_measurement<N, 0>(args, bar, ncores);   \
        else            run_measurement<N, 1>(args, bar, ncores);  \
    } while (0)

    switch (lines_per_access) {
        case  1: BW_DISPATCH( 1); break;
        case  2: BW_DISPATCH( 2); break;
        case  4: BW_DISPATCH( 4); break;
        case  8: BW_DISPATCH( 8); break;
        case 16: BW_DISPATCH(16); break;
    }
#undef BW_DISPATCH

    auto t_end   = std::chrono::steady_clock::now();
    auto t_start = g_t_start.load(std::memory_order_acquire);
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // XOR all sinks so the compiler cannot elide the loads
    uint64_t total_sink = 0;
    for (int i = 0; i < ncores; i++) total_sink ^= args[i].sink_out;

    // total_bytes: N-independent — iters counts cachelines, not accesses
    double total_bytes = static_cast<double>(ncores) * iters_per_thread * 64.0;
    // total_iters: address-generation count (each access → lines_per_access cachelines)
    double total_iters = static_cast<double>(ncores) * iters_per_thread / lines_per_access;
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
