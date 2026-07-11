// latency_bw.cpp
// Measures single-access DRAM latency for two address relationships, using
// the full DRAM address decode (decode_masks.h, from address_mapping.py):
//
//   1. same channel/rank/bank-group/bank, different row  (row-buffer conflict)
//   2. same channel/rank, different bank-group            (independent banks)
//
// For each category, `pairs` (A,B) address pairs are drawn from a huge-page
// pool; both addresses are flushed from cache, then accessed back-to-back
// (A, then B) and the elapsed TSC ticks are recorded (DRAMA-style timing).
// Physical addresses are obtained via /proc/self/pagemap so pairs may span
// multiple 1GB hugepages correctly — this requires root (PFNs read as 0 for
// unprivileged processes since Linux 4.0).
//
// Usage: ./latency_bw [hugepages_1gb] [pairs] [reps] [cpu] [seed]
//        e.g.: sudo ./latency_bw 8 50 1000000 0
// ACCESS_MAP env var selects the DRAM mapping (see decode_masks.h); default
// is the compiled-in DECODE_DEFAULT_MAP (== config.ADDR_MAP).

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "decode_masks.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

// ─────────────────────────────────────────────────────────────────────────
// decode(): C++ port of address_mapping.py's decode()/_parity()/_bits_value()
// ─────────────────────────────────────────────────────────────────────────

static inline int parity64(uint64_t x) {
    return __builtin_popcountll(x) & 1;
}

// Gather the bits set in `mask` from `addr`, packed LSB-first in ascending
// physical-bit order — mirrors address_mapping._bits_value().
static inline uint64_t bits_gather(uint64_t addr, uint64_t mask) {
    uint64_t value = 0;
    int out_bit = 0;
    for (int bit = 0; bit < 64 && mask; bit++, mask >>= 1) {
        if (mask & 1) {
            value |= ((addr >> bit) & 1ULL) << out_bit;
            out_bit++;
        }
    }
    return value;
}

struct Decoded {
    int channel = 0, rank = 0, bank_group = 0, bank = 0;
    uint64_t row = 0, col = 0;
};

static Decoded decode_addr(const DecodeMap &m, uint64_t addr) {
    Decoded d;
    for (int i = 0; i < m.n_channel; i++)    d.channel    |= parity64(addr & m.channel[i])    << i;
    for (int i = 0; i < m.n_rank; i++)       d.rank       |= parity64(addr & m.rank[i])       << i;
    for (int i = 0; i < m.n_bank_group; i++) d.bank_group |= parity64(addr & m.bank_group[i]) << i;
    for (int i = 0; i < m.n_bank; i++)       d.bank       |= parity64(addr & m.bank[i])       << i;
    d.row = bits_gather(addr, m.row_mask);
    d.col = bits_gather(addr, m.col_mask);
    return d;
}

// Self-check: verify decode_addr() reproduces every DECODE_SELFCHECKS vector
// (computed by the Python reference, address_mapping.decode()) for `m`.
static void self_check(const DecodeMap &m) {
    int checked = 0;
    for (int i = 0; i < DECODE_NUM_SELFCHECKS; i++) {
        const DecodeSelfCheck &tc = DECODE_SELFCHECKS[i];
        if (std::strcmp(tc.name, m.name) != 0) continue;
        Decoded d = decode_addr(m, tc.addr);
        if (d.channel != tc.channel || d.rank != tc.rank || d.bank_group != tc.bank_group ||
            d.bank != tc.bank || d.row != tc.row || d.col != tc.col) {
            std::fprintf(stderr,
                "self-check FAILED: map=%s addr=0x%llx\n"
                "  expected: ch=%d rk=%d bg=%d bank=%d row=0x%llx col=0x%llx\n"
                "  got:      ch=%d rk=%d bg=%d bank=%d row=0x%llx col=0x%llx\n",
                m.name, (unsigned long long)tc.addr,
                tc.channel, tc.rank, tc.bank_group, tc.bank,
                (unsigned long long)tc.row, (unsigned long long)tc.col,
                d.channel, d.rank, d.bank_group, d.bank,
                (unsigned long long)d.row, (unsigned long long)d.col);
            std::exit(1);
        }
        checked++;
    }
    if (checked == 0) {
        std::fprintf(stderr, "error: no self-check vectors found for map=%s\n", m.name);
        std::exit(1);
    }
    std::printf("self-check passed (%d vectors) for map=%s\n", checked, m.name);
}

// D = a ^ b. Each selector field is a list of XOR-parity masks; since parity
// is linear under XOR (parity(x^y) == parity(x)^parity(y)), the field values
// of a and b are equal iff parity(D & mask_i) == 0 for every mask in the field.
static inline bool fields_equal(const uint64_t *masks, int n, uint64_t D) {
    for (int i = 0; i < n; i++)
        if (parity64(D & masks[i]) != 0) return false;
    return true;
}

static inline bool same_bank_diff_row(const DecodeMap &m, uint64_t a, uint64_t b) {
    uint64_t D = a ^ b;
    return fields_equal(m.channel, m.n_channel, D) &&
           fields_equal(m.rank, m.n_rank, D) &&
           fields_equal(m.bank_group, m.n_bank_group, D) &&
           fields_equal(m.bank, m.n_bank, D) &&
           (D & m.row_mask) != 0;
}

static inline bool diff_bank_group(const DecodeMap &m, uint64_t a, uint64_t b) {
    uint64_t D = a ^ b;
    return fields_equal(m.channel, m.n_channel, D) &&
           fields_equal(m.rank, m.n_rank, D) &&
           !fields_equal(m.bank_group, m.n_bank_group, D);
}

// ─────────────────────────────────────────────────────────────────────────
// /proc/self/pagemap: virtual -> physical address translation
// ─────────────────────────────────────────────────────────────────────────

struct Pagemap {
    int fd;
    long page_size;

    Pagemap() : page_size(sysconf(_SC_PAGESIZE)) {
        fd = open("/proc/self/pagemap", O_RDONLY);
        if (fd < 0) {
            std::perror("open /proc/self/pagemap");
            std::exit(1);
        }
    }
    ~Pagemap() { if (fd >= 0) close(fd); }

    uint64_t virt_to_phys(uint64_t vaddr) const {
        uint64_t vpn = vaddr / static_cast<uint64_t>(page_size);
        uint64_t entry;
        ssize_t n = pread(fd, &entry, sizeof(entry),
                          static_cast<off_t>(vpn * sizeof(entry)));
        if (n != static_cast<ssize_t>(sizeof(entry))) {
            std::fprintf(stderr, "error: pread /proc/self/pagemap failed for vaddr=0x%llx\n",
                        (unsigned long long)vaddr);
            std::exit(1);
        }
        bool present = (entry >> 63) & 1ULL;
        if (!present) {
            std::fprintf(stderr, "error: page not present for vaddr=0x%llx\n",
                        (unsigned long long)vaddr);
            std::exit(1);
        }
        uint64_t pfn = entry & ((1ULL << 55) - 1);
        if (pfn == 0) {
            std::fprintf(stderr,
                "error: pagemap PFN==0 for vaddr=0x%llx — re-run with sudo/root "
                "(kernel masks PFNs for unprivileged processes since Linux 4.0)\n",
                (unsigned long long)vaddr);
            std::exit(1);
        }
        return pfn * static_cast<uint64_t>(page_size) + (vaddr % static_cast<uint64_t>(page_size));
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Latency measurement kernel
// ─────────────────────────────────────────────────────────────────────────

static uint64_t g_dce_sink = 0;  // XOR accumulator; printed at the end to prevent DCE

// Flush A and B, then time a back-to-back A->B read (DRAMA-style row-conflict
// timing). Returns elapsed TSC ticks.
static inline uint64_t measure_once(const void *A, const void *B) {
    _mm_clflush(A);
    _mm_clflush(B);
    _mm_mfence();

    unsigned aux;
    _mm_lfence();
    uint64_t t0 = __rdtscp(&aux);
    _mm_lfence();

    uint64_t v = *static_cast<const volatile uint64_t *>(A);
    v ^= *static_cast<const volatile uint64_t *>(B);

    _mm_lfence();
    uint64_t t1 = __rdtscp(&aux);
    _mm_lfence();

    g_dce_sink ^= v;
    return t1 - t0;
}

struct PairStats {
    uint64_t median_cyc;
    uint64_t min_cyc;
};

static PairStats measure_pair(const void *A, const void *B, int64_t reps,
                              std::vector<uint64_t> &buf) {
    if (static_cast<int64_t>(buf.size()) != reps) buf.resize(reps);

    for (int i = 0; i < 100; i++) measure_once(A, B);  // warm-up, discarded

    uint64_t mn = UINT64_MAX;
    for (int64_t i = 0; i < reps; i++) {
        uint64_t s = measure_once(A, B);
        buf[i] = s;
        if (s < mn) mn = s;
    }
    int64_t mid = reps / 2;
    std::nth_element(buf.begin(), buf.begin() + mid, buf.begin() + reps);
    return { buf[mid], mn };
}

static double calibrate_tsc_hz() {
    struct timespec ts0, ts1;
    unsigned aux;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    uint64_t c0 = __rdtscp(&aux);

    struct timespec target = ts0;
    target.tv_nsec += 50'000'000L;
    if (target.tv_nsec >= 1'000'000'000L) {
        target.tv_sec++;
        target.tv_nsec -= 1'000'000'000L;
    }
    struct timespec now;
    do {
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while (now.tv_sec < target.tv_sec ||
             (now.tv_sec == target.tv_sec && now.tv_nsec < target.tv_nsec));

    uint64_t c1 = __rdtscp(&aux);
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    double dt = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    return static_cast<double>(c1 - c0) / dt;
}

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        std::perror("sched_setaffinity");
}

// ─────────────────────────────────────────────────────────────────────────
// Address sampling
// ─────────────────────────────────────────────────────────────────────────

struct AddrInfo {
    uint64_t vaddr;
    uint64_t phys;
    Decoded  d;
};

static AddrInfo make_addr_info(const Pagemap &pm, const DecodeMap &m,
                               uint8_t *base, uint64_t voffset) {
    AddrInfo info;
    info.vaddr = reinterpret_cast<uint64_t>(base + voffset);
    info.phys  = pm.virt_to_phys(info.vaddr);
    info.d     = decode_addr(m, info.phys);
    return info;
}

int main(int argc, char *argv[]) {
    int     hugepages_1gb = (argc > 1) ? std::atoi(argv[1]) : DECODE_DEFAULT_HUGEPAGES;
    int     pairs         = (argc > 2) ? std::atoi(argv[2]) : 50;
    int64_t reps          = (argc > 3) ? std::atoll(argv[3]) : 1'000'000LL;
    int     cpu           = (argc > 4) ? std::atoi(argv[4]) : 0;
    uint64_t seed         = (argc > 5) ? std::strtoull(argv[5], nullptr, 0)
                                       : static_cast<uint64_t>(
                                             std::chrono::steady_clock::now().time_since_epoch().count());

    if (hugepages_1gb < 1) {
        std::fprintf(stderr, "error: hugepages_1gb must be >= 1\n");
        return 1;
    }
    if (pairs < 1) {
        std::fprintf(stderr, "error: pairs must be >= 1\n");
        return 1;
    }
    if (reps < 1) {
        std::fprintf(stderr, "error: reps must be >= 1\n");
        return 1;
    }

    pin_to_cpu(cpu);

    const char *want = std::getenv("ACCESS_MAP");
    if (!want || !*want) want = DECODE_DEFAULT_MAP;
    const DecodeMap *sel = nullptr;
    for (int i = 0; i < DECODE_NUM_MAPS; i++)
        if (std::strcmp(DECODE_MAPS[i].name, want) == 0) { sel = &DECODE_MAPS[i]; break; }
    if (!sel) {
        std::fprintf(stderr, "error: ACCESS_MAP=%s unknown; available:", want);
        for (int i = 0; i < DECODE_NUM_MAPS; i++)
            std::fprintf(stderr, " %s", DECODE_MAPS[i].name);
        std::fprintf(stderr, "\n");
        return 1;
    }

    self_check(*sel);

    if (geteuid() != 0) {
        std::fprintf(stderr,
            "warning: not running as root — /proc/self/pagemap PFNs will read as 0; "
            "re-run with sudo\n");
    }

    uint64_t total = static_cast<uint64_t>(hugepages_1gb) * (1ULL << 30);

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

    std::printf("Warming up %d GB ... ", hugepages_1gb);
    std::fflush(stdout);
    std::memset(base_ptr, 0x5A, total);
    std::printf("done\n");

    Pagemap pm;
    double tsc_hz = calibrate_tsc_hz();

    std::printf("map=%s hugepages=%d pairs=%d reps=%lld cpu=%d seed=%llu tsc_hz=%.3f MHz\n\n",
        sel->name, hugepages_1gb, pairs, (long long)reps, cpu,
        (unsigned long long)seed, tsc_hz / 1e6);

    std::mt19937_64 rng(seed);
    uint8_t *base = static_cast<uint8_t *>(base_ptr);
    uint64_t ncl = total / 64;

    std::vector<uint64_t> sample_buf;

    // Runs one category (same_bank_mode selects the matching predicate),
    // printing each pair's result as soon as it is measured, then a summary
    // over all `pairs` pair-medians. Returns the summary mean (ns).
    auto run_category = [&](const char *label, bool same_bank_mode) -> double {
        std::printf("=== %s (%d pairs) ===\n", label, pairs);
        std::set<std::pair<uint64_t, uint64_t>> seen;
        std::vector<double> pair_medians_ns;
        pair_medians_ns.reserve(pairs);

        int64_t budget = static_cast<int64_t>(pairs) * 2'000'000LL;
        int found = 0;
        while (found < pairs && budget > 0) {
            AddrInfo A = make_addr_info(pm, *sel, base, (rng() % ncl) * 64);
            AddrInfo B{};
            bool got = false;
            for (int t = 0; t < 200000 && budget > 0; t++, budget--) {
                AddrInfo cand = make_addr_info(pm, *sel, base, (rng() % ncl) * 64);
                if (cand.vaddr == A.vaddr) continue;
                bool match = same_bank_mode ? same_bank_diff_row(*sel, A.phys, cand.phys)
                                            : diff_bank_group(*sel, A.phys, cand.phys);
                if (!match) continue;
                auto key = std::minmax(A.vaddr, cand.vaddr);
                if (seen.count(key)) continue;
                seen.insert(key);
                B = cand;
                got = true;
                break;
            }
            if (!got) continue;

            PairStats st = measure_pair(reinterpret_cast<const void *>(A.vaddr),
                                        reinterpret_cast<const void *>(B.vaddr),
                                        reps, sample_buf);
            double median_ns = static_cast<double>(st.median_cyc) / tsc_hz * 1e9;
            double min_ns    = static_cast<double>(st.min_cyc) / tsc_hz * 1e9;

            if (same_bank_mode) {
                std::printf(
                    "  [%2d] A=0x%011llx B=0x%011llx  ch=%d rk=%d bg=%d bank=%d  rowA=0x%llx rowB=0x%llx  "
                    "median=%llu cyc (%.1f ns)  min=%llu cyc (%.1f ns)\n",
                    found, (unsigned long long)A.phys, (unsigned long long)B.phys,
                    A.d.channel, A.d.rank, A.d.bank_group, A.d.bank,
                    (unsigned long long)A.d.row, (unsigned long long)B.d.row,
                    (unsigned long long)st.median_cyc, median_ns,
                    (unsigned long long)st.min_cyc, min_ns);
            } else {
                std::printf(
                    "  [%2d] A=0x%011llx B=0x%011llx  ch=%d rk=%d bgA=%d bgB=%d  "
                    "median=%llu cyc (%.1f ns)  min=%llu cyc (%.1f ns)\n",
                    found, (unsigned long long)A.phys, (unsigned long long)B.phys,
                    A.d.channel, A.d.rank, A.d.bank_group, B.d.bank_group,
                    (unsigned long long)st.median_cyc, median_ns,
                    (unsigned long long)st.min_cyc, min_ns);
            }
            std::fflush(stdout);

            pair_medians_ns.push_back(median_ns);
            found++;
        }

        if (found < pairs) {
            std::fprintf(stderr,
                "error: could not find %d %s pairs (only found %d) — retry budget exhausted\n",
                pairs, label, found);
            std::exit(1);
        }

        double sum = 0;
        for (double v : pair_medians_ns) sum += v;
        double mean = sum / pair_medians_ns.size();
        double mn = *std::min_element(pair_medians_ns.begin(), pair_medians_ns.end());
        double mx = *std::max_element(pair_medians_ns.begin(), pair_medians_ns.end());
        double var = 0;
        for (double v : pair_medians_ns) var += (v - mean) * (v - mean);
        double stddev = std::sqrt(var / pair_medians_ns.size());

        std::printf("  summary  mean=%.2f ns  min=%.2f ns  max=%.2f ns  stddev=%.2f ns\n\n",
                    mean, mn, mx, stddev);
        return mean;
    };

    double mean_samebank = run_category("same-bank / different-row", true);
    double mean_diffbg   = run_category("different-bankgroup", false);

    std::printf("same bank: %.2f ns, diff bg: %.2f ns\n", mean_samebank, mean_diffbg);
    std::printf("ratio (samebank / diffbg) : %.3f\n", mean_samebank / mean_diffbg);
    std::printf("Checksum : %016llx\n", (unsigned long long)g_dce_sink);

    munmap(base_ptr, total);
    return 0;
}
