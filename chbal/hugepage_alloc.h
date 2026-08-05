// hugepage_alloc.h
// 1GB-hugepage region allocator with optional DRAM-channel balancing.
//
// Why: a channel selector function whose set bits are ALL >= 30 (e.g.
// cascade_4ch_1dpc_2rank_32gb's channel[1] = 0x1000000000, bit 36) is constant
// across a whole 1GB hugepage, so one hugepage only ever reaches half of the
// channels. A plain mmap(MAP_HUGETLB) takes whatever the kernel's free list
// hands out, so the benchmark region can end up entirely on one side of that
// bit and hammer only half the memory controllers.
//
// hp_alloc() fixes that by probing candidate hugepages through a hugetlb memfd,
// classifying each one by its page-constant channel functions (via
// /proc/self/pagemap), and then mapping a balanced, class-interleaved selection
// into one contiguous VA range: VA slot i gets a page of class (i % K). Pages
// that were probed but not selected are punched out of the memfd and go back to
// the pool.
//
// Balancing needs root (unprivileged processes read PFN 0 from pagemap since
// Linux 4.0). Every failure path falls back to the plain anonymous
// mmap(MAP_HUGETLB) the benchmarks used before, with a warning.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "decode_masks.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_1GB
#define MFD_HUGE_1GB (30U << 26)
#endif
#ifndef FALLOC_FL_KEEP_SIZE
#define FALLOC_FL_KEEP_SIZE 0x01
#endif
#ifndef FALLOC_FL_PUNCH_HOLE
#define FALLOC_FL_PUNCH_HOLE 0x02
#endif

static constexpr uint64_t HP_SIZE = 1ULL << 30;   // 1GB hugepage

// ─────────────────────────────────────────────────────────────────────────
// Address helpers (shared with latency_bw.cpp's decode path)
// ─────────────────────────────────────────────────────────────────────────

static inline int parity64(uint64_t x) {
    return __builtin_popcountll(x) & 1;
}

// System physical address (what /proc/self/pagemap reports) -> the address the
// memory controller actually decodes. The BIOS maps DRAM around the sub-4GB
// MMIO hole, so everything above it is shifted down by the hole size.
//
// This is arithmetic, not a bit permutation: the borrow it propagates cannot be
// expressed as an XOR mask, so it can neither be folded into the masks in
// address_mapping.py nor applied to an already-XORed address difference. Every
// address must go through here individually, before decoding.
static inline uint64_t phys_to_dram(uint64_t phys) {
    return phys >= DECODE_HOLE_END ? phys - DECODE_HOLE_SIZE : phys;
}

// ─────────────────────────────────────────────────────────────────────────
// /proc/self/pagemap: virtual -> physical address translation
// ─────────────────────────────────────────────────────────────────────────

struct Pagemap {
    int  fd;
    long page_size;

    Pagemap() : fd(open("/proc/self/pagemap", O_RDONLY)),
                page_size(sysconf(_SC_PAGESIZE)) {}
    ~Pagemap() { if (fd >= 0) close(fd); }

    Pagemap(const Pagemap &) = delete;
    Pagemap &operator=(const Pagemap &) = delete;

    bool ok() const { return fd >= 0; }

    // Returns 0 when the translation is unavailable (pagemap unreadable, page
    // not present, or PFN masked because we are not root). Callers decide
    // whether that is fatal.
    uint64_t virt_to_phys(uint64_t vaddr) const {
        if (fd < 0) return 0;
        uint64_t vpn = vaddr / static_cast<uint64_t>(page_size);
        uint64_t entry = 0;
        ssize_t n = pread(fd, &entry, sizeof(entry),
                          static_cast<off_t>(vpn * sizeof(entry)));
        if (n != static_cast<ssize_t>(sizeof(entry))) return 0;
        if (!((entry >> 63) & 1ULL)) return 0;             // not present
        uint64_t pfn = entry & ((1ULL << 55) - 1);
        if (pfn == 0) return 0;                            // masked (non-root)
        return pfn * static_cast<uint64_t>(page_size) +
               (vaddr % static_cast<uint64_t>(page_size));
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Channel classes: the channel selector functions that a 1GB page cannot vary
// ─────────────────────────────────────────────────────────────────────────

// A channel XOR function whose bits are all >= 30 has the same parity for every
// address inside a 1GB page, so it splits the hugepage pool into 2**n disjoint
// "classes"; every other channel function still toggles within the page and
// needs no help. Balancing means picking equally many pages of each class.
struct HpClassInfo {
    int      n = 0;                       // page-constant channel functions
    uint64_t mask[DECODE_MAX_BITS] = {0};

    int nclass() const { return 1 << n; }
    int classify(uint64_t dram_page_base) const {
        int c = 0;
        for (int i = 0; i < n; i++)
            c |= parity64(dram_page_base & mask[i]) << i;
        return c;
    }
};

static inline HpClassInfo hp_class_info(const DecodeMap &m) {
    HpClassInfo info;
    for (int i = 0; i < m.n_channel; i++)
        if (m.channel[i] != 0 && (m.channel[i] & (HP_SIZE - 1)) == 0)
            info.mask[info.n++] = m.channel[i];
    return info;
}

// ─────────────────────────────────────────────────────────────────────────
// Allocated region
// ─────────────────────────────────────────────────────────────────────────

struct HugeRegion {
    void    *base   = nullptr;   // 1GB-aligned, npages * 1GB contiguous VA
    uint64_t bytes  = 0;
    int      npages = 0;
    int      memfd  = -1;        // -1 => plain anonymous mmap fallback path
    std::vector<uint64_t> phys;  // physical base of VA slot i (0 = unknown)
    std::vector<int>      cls;   // channel class of VA slot i (-1 = unknown)
    bool     balanced = false;   // class counts came out exactly even
};

static inline void hp_free(HugeRegion &r) {
    if (r.base) munmap(r.base, r.bytes);
    if (r.memfd >= 0) close(r.memfd);
    r.base = nullptr;
    r.bytes = 0;
    r.memfd = -1;
}

// ─────────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────────

static inline int hp_memfd_create(const char *name, unsigned flags) {
    return static_cast<int>(syscall(SYS_memfd_create, name, flags));
}

static inline long hp_read_long(const char *path) {
    FILE *f = std::fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (std::fscanf(f, "%ld", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

// Free 1GB pages we can actually fault in: hugetlb reservations are global but
// the pages themselves come from the nodes the current mempolicy allows, so an
// over-eager probe would SIGBUS. Sum free_hugepages over the mempolicy's node
// mask (numactl -m N), falling back to the global counter.
static inline long hp_free_pages_available() {
    unsigned long nodemask[16] = {0};
    int mode = 0;
    long rc = syscall(SYS_get_mempolicy, &mode, nodemask,
                      static_cast<unsigned long>(sizeof(nodemask) * 8), 0, 0);
    bool have_mask = false;
    if (rc == 0 && (mode == 2 /*MPOL_BIND*/ || mode == 1 /*MPOL_PREFERRED*/ ||
                    mode == 3 /*MPOL_INTERLEAVE*/)) {
        for (size_t i = 0; i < sizeof(nodemask) / sizeof(nodemask[0]); i++)
            if (nodemask[i]) { have_mask = true; break; }
    }

    if (have_mask) {
        long total = 0;
        for (int node = 0; node < static_cast<int>(sizeof(nodemask) * 8); node++) {
            if (!((nodemask[node / 64] >> (node % 64)) & 1ULL)) continue;
            char path[160];
            std::snprintf(path, sizeof(path),
                "/sys/devices/system/node/node%d/hugepages/hugepages-1048576kB/free_hugepages",
                node);
            long v = hp_read_long(path);
            if (v > 0) total += v;
        }
        if (total > 0) return total;
    }
    return hp_read_long("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages");
}

// Per-class target counts: npages spread as evenly as K classes allow.
static inline std::vector<int> hp_quota(int npages, int K) {
    std::vector<int> q(K, npages / K);
    for (int i = 0; i < npages % K; i++) q[i]++;
    return q;
}

// Fault one page of `fd` at file page index `idx` through a scratch mapping and
// return its physical base (0 on failure). The page stays alive in the file
// after the scratch mapping goes away.
static inline uint64_t hp_probe_page(int fd, long idx, const Pagemap &pm) {
    void *p = mmap(nullptr, HP_SIZE, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_HUGETLB | MAP_HUGE_1GB,
                   fd, static_cast<off_t>(idx) * static_cast<off_t>(HP_SIZE));
    if (p == MAP_FAILED) return 0;
    *static_cast<volatile char *>(p) = 0;   // fault it in so pagemap has a PTE
    uint64_t phys = pm.virt_to_phys(reinterpret_cast<uint64_t>(p));
    munmap(p, HP_SIZE);
    return phys;
}

// Reserve npages*1GB of 1GB-aligned VA (PROT_NONE); slots are replaced by
// MAP_FIXED hugepage mappings afterwards.
static inline void *hp_reserve_va(int npages) {
    uint64_t span = (static_cast<uint64_t>(npages) + 1) * HP_SIZE;
    void *res = mmap(nullptr, span, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (res == MAP_FAILED) return nullptr;

    uint64_t raw   = reinterpret_cast<uint64_t>(res);
    uint64_t base  = (raw + HP_SIZE - 1) & ~(HP_SIZE - 1);
    uint64_t end   = base + static_cast<uint64_t>(npages) * HP_SIZE;
    if (base > raw) munmap(res, base - raw);
    if (raw + span > end) munmap(reinterpret_cast<void *>(end), raw + span - end);
    return reinterpret_cast<void *>(base);
}

// Balanced path. Returns a region with base == nullptr if it could not be done
// (caller then falls back to the plain mmap path).
static inline HugeRegion hp_alloc_balanced(int npages, const HpClassInfo &info,
                                    const Pagemap &pm) {
    HugeRegion r;
    const int K = info.nclass();

    int fd = hp_memfd_create("hugepages-1g", MFD_HUGETLB | MFD_HUGE_1GB);
    if (fd < 0) {
        std::fprintf(stderr, "warning: memfd_create(MFD_HUGETLB|MFD_HUGE_1GB) failed: %s\n",
                     std::strerror(errno));
        return r;
    }

    long cap = hp_free_pages_available();
    if (cap < npages) cap = npages;
    if (ftruncate(fd, static_cast<off_t>(cap) * static_cast<off_t>(HP_SIZE)) != 0) {
        std::fprintf(stderr, "warning: ftruncate hugetlb memfd failed: %s\n",
                     std::strerror(errno));
        close(fd);
        return r;
    }

    // Probe candidate pages one at a time, stopping as soon as every class has
    // its quota — faulting a 1GB page costs a 1GB kernel clear, so probing the
    // whole pool when the first few pages already balance would be wasteful.
    const std::vector<int> quota = hp_quota(npages, K);
    std::vector<std::vector<long>> by_class(K);
    long probed = 0;
    for (; probed < cap; probed++) {
        uint64_t phys = hp_probe_page(fd, probed, pm);
        if (phys == 0) break;
        by_class[info.classify(phys_to_dram(phys))].push_back(probed);

        bool done = true;
        for (int c = 0; c < K; c++)
            if (static_cast<int>(by_class[c].size()) < quota[c]) { done = false; break; }
        if (done) { probed++; break; }
    }

    // Pick pages class-round-robin so VA slot i holds class (i % K). If a class
    // ran dry, fill from whichever class still has the most pages left.
    std::vector<long> chosen;
    std::vector<int>  chosen_cls;
    std::vector<size_t> next(K, 0);
    bool balanced = true;
    for (int i = 0; i < npages; i++) {
        int c = i % K;
        if (next[c] >= by_class[c].size()) {
            int best = -1;
            size_t best_left = 0;
            for (int j = 0; j < K; j++) {
                size_t left = by_class[j].size() - next[j];
                if (left > best_left) { best_left = left; best = j; }
            }
            if (best < 0) break;          // out of pages entirely
            c = best;
            balanced = false;
        }
        chosen.push_back(by_class[c][next[c]++]);
        chosen_cls.push_back(c);
    }

    if (static_cast<int>(chosen.size()) < npages) {
        std::fprintf(stderr,
            "warning: only %zu of %d hugepages available (probed %ld) — "
            "falling back to plain mmap\n", chosen.size(), npages, probed);
        close(fd);
        return r;
    }
    if (!balanced)
        std::fprintf(stderr,
            "warning: hugepage pool cannot fill every channel class evenly "
            "(probed %ld pages, %d classes) — continuing with the best mix found\n",
            probed, K);

    void *base = hp_reserve_va(npages);
    if (!base) {
        std::fprintf(stderr, "warning: could not reserve %d GB of VA: %s\n",
                     npages, std::strerror(errno));
        close(fd);
        return r;
    }

    for (int i = 0; i < npages; i++) {
        void *slot = static_cast<uint8_t *>(base) + static_cast<uint64_t>(i) * HP_SIZE;
        void *got = mmap(slot, HP_SIZE, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED | MAP_HUGETLB | MAP_HUGE_1GB,
                         fd, static_cast<off_t>(chosen[i]) * static_cast<off_t>(HP_SIZE));
        if (got == MAP_FAILED) {
            std::fprintf(stderr, "warning: MAP_FIXED hugepage slot %d failed: %s\n",
                         i, std::strerror(errno));
            munmap(base, static_cast<uint64_t>(npages) * HP_SIZE);
            close(fd);
            return r;
        }
    }

    // Return every probed-but-unused page to the pool.
    std::vector<bool> used(probed, false);
    for (long idx : chosen) used[idx] = true;
    for (long idx = 0; idx < probed; idx++) {
        if (used[idx]) continue;
        fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                  static_cast<off_t>(idx) * static_cast<off_t>(HP_SIZE),
                  static_cast<off_t>(HP_SIZE));
    }

    r.base   = base;
    r.bytes  = static_cast<uint64_t>(npages) * HP_SIZE;
    r.npages = npages;
    r.memfd  = fd;
    r.phys.assign(npages, 0);
    r.cls.assign(npages, -1);
    r.balanced = balanced;

    // Verify the placement: each slot must resolve to the physical page we
    // classified during the probe.
    for (int i = 0; i < npages; i++) {
        uint8_t *slot = static_cast<uint8_t *>(base) + static_cast<uint64_t>(i) * HP_SIZE;
        *reinterpret_cast<volatile char *>(slot) = 0;
        uint64_t phys = pm.virt_to_phys(reinterpret_cast<uint64_t>(slot));
        r.phys[i] = phys;
        r.cls[i]  = phys ? info.classify(phys_to_dram(phys)) : -1;
        if (r.cls[i] != chosen_cls[i]) {
            std::fprintf(stderr,
                "warning: slot %d landed in class %d, expected %d "
                "(phys=0x%llx)\n", i, r.cls[i], chosen_cls[i],
                (unsigned long long)phys);
            r.balanced = false;
        }
    }
    return r;
}

// Plain anonymous hugepage mapping — the original behaviour. Fills in phys/cls
// too when pagemap is readable, so the address dump still works.
static inline HugeRegion hp_alloc_plain(int npages, const HpClassInfo &info,
                                 const Pagemap &pm) {
    HugeRegion r;
    uint64_t bytes = static_cast<uint64_t>(npages) * HP_SIZE;
    void *base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
    if (base == MAP_FAILED) return r;

    r.base   = base;
    r.bytes  = bytes;
    r.npages = npages;
    r.phys.assign(npages, 0);
    r.cls.assign(npages, -1);

    std::vector<int> count(info.nclass(), 0);
    bool all_known = true;
    for (int i = 0; i < npages; i++) {
        uint8_t *page = static_cast<uint8_t *>(base) + static_cast<uint64_t>(i) * HP_SIZE;
        *reinterpret_cast<volatile char *>(page) = 0;
        uint64_t phys = pm.virt_to_phys(reinterpret_cast<uint64_t>(page));
        r.phys[i] = phys;
        if (!phys) { all_known = false; continue; }
        r.cls[i] = info.classify(phys_to_dram(phys));
        count[r.cls[i]]++;
    }
    if (all_known) {
        r.balanced = true;
        const std::vector<int> quota = hp_quota(npages, info.nclass());
        for (int c = 0; c < info.nclass(); c++)
            if (count[c] != quota[c]) r.balanced = false;
    }
    return r;
}

static inline void hp_print(const HugeRegion &r, const DecodeMap &m,
                     const HpClassInfo &info, bool balanced_alloc) {
    std::printf("hugepages: %d x 1GB  map=%s  balance=%s  classes=%d",
                r.npages, m.name, balanced_alloc ? "on" : "off", info.nclass());
    for (int i = 0; i < info.n; i++)
        std::printf("%s ch_fn[%d]=0x%llx", i == 0 ? "  page-constant:" : "",
                    i, (unsigned long long)info.mask[i]);
    std::printf("\n");

    std::printf(" %-4s %-18s %-18s %-18s %s\n",
                "idx", "vaddr", "phys", "dram", "ch_class");
    std::vector<int> count(info.nclass(), 0);
    for (int i = 0; i < r.npages; i++) {
        uint64_t va = reinterpret_cast<uint64_t>(r.base) + static_cast<uint64_t>(i) * HP_SIZE;
        if (r.phys[i] == 0) {
            std::printf(" %-4d 0x%016llx %-18s %-18s %s\n",
                        i, (unsigned long long)va, "unknown", "unknown", "?");
            continue;
        }
        count[r.cls[i]]++;
        std::printf(" %-4d 0x%016llx 0x%016llx 0x%016llx %d\n",
                    i, (unsigned long long)va,
                    (unsigned long long)r.phys[i],
                    (unsigned long long)phys_to_dram(r.phys[i]),
                    r.cls[i]);
    }
    std::printf(" class histogram:");
    for (int c = 0; c < info.nclass(); c++) std::printf(" [%d]=%d", c, count[c]);
    std::printf("   balanced=%s\n", r.balanced ? "yes" : "no");
}

// ─────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────

// Allocate npages 1GB hugepages as one contiguous, 1GB-aligned region.
// balance:     select physical pages so every page-constant channel class is
//              equally represented, and interleave them in VA order.
// print_addrs: dump the per-page virtual/physical address table.
// Returns base == nullptr if even the plain mmap failed.
static inline HugeRegion hp_alloc(int npages, const DecodeMap &m,
                           bool balance, bool print_addrs) {
    Pagemap pm;
    HpClassInfo info = hp_class_info(m);

    bool want_balance = balance;
    if (want_balance && info.n == 0) {
        std::printf("note: every channel function of map=%s toggles inside a 1GB page — "
                    "each hugepage already covers all channels; balancing not needed\n",
                    m.name);
        want_balance = false;
    }
    if (want_balance && (DECODE_HOLE_SIZE % HP_SIZE) != 0) {
        std::fprintf(stderr,
            "warning: DRAM hole size 0x%llx is not a multiple of 1GB, so hugepage "
            "bases are not 1GB-aligned in DRAM address space — skipping balancing\n",
            (unsigned long long)DECODE_HOLE_SIZE);
        want_balance = false;
    }
    if ((want_balance || print_addrs) && geteuid() != 0) {
        std::fprintf(stderr,
            "warning: not running as root — /proc/self/pagemap reports PFN=0, so "
            "physical addresses and channel balancing are unavailable (re-run with sudo)\n");
        want_balance = false;
    }
    if (want_balance && !pm.ok()) {
        std::fprintf(stderr, "warning: cannot open /proc/self/pagemap: %s — skipping balancing\n",
                     std::strerror(errno));
        want_balance = false;
    }

    HugeRegion r;
    if (want_balance) r = hp_alloc_balanced(npages, info, pm);
    bool balanced_alloc = r.base != nullptr;
    if (!r.base) r = hp_alloc_plain(npages, info, pm);

    if (r.base && print_addrs) hp_print(r, m, info, balanced_alloc);
    return r;
}
