// Reports how many 1GB hugepages are configured in the system's hugetlb pool
// and the physical address of each one, by temporarily mmap'ing the free
// pages and resolving them through /proc/self/pagemap.
//
// Physical Frame Numbers in pagemap are zeroed for non-root processes
// (CAP_SYS_ADMIN required since Linux 4.0), so this must be run with sudo
// to get real addresses.
//
// Usage: sudo ./hugepage_addr
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

static const char *NR_PATH   = "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages";
static const char *FREE_PATH = "/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages";

static long read_sysfs_long(const char *path) {
    FILE *f = std::fopen(path, "r");
    if (!f) { std::perror(path); std::exit(1); }
    long v = -1;
    if (std::fscanf(f, "%ld", &v) != 1) {
        std::fprintf(stderr, "failed to parse %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return v;
}

int main() {
    const uint64_t GB = 1ULL << 30;
    const long page_size = sysconf(_SC_PAGESIZE);

    long total = read_sysfs_long(NR_PATH);
    long free_pages = read_sysfs_long(FREE_PATH);
    std::printf("1GB hugepages: total=%ld free=%ld\n", total, free_pages);

    if (total == 0) {
        std::printf("no 1GB hugepages configured on this system.\n");
        return 0;
    }
    if (free_pages == 0) {
        std::fprintf(stderr,
            "error: 0 free 1GB hugepages (all %ld are already mapped by another "
            "process) -- can't probe them without disturbing that process.\n",
            total);
        return 1;
    }

    if (geteuid() != 0) {
        std::fprintf(stderr,
            "warning: not running as root -- /proc/self/pagemap will report "
            "PFN=0 for every page. Re-run with sudo to get real physical "
            "addresses.\n");
    }

    long n = free_pages; // only currently-free pages can be safely claimed
    uint64_t span = static_cast<uint64_t>(n) * GB;

    void *base = mmap(nullptr, span, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                       -1, 0);
    if (base == MAP_FAILED) {
        std::perror("mmap 1GB hugepages failed");
        return 1;
    }

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        std::perror("open /proc/self/pagemap");
        munmap(base, span);
        return 1;
    }

    std::printf("%-4s %-18s %-18s %s\n", "idx", "virtual_addr", "physical_addr", "present");
    for (long i = 0; i < n; ++i) {
        uint8_t *page_va = static_cast<uint8_t *>(base) + i * GB;
        page_va[0] = 0; // fault the page in so pagemap has a real mapping

        uint64_t vpn = reinterpret_cast<uint64_t>(page_va) / page_size;
        off_t off = static_cast<off_t>(vpn * sizeof(uint64_t));
        uint64_t entry = 0;
        if (pread(fd, &entry, sizeof(entry), off) != sizeof(entry)) {
            std::perror("pread pagemap");
            continue;
        }

        bool present = (entry >> 63) & 1;
        uint64_t pfn = entry & ((1ULL << 55) - 1);
        uint64_t phys = pfn * page_size;

        std::printf("%-4ld 0x%016lx 0x%016lx %s\n",
                    i, reinterpret_cast<uint64_t>(page_va), phys,
                    present ? "yes" : "no");
    }

    close(fd);
    munmap(base, span);
    return 0;
}
