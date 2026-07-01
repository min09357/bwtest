#!/usr/bin/env python3
"""
randread_bw / stream_bw core sweep
Runs ./randread_bw or ./stream_bw from 1 core up to CORE_MAX with CORE_STEP increments,
then reports bandwidth and the percentage of theoretical DRAM peak.

Usage:
  python3 sweep_bw.py          # random access (default)
  python3 sweep_bw.py rand     # random access
  python3 sweep_bw.py stream   # sequential access

Edit config.py to change hardware/sweep settings.
"""

import re
import subprocess
import sys
import os

import config as cfg

# Paths to binaries (resolved relative to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_RAND   = os.path.join(_SCRIPT_DIR, "randread_bw")
BINARY_STREAM = os.path.join(_SCRIPT_DIR, "stream_bw")

# ─────────────────────────────────────────────────────────────────────────────


def numa_node_cpus(node: int) -> list[int]:
    """Return sorted CPU list for the given NUMA node via sysfs.
    Falls back to all logical CPUs if sysfs is unavailable."""
    path = f"/sys/devices/system/node/node{node}/cpulist"
    try:
        text = open(path).read().strip()
    except OSError:
        return list(range(os.cpu_count() or 1))
    cpus: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(token))
    return sorted(cpus)


def resolve_mode(argv: list[str]) -> tuple[str, str]:
    """Return (mode, binary_path) based on command-line argument."""
    arg = argv[1].lower() if len(argv) > 1 else "rand"
    if arg == "stream":
        return "stream", BINARY_STREAM
    if arg in ("rand", "random"):
        return "rand", BINARY_RAND
    sys.exit(f"Unknown mode '{argv[1]}'. Use: rand (default) or stream")


def theoretical_peak_gb_s() -> float:
    """Peak BW = rate (MT/s) × 8 B/transfer × channels / 1000"""
    return cfg.DIMM_TRANSFER_RATE_MT_S * 8 * cfg.DIMM_CHANNELS / 1000.0


def parse_bandwidth(output: str) -> tuple[float, float]:
    """Return (elapsed_s, bw_gb_s) from randread_bw stdout."""
    m_elapsed = re.search(r"Elapsed\s*:\s*([\d.]+)\s*s", output)
    m_bw      = re.search(r"Bandwidth\s*:\s*([\d.]+)\s*GB/s", output)
    if not m_elapsed or not m_bw:
        raise ValueError(f"Could not parse output:\n{output}")
    return float(m_elapsed.group(1)), float(m_bw.group(1))


def probe_simd_width(binary: str) -> str:
    """Ask the binary which SIMD width it was compiled with (--simd: no measurement run)."""
    try:
        result = subprocess.run(
            [binary, "--simd"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_benchmark(cpus: list[int], binary: str, mode: str) -> tuple[float, float]:
    """Run the benchmark binary under numactl and return (elapsed_s, bw_gb_s)."""
    cpulist = ",".join(map(str, cpus))
    cmd = []
    if cfg.USE_SUDO:
        cmd.append("sudo")
    cmd += [
        "numactl",
        "-C", cpulist,
        "-m", str(cfg.NUMA_NODE),
        binary, str(len(cpus)), str(cfg.ITERS_PER_THREAD), str(cfg.HUGEPAGES_1GB),
    ]
    if mode == "rand":
        cmd.append(str(cfg.LINES_PER_ACCESS))
        cmd.append(str(cfg.ACCESS_MODE))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] core={len(cpus)} failed (rc={result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None, None
    return parse_bandwidth(result.stdout)


def bar(value: float, max_value: float, width: int = 30) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    mode, binary = resolve_mode(sys.argv)

    if not os.path.isfile(binary):
        sys.exit(f"Binary not found: {binary}\nRun 'make' first.")

    # randread uses LFSR mask addressing — hugepage count must be a power of 2
    if mode == "rand":
        hp = cfg.HUGEPAGES_1GB
        if hp < 1 or (hp & (hp - 1)) != 0:
            sys.exit(
                f"config.HUGEPAGES_1GB={hp} must be a power of 2 (1,2,4,8,...) "
                "for rand mode (LFSR mask addressing)."
            )
        lpa = cfg.LINES_PER_ACCESS
        if lpa not in (1, 2, 4, 8, 16):
            sys.exit(
                f"config.LINES_PER_ACCESS={lpa} must be one of {{1,2,4,8,16}}."
            )
        if cfg.ACCESS_MODE not in (0, 1):
            sys.exit(
                f"config.ACCESS_MODE={cfg.ACCESS_MODE} must be 0 (consecutive) or 1 (samebank)."
            )
        if cfg.ACCESS_MODE == 1:
            import address_mapping as am
            if cfg.ADDR_MAP not in am.SYSTEMS:
                sys.exit(
                    f"config.ADDR_MAP={cfg.ADDR_MAP!r} not found in address_mapping.SYSTEMS "
                    f"(available: {', '.join(sorted(am.SYSTEMS))})."
                )

    node_cpus = numa_node_cpus(cfg.NUMA_NODE)
    max_cores = min(cfg.CORE_MAX if cfg.CORE_MAX is not None else len(node_cpus), len(node_cpus))
    peak      = theoretical_peak_gb_s()
    bus_width = 8  # bytes per transfer (DDR standard)

    core_list = list(range(cfg.CORE_START, max_cores + 1, cfg.CORE_STEP))
    if core_list[-1] != max_cores:
        core_list.append(max_cores)

    simd_width = probe_simd_width(binary)
    cpu_preview = ",".join(map(str, node_cpus[:4])) + (",..." if len(node_cpus) > 4 else "")
    mode_label = "random access" if mode == "rand" else "sequential access"
    print("=" * 65)
    print(f" {os.path.basename(binary)} — core sweep  [{mode_label}]")
    print("=" * 65)
    print(f"  DIMM rate   : {cfg.DIMM_TRANSFER_RATE_MT_S} MT/s × {bus_width} B × "
          f"{cfg.DIMM_CHANNELS} ch  →  peak {peak:.1f} GB/s")
    print(f"  NUMA node   : {cfg.NUMA_NODE}  (CPUs: {cpu_preview})")
    print(f"  Core range  : {cfg.CORE_START} .. {max_cores}  (step {cfg.CORE_STEP})")
    print(f"  Iters/thread: {cfg.ITERS_PER_THREAD:,}")
    print(f"  Hugepages   : {cfg.HUGEPAGES_1GB} × 1GB  ({cfg.HUGEPAGES_1GB} GB region)")
    if mode == "rand":
        print(f"  Lines/access: {cfg.LINES_PER_ACCESS}  ({cfg.LINES_PER_ACCESS * 64} B/access)")
        access_label = "consecutive" if cfg.ACCESS_MODE == 0 else "samebank"
        addr_map_note = f"  (addr_map={cfg.ADDR_MAP})" if cfg.ACCESS_MODE == 1 else ""
        print(f"  Access mode : {cfg.ACCESS_MODE} ({access_label}){addr_map_note}")
    print(f"  Binary      : {binary}")
    print(f"  SIMD width  : {simd_width}")
    print("=" * 65)
    print()

    header = f"{'Cores':>6}  {'Elapsed':>8}  {'BW (GB/s)':>10}  {'% peak':>7}  Chart"
    print(header)
    print("-" * 65)

    results: list[tuple[int, float, float]] = []

    for ncores in core_list:
        cpus = node_cpus[:ncores]
        print(f"  Running {ncores:>2} core(s) ... ", end="", flush=True)

        elapsed, bw = run_benchmark(cpus, binary, mode)
        if bw is None:
            print("FAILED")
            continue

        pct = bw / peak * 100.0
        results.append((ncores, elapsed, bw))
        chart = bar(bw, peak)
        print(f"\r{ncores:>6}  {elapsed:>7.2f}s  {bw:>10.3f}  {pct:>6.1f}%  {chart}")

    if not results:
        sys.exit("No successful measurements.")

    print()
    print("=" * 65)
    max_bw    = max(bw for _, _, bw in results)
    max_cores_result, _, _ = max(results, key=lambda x: x[2])
    print(f"  Peak measured : {max_bw:.3f} GB/s at {max_cores_result} core(s)")
    print(f"  % of DRAM peak: {max_bw / peak * 100:.1f}%  "
          f"(theoretical {peak:.1f} GB/s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
