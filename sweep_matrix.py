#!/usr/bin/env python3
"""
Full sweep matrix: stream + rand(consecutive) + rand(samerow), across all
lines_per_access values and the configured core range.

Writes two CSVs:
  - full CSV    : one row per (pattern/access_mode/lines_per_access, cores)
  - summary CSV : one row per (pattern/access_mode/lines_per_access) with the
                  best bandwidth measured across the core sweep

Usage:
  python3 sweep_matrix.py                 # full matrix
  python3 sweep_matrix.py -c              # skip rand consecutive
  python3 sweep_matrix.py -s              # skip rand samerow
  python3 sweep_matrix.py -o out.csv --summary summary.csv

Edit config.py to change hardware/sweep settings (same config as sweep_bw.py).
"""

import argparse
import csv
import os
import sys

import config as cfg
import sweep_bw as sbw

LPA_VALUES = [1, 2, 4, 8, 16]

FULL_HEADER = [
    "pattern", "access_mode", "mode_label", "lines_per_access",
    "cores", "elapsed_s", "bw_gb_s", "pct_peak",
]
SUMMARY_HEADER = [
    "pattern", "access_mode", "mode_label", "lines_per_access",
    "max_bw_gb_s", "pct_peak", "cores",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--no-consecutive", action="store_true",
                   help="skip rand+consecutive (access_mode=0) cases")
    p.add_argument("-s", "--no-samerow", action="store_true",
                   help="skip rand+samerow (access_mode=1) cases")
    p.add_argument("-o", "--output", default="results/sweep_full.csv",
                   help="full-matrix CSV path (default: results/sweep_full.csv)")
    p.add_argument("--summary", default="results/sweep_summary.csv",
                   help="per-case best-result CSV path (default: results/sweep_summary.csv)")
    return p.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict]:
    """Return the ordered list of jobs to sweep. Each job carries everything
    needed to run the benchmark and to label its CSV rows."""
    jobs = [{
        "pattern": "stream", "access_mode": "", "mode_label": "",
        "lines_per_access": "", "binary": sbw.BINARY_STREAM, "mode": "stream",
    }]

    if not args.no_consecutive:
        for lpa in LPA_VALUES:
            jobs.append({
                "pattern": "rand", "access_mode": 0, "mode_label": "consecutive",
                "lines_per_access": lpa, "binary": sbw.BINARY_RAND, "mode": "rand",
            })

    if not args.no_samerow:
        import address_mapping as am
        max_col_bits = len(am.col_step_masks(cfg.ADDR_MAP))
        max_lpa = 2 ** max_col_bits
        for lpa in LPA_VALUES:
            if lpa > max_lpa:
                print(f"[WARN] skipping samerow lines_per_access={lpa}: "
                      f"exceeds max columns ({max_lpa}) for ADDR_MAP={cfg.ADDR_MAP!r}",
                      file=sys.stderr)
                continue
            jobs.append({
                "pattern": "rand", "access_mode": 1, "mode_label": "samerow",
                "lines_per_access": lpa, "binary": sbw.BINARY_RAND, "mode": "rand",
            })

    return jobs


def validate(jobs: list[dict]) -> None:
    if any(j["mode"] == "stream" for j in jobs) and not os.path.isfile(sbw.BINARY_STREAM):
        sys.exit(f"Binary not found: {sbw.BINARY_STREAM}\nRun 'make' first.")
    if any(j["mode"] == "rand" for j in jobs):
        if not os.path.isfile(sbw.BINARY_RAND):
            sys.exit(f"Binary not found: {sbw.BINARY_RAND}\nRun 'make' first.")
        hp = cfg.HUGEPAGES_1GB
        if hp < 1 or (hp & (hp - 1)) != 0:
            sys.exit(
                f"config.HUGEPAGES_1GB={hp} must be a power of 2 (1,2,4,8,...) "
                "for rand mode (LFSR mask addressing)."
            )
        if any(j["access_mode"] == 1 for j in jobs):
            import address_mapping as am
            if cfg.ADDR_MAP not in am.SYSTEMS:
                sys.exit(
                    f"config.ADDR_MAP={cfg.ADDR_MAP!r} not found in address_mapping.SYSTEMS "
                    f"(available: {', '.join(sorted(am.SYSTEMS))})."
                )


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    validate(jobs)

    node_cpus = sbw.numa_node_cpus(cfg.NUMA_NODE)
    max_cores = min(cfg.CORE_MAX if cfg.CORE_MAX is not None else len(node_cpus), len(node_cpus))
    peak = sbw.theoretical_peak_gb_s()

    core_list = list(range(cfg.CORE_START, max_cores + 1, cfg.CORE_STEP))
    if core_list[-1] != max_cores:
        core_list.append(max_cores)

    print("=" * 65)
    print(" sweep_matrix — full stream + rand(consecutive/samerow) sweep")
    print("=" * 65)
    print(f"  Peak (theoretical): {peak:.1f} GB/s")
    print(f"  Core range        : {cfg.CORE_START} .. {max_cores} (step {cfg.CORE_STEP})")
    print(f"  Jobs              : {len(jobs)}")
    print("=" * 65)
    print()

    full_rows: list[list] = []
    summary_rows: list[list] = []

    for job in jobs:
        label = job["pattern"]
        if job["mode_label"]:
            label += f"/{job['mode_label']}/lpa={job['lines_per_access']}"
        print(f"--- {label} ---")

        best: tuple[float, float, int] | None = None  # (bw, pct, cores)
        for ncores in core_list:
            cpus = node_cpus[:ncores]
            print(f"  Running {ncores:>2} core(s) ... ", end="", flush=True)

            elapsed, bw = sbw.run_benchmark(
                cpus, job["binary"], job["mode"],
                lines_per_access=job["lines_per_access"] or None,
                access_mode=job["access_mode"] if job["access_mode"] != "" else None,
            )
            if bw is None:
                print("FAILED")
                continue

            pct = round(bw / peak * 100.0, 1)
            chart = sbw.bar(bw, peak)
            print(f"\r  {ncores:>2} core(s): {elapsed:.2f}s  {bw:>7.3f} GB/s  {pct:>5.1f}%  {chart}")

            full_rows.append([
                job["pattern"], job["access_mode"], job["mode_label"],
                job["lines_per_access"], ncores, elapsed, bw, pct,
            ])
            if best is None or bw > best[0]:
                best = (bw, pct, ncores)

        if best is not None:
            summary_rows.append([
                job["pattern"], job["access_mode"], job["mode_label"],
                job["lines_per_access"], best[0], best[1], best[2],
            ])
        print()

    for path in (args.output, args.summary):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FULL_HEADER)
        w.writerows(full_rows)

    with open(args.summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        w.writerows(summary_rows)

    print("=" * 65)
    print(f"  Full matrix  -> {args.output}  ({len(full_rows)} rows)")
    print(f"  Summary      -> {args.summary}  ({len(summary_rows)} rows)")
    print("=" * 65)


if __name__ == "__main__":
    main()
