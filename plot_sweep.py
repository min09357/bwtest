#!/usr/bin/env python3
"""
Plot bandwidth-utilization graphs from sweep_matrix.py's full CSV output.

Generates four PNGs next to the input CSV:
  <stem>_cores.png             : cores vs utilization (stream, random @ LPA=1)
  <stem>_lpa.png                : lines_per_access vs best utilization (consecutive + samerow)
  <stem>_lpa_consecutive.png    : lines_per_access vs best utilization (consecutive only)
  <stem>_lpa_samerow.png        : lines_per_access vs best utilization (same row only)

Usage:
  python3 plot_sweep.py [results/sweep_full.csv]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOR_STREAM = "tab:orange"
COLOR_RANDOM_CONSECUTIVE = "tab:blue"
COLOR_RANDOM_SAMEROW = "yellowgreen"

TREFI_NS = 1950.0
TRFC_NS = 160.0
REFRESH_CAP = (TREFI_NS - TRFC_NS) / TREFI_NS * 100.0

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["access_mode"] = pd.to_numeric(df["access_mode"], errors="coerce")
    df["lines_per_access"] = pd.to_numeric(df["lines_per_access"], errors="coerce")
    return df


def plot_cores(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    cons = df[(df["pattern"] == "rand") & (df["access_mode"] == 0) & (df["lines_per_access"] == 1)].sort_values("cores")
    if not cons.empty:
        ax.plot(cons["cores"], cons["pct_peak"], marker="s", color=COLOR_RANDOM_CONSECUTIVE, label="Random")

    stream = df[df["pattern"] == "stream"].sort_values("cores")
    if not stream.empty:
        ax.plot(stream["cores"], stream["pct_peak"], marker="o", color=COLOR_STREAM, label="Stream")

    ax.axhline(REFRESH_CAP, color="red", linestyle="--", linewidth=2.5,
               label=f"All-bank refresh cap ({REFRESH_CAP:.1f}%)")

    ax.set_xlabel("Cores")
    ax.set_ylabel("Bandwidth utilization (%)")
    ax.set_title("Bandwidth utilization vs Cores")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lpa(df: pd.DataFrame, out_path: Path, include_consecutive: bool = True,
             include_samerow: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    if include_consecutive:
        cons = df[(df["pattern"] == "rand") & (df["access_mode"] == 0)]
        if not cons.empty:
            cons_best = cons.groupby("lines_per_access")["pct_peak"].max().sort_index()
            ax.plot(cons_best.index, cons_best.values, marker="s", color=COLOR_RANDOM_CONSECUTIVE,
                    label="Random max (consecutive address)")

    if include_samerow:
        same = df[(df["pattern"] == "rand") & (df["access_mode"] == 1)]
        if not same.empty:
            same_best = same.groupby("lines_per_access")["pct_peak"].max().sort_index()
            ax.plot(same_best.index, same_best.values, marker="^", color=COLOR_RANDOM_SAMEROW,
                    label="Random max (same row)")

    stream = df[df["pattern"] == "stream"]
    if not stream.empty:
        stream_max = stream["pct_peak"].max()
        ax.axhline(stream_max, color=COLOR_STREAM, linestyle="--", linewidth=1.8,
                   label=f"Stream max ({stream_max:.1f}%)")

    ax.axhline(REFRESH_CAP, color="red", linestyle="--", linewidth=2.5,
               label=f"All-bank refresh cap ({REFRESH_CAP:.1f}%)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("LINES_PER_ACCESS")
    ax.set_ylabel("Bandwidth utilization (%)")
    ax.set_title("Bandwidth utilization vs LINES_PER_ACCESS")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    lpa_values = sorted(df.loc[df["pattern"] == "rand", "lines_per_access"].dropna().unique())
    if lpa_values:
        ax.set_xticks(lpa_values)
        ax.set_xticklabels([str(int(v)) for v in lpa_values])
        ax.minorticks_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sweep_full.csv")
    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}")

    df = load(csv_path)

    cores_out = csv_path.with_name(csv_path.stem + "_cores.png")
    lpa_out = csv_path.with_name(csv_path.stem + "_lpa.png")
    lpa_cons_out = csv_path.with_name(csv_path.stem + "_lpa_consecutive.png")
    lpa_same_out = csv_path.with_name(csv_path.stem + "_lpa_samerow.png")

    plot_cores(df, cores_out)
    plot_lpa(df, lpa_out)
    plot_lpa(df, lpa_cons_out, include_samerow=False)
    plot_lpa(df, lpa_same_out, include_consecutive=False)

    print(f"Wrote {cores_out}")
    print(f"Wrote {lpa_out}")
    print(f"Wrote {lpa_cons_out}")
    print(f"Wrote {lpa_same_out}")


if __name__ == "__main__":
    main()
