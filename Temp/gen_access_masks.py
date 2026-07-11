#!/usr/bin/env python3
"""Generate access_masks.h from the ADDR_MAP selected in config.py.

randread_bw walks LINES_PER_ACCESS cachelines of the same
channel/rank/bank-group/bank/row by XOR-ing a randomly picked cacheline
address with a small set of per-column-bit masks. Two mask sets are emitted
per registered system:
  low  (mode 1 "samebank"): lowest column bits  -> ADJACENT columns.
  high (mode 2 "spread")  : highest column bits  -> FAR-APART columns
                            (the upper column bits differ).
This script computes both via address_mapping.col_step_masks() (high=True for
the spread set), verifies them against address_mapping.decode(), and emits
them as a C header consumed by randread_bw.cpp.

Run manually with: python3 gen_access_masks.py [-o access_masks.h]
The Makefile also runs this automatically before building randread_bw.
"""

import argparse
import random
import sys

import config as cfg
import address_mapping as am

# LINES_PER_ACCESS tops out at 16 (see config_template.py), which needs 4
# column bits (2**4 == 16); no point solving for more.
MAX_COL_BITS = 4


def self_check(name: str, kind: str, masks: list) -> None:
    """Verify each mask toggles exactly one column bit and that the toggled
    bits are all distinct, while leaving channel/rank/bank_group/bank/row
    untouched, across random addresses. Works for both the low (adjacent) and
    high (spread) mask sets — the toggled bit need not equal the mask index."""
    rnd = random.Random(0)
    toggled = []  # toggled[i] = column bit index that masks[i] flips
    for _ in range(256):
        base = rnd.getrandbits(36) & ~0x3f  # cacheline-aligned test address
        d0 = am.decode(name, base)
        for i, mask in enumerate(masks):
            d1 = am.decode(name, base ^ mask)
            for key in ("channel", "rank", "bank_group", "bank", "row"):
                if d0[key] != d1[key]:
                    sys.exit(
                        f"address_mapping self-check failed: system={name!r} "
                        f"{kind} mask[{i}]=0x{mask:x} changed {key} "
                        f"({d0[key]} -> {d1[key]}) at addr=0x{base:x}"
                    )
            diff = d0["col"] ^ d1["col"]
            if diff == 0 or (diff & (diff - 1)) != 0:
                sys.exit(
                    f"address_mapping self-check failed: system={name!r} "
                    f"{kind} mask[{i}]=0x{mask:x} did not toggle exactly one "
                    f"column bit (col {d0['col']} -> {d1['col']}) at addr=0x{base:x}"
                )
            bit = diff.bit_length() - 1
            if i == len(toggled):
                toggled.append(bit)
            elif toggled[i] != bit:
                sys.exit(
                    f"address_mapping self-check failed: system={name!r} "
                    f"{kind} mask[{i}]=0x{mask:x} toggles column bit {bit} but "
                    f"{toggled[i]} on another address"
                )
    if len(set(toggled)) != len(toggled):
        sys.exit(
            f"address_mapping self-check failed: system={name!r} {kind} masks "
            f"toggle duplicate column bits {toggled}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="access_masks.h",
                         help="output header path (default: access_masks.h)")
    args = parser.parse_args()

    # config.ADDR_MAP is the compile-time DEFAULT map (used at runtime when the
    # ACCESS_MAP env var is unset); it must exist in the registry.
    default = cfg.ADDR_MAP
    if default not in am.SYSTEMS:
        sys.exit(
            f"config.ADDR_MAP={default!r} not found in address_mapping.SYSTEMS "
            f"(available: {', '.join(sorted(am.SYSTEMS))})"
        )

    # Solve + self-check every registered system; emit them all as a table so
    # randread_bw can pick one at runtime (ACCESS_MAP env) without recompiling.
    # Two mask sets per system:
    #   low  -> mode 1 "samebank": lowest column bits  (adjacent columns)
    #   high -> mode 2 "spread"  : highest column bits  (far-apart columns)
    entries = []  # (name, low_masks, high_masks)
    for name in sorted(am.SYSTEMS):
        low = am.col_step_masks(name, max_bits=MAX_COL_BITS)
        high = am.col_step_masks(name, max_bits=MAX_COL_BITS, high=True)
        self_check(name, "low", low)    # no-op loop body when masks is empty
        self_check(name, "high", high)
        entries.append((name, low, high))

    def mask_row(masks: list) -> str:
        # Pad each mask row to ACCESS_MAX_COL_BITS so the struct layout is fixed.
        padded = list(masks) + [0] * (MAX_COL_BITS - len(masks))
        return ", ".join(f"0x{m:x}ULL" for m in padded)

    def entry_line(name: str, low: list, high: list) -> str:
        return (f'    {{ "{name}", '
                f'{len(low)}, {{ {mask_row(low)} }}, '
                f'{len(high)}, {{ {mask_row(high)} }} }}')

    table = ",\n".join(entry_line(n, lo, hi) for n, lo, hi in entries)
    header = f"""// AUTO-GENERATED by gen_access_masks.py from address_mapping.SYSTEMS.
// Do not edit by hand — re-run gen_access_masks.py (the Makefile does this
// automatically whenever config.py or address_mapping.py change).
#pragma once
#include <cstdint>

// ACCESS_MAX_COL_BITS bounds each col_step_mask[] per map. It equals
// gen_access_masks.MAX_COL_BITS, which is {MAX_COL_BITS} because lines_per_access <= 16 == 2**{MAX_COL_BITS},
// so no mapping ever needs more than {MAX_COL_BITS} steppable column bits. Using this fixed
// cap (not the per-registry max) keeps the struct layout stable.
#define ACCESS_MAX_COL_BITS {MAX_COL_BITS}

// Default map used at runtime when the ACCESS_MAP env var is unset (== config.ADDR_MAP).
#define ACCESS_DEFAULT_MAP "{default}"

struct AccessMap {{
    const char *name;
    // Each col_step_mask[i] toggles exactly one column bit while preserving
    // channel/rank/bank_group/bank/row. XOR together the masks for the set
    // bits of d (0 <= d < 2**num_col_bits) to step the column by d.
    //   low  : lowest column bits  -> mode 1 "samebank", ADJACENT columns.
    //   high : highest column bits, topmost-first -> mode 2 "spread",
    //          FAR-APART columns (upper column bits differ).
    int         num_col_bits_low;               // valid entries in col_step_mask_low
    uint64_t    col_step_mask_low[ACCESS_MAX_COL_BITS];
    int         num_col_bits_high;              // valid entries in col_step_mask_high
    uint64_t    col_step_mask_high[ACCESS_MAX_COL_BITS];
}};

static const AccessMap ACCESS_MAPS[] = {{
{table}
}};
static const int ACCESS_NUM_MAPS = (int)(sizeof(ACCESS_MAPS) / sizeof(ACCESS_MAPS[0]));
"""
    with open(args.output, "w") as f:
        f.write(header)

    print(f"wrote {args.output}  ({len(entries)} map(s), default={default}; "
          f"col bits low/high: "
          + ", ".join(f"{n}={len(lo)}/{len(hi)}" for n, lo, hi in entries) + ")")


if __name__ == "__main__":
    main()
