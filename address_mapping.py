#!/usr/bin/env python3
"""DRAM physical-address -> DRAM-location mapping registry, and a GF(2) solver
that derives XOR masks for stepping through adjacent columns while staying in
the same channel/rank/bank-group/bank/row.

Each entry in SYSTEMS describes how a memory controller's XOR-hash address
decoding carves up a physical address (bit 0 = LSB, offset bits excluded):

  offset_bits : number of low bits that are the cacheline byte offset.
  channel     : list of bit-masks; each mask is one XOR-parity function
                (one output bit of the channel selector).
  rank        : same, for rank selection.
  bank_group  : same, for bank-group (BG) selection.
  bank        : same, for bank (BA) selection.
  row         : single bit-mask of the physical bits that are copied
                directly (no XOR) into the row address.
  column      : single bit-mask of the physical bits that are copied
                directly into the column address; the set bits, taken in
                ascending physical-bit order, define column bit 0, 1, 2, ...

channel/rank/bank_group/bank bits may overlap with row bits (common in
real controllers, so that consecutive rows interleave across banks) — the
solver below accounts for that.
"""

# ─────────────────────────────────────────────────────────────────────────────
# System registry
# ─────────────────────────────────────────────────────────────────────────────
# Add new systems here (e.g. SPR configurations) using the same shape.

SYSTEMS = {
    # DDR5, Arrow Lake (gpu17): 1 channel populated, 1 DPC, 2 ranks, 32GB.
    "arrow_1ch_1dpc_2rank_32gb": {
        "offset_bits": 6,
        "channel":     [0x82600],
        "rank":        [0x42120000],
        "bank_group":  [0x84042100, 0x108404000, 0x210808000],
        "bank":        [0x421090000, 0x240000],
        "row":         0x7fff80000,
        "column":      0x1bc0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Decode: physical address -> DRAM location (used for self-checking masks)
# ─────────────────────────────────────────────────────────────────────────────

def _parity(addr: int, mask: int) -> int:
    return bin(addr & mask).count("1") & 1


def _bits_value(addr: int, mask: int) -> int:
    """Extract the bits set in `mask` from `addr`, packed LSB-first in
    ascending physical-bit order, as a single integer."""
    value = 0
    out_bit = 0
    bit = 0
    m = mask
    while m:
        if m & 1:
            value |= ((addr >> bit) & 1) << out_bit
            out_bit += 1
        m >>= 1
        bit += 1
    return value


def decode(system: str, addr: int) -> dict:
    """Return {channel, rank, bank_group, bank, row, col} indices for addr."""
    s = SYSTEMS[system]

    def multi(masks):
        return sum(_parity(addr, m) << i for i, m in enumerate(masks))

    return {
        "channel":    multi(s["channel"]),
        "rank":       multi(s["rank"]),
        "bank_group": multi(s["bank_group"]),
        "bank":       multi(s["bank"]),
        "row":        _bits_value(addr, s["row"]),
        "col":        _bits_value(addr, s["column"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) linear basis (a.k.a. XOR basis), tracking which original slack bit
# indices combine to reproduce a target signature vector.
# ─────────────────────────────────────────────────────────────────────────────

class _XorBasis:
    """Basis over GF(2) vectors of `nbits` bits. Each inserted vector carries
    a `combo` bitmask (over caller-defined indices) recording which original
    elements XOR together to form it, so solved targets can be traced back
    to a concrete subset of original elements."""

    def __init__(self, nbits: int):
        self.nbits = nbits
        self.slot = [None] * nbits  # slot[b] = (vector, combo) with top bit == b

    def insert(self, vector: int, combo: int) -> None:
        v, c = vector, combo
        for b in reversed(range(self.nbits)):
            if not (v >> b) & 1:
                continue
            if self.slot[b] is None:
                self.slot[b] = (v, c)
                return
            v ^= self.slot[b][0]
            c ^= self.slot[b][1]
        # v == 0 here: vector was linearly dependent, nothing to add.

    def solve(self, target: int):
        """Return a combo bitmask that reproduces `target`, or None if
        `target` is not in the span of the inserted vectors."""
        v, c = target, 0
        for b in reversed(range(self.nbits)):
            if not (v >> b) & 1:
                continue
            if self.slot[b] is None:
                return None
            v ^= self.slot[b][0]
            c ^= self.slot[b][1]
        return c if v == 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Column-step mask solver
# ─────────────────────────────────────────────────────────────────────────────

def _bank_functions(s: dict) -> list:
    return list(s["channel"]) + list(s["rank"]) + list(s["bank_group"]) + list(s["bank"])


def col_step_masks(system: str, max_bits: int | None = None) -> list:
    """Compute, for each column bit (ascending, LSB-first), the smallest XOR
    mask that flips exactly that column bit while leaving channel, rank,
    bank_group, bank and row unchanged.

    Returns a list of masks; mask[i] toggles column bit i only. Combining
    masks for the set bits of d (0 <= d < 2**len(masks)) via XOR yields the
    mask that steps the column by d while staying in the same bank/row:

        step_mask(d) = XOR of mask[i] for each bit i set in d

    If a column bit's bank-function signature cannot be cancelled using the
    available slack bits, the list stops there (columns beyond that point
    are not reachable while preserving bank/row) — callers should treat the
    returned length as the number of usable column bits.
    """
    s = SYSTEMS[system]
    functions = _bank_functions(s)
    row_mask = s["row"]
    col_mask = s["column"]

    col_bits = [i for i in range(64) if (col_mask >> i) & 1]
    if max_bits is not None:
        col_bits = col_bits[:max_bits]

    # Slack bits: not part of column or row, but relevant to at least one
    # bank-selection function (bits irrelevant to every function can never
    # help cancel anything, so they're excluded).
    slack_bits = [
        i for i in range(64)
        if not (col_mask >> i) & 1
        and not (row_mask >> i) & 1
        and any((f >> i) & 1 for f in functions)
    ]

    def signature(bit: int) -> int:
        return sum(((f >> bit) & 1) << i for i, f in enumerate(functions))

    basis = _XorBasis(len(functions))
    for idx, bit in enumerate(slack_bits):
        basis.insert(signature(bit), 1 << idx)

    masks = []
    for bit in col_bits:
        combo = basis.solve(signature(bit))
        if combo is None:
            break
        slack_mask = 0
        for idx, slack_bit in enumerate(slack_bits):
            if (combo >> idx) & 1:
                slack_mask |= 1 << slack_bit
        masks.append((1 << bit) | slack_mask)

    return masks
