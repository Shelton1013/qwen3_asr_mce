"""Levenshtein alignment that keeps enough position information to localise errors.

A plain edit distance gives you a number.  To answer "is the model dropping
English?" or "are the errors concentrated at switch points?" you need to know
*which* reference token each edit landed on, including for insertions -- which
by definition have no reference token.  Every op therefore carries ``ref_pos``:
the reference index the op is anchored at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

EQUAL = "eq"
SUB = "sub"
DELETE = "del"
INSERT = "ins"


@dataclass
class Op:
    """One edit operation in the alignment.

    ``ref_idx`` / ``hyp_idx`` are the token indices involved (``None`` when the
    operation has no counterpart on that side).  ``ref_pos`` is always set: for
    an insertion it is the index of the reference token the inserted material
    sits *before*, which is what lets us test whether an insertion fell inside a
    switch-point window.
    """

    op: str
    ref_idx: Optional[int] = None
    hyp_idx: Optional[int] = None
    ref_pos: int = 0

    @property
    def is_error(self) -> bool:
        return self.op != EQUAL


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Op]:
    """Align two token sequences, returning the edit path in reading order.

    Uniform unit costs, ties broken towards substitution then deletion. That
    tie-break is arbitrary but fixed, so per-token attributions stay stable
    across runs and across models.
    """
    n, m = len(ref), len(hyp)
    # Full matrix: utterances are short and we need it for the backtrace.
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i
    for j in range(1, m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        ref_tok = ref[i - 1]
        row, prev = dist[i], dist[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_tok == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)

    ops: List[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dist[i][j] == dist[i - 1][j - 1] + cost:
                ops.append(Op(EQUAL if cost == 0 else SUB, i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(Op(DELETE, i - 1, None))
            i -= 1
            continue
        ops.append(Op(INSERT, None, j - 1))
        j -= 1
    ops.reverse()
    return _annotate_positions(ops)


def _annotate_positions(ops: List[Op]) -> List[Op]:
    consumed = 0
    for op in ops:
        if op.op == INSERT:
            op.ref_pos = consumed
        else:
            op.ref_pos = op.ref_idx if op.ref_idx is not None else consumed
            consumed = op.ref_pos + 1
    return ops


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Plain distance, used by tests and by the length-ratio sanity checks."""
    return sum(1 for op in align(ref, hyp) if op.is_error)
