#!/usr/bin/env python
"""Test whether code-mixing density predicts language-identification collapse.

A mean comparison ("collapsed utterances have CMI 0.357 vs 0.165") establishes
that the two groups differ, and the direction is safe because CMI is computed on
the reference and is therefore fixed before the model runs. It does not
establish that *mixing density* is what matters, because CMI travels with
several other things:

* absolute English token count -- maybe the trigger is "enough English", not
  "a high proportion of English"
* utterance length
* speaker -- if the collapses cluster in two voices it is an acoustic effect
  wearing a linguistic costume
* topic -- travel and sports carry more English than weather

This script runs the checks that separate those, plus the one piece of evidence
worth more than any of them: a dose-response curve. A mean difference is one
number; a collapse rate that rises monotonically with mixing density across
deciles is a mechanism.

Usage::

    python scripts/analyze_collapse.py exp/mce_stage0/qwen3-asr-0.6b.utts.jsonl \\
        --manifest data/mce/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mce.data import read_jsonl  # noqa: E402
from mce.tokenizer import EN, ZH  # noqa: E402

_SPEAKER_RE = re.compile(r"^(\d+)_MCE")


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def permutation_test(
    group_a: Sequence[float], group_b: Sequence[float], n: int = 10000, seed: int = 0
) -> float:
    """Two-sided p-value for a difference in means, without scipy.

    The groups here are wildly unbalanced (43 vs 4121), which makes parametric
    assumptions uncomfortable; shuffling the labels assumes nothing.
    """
    observed = abs(mean(group_a) - mean(group_b))
    pool = list(group_a) + list(group_b)
    k = len(group_a)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(mean(pool[:k]) - mean(pool[k:])) >= observed:
            hits += 1
    return (hits + 1) / (n + 1)


def dose_response(rows: Sequence[dict], key: str, bins: int = 10) -> List[tuple]:
    """Collapse rate per quantile bin of ``key``. Returns (lo, hi, n, collapsed)."""
    usable = [r for r in rows if r.get(key) is not None]
    usable.sort(key=lambda r: r[key])
    if not usable:
        return []
    out = []
    size = len(usable) / bins
    for b in range(bins):
        chunk = usable[int(b * size) : int((b + 1) * size)]
        if not chunk:
            continue
        collapsed = sum(1 for r in chunk if r.get("foreign_script"))
        out.append((chunk[0][key], chunk[-1][key], len(chunk), collapsed))
    return out


def print_dose_response(rows: Sequence[dict], key: str, label: str, bins: int) -> None:
    print(f"\n-- collapse rate by {label} ({bins} equal-count bins) --")
    print(f"  {'range':>18}  {'n':>6}  {'collapsed':>9}  {'rate':>7}")
    for lo, hi, n, collapsed in dose_response(rows, key, bins):
        bar = "#" * int(collapsed / n * 200)
        print(f"  {lo:7.3f} - {hi:7.3f}  {n:6d}  {collapsed:9d}  "
              f"{collapsed / n * 100:6.2f}%  {bar}")


def stratified_grid(rows: Sequence[dict], strata: int = 3, cells: int = 3) -> None:
    """Collapse rate in a length x CMI grid.

    CMI is ``(N - max(zh, en)) / N``, so it is bounded by the minority side: the
    same amount of English yields a higher CMI in a shorter utterance. Collapsed
    utterances are also shorter. That entanglement means a marginal CMI effect
    could be a length effect in disguise.

    Holding length roughly constant within a stratum breaks the tie. If the
    collapse rate still climbs across CMI cells *inside* each length band, the
    mixing proportion is doing the work; if it flattens, length was.
    """
    usable = [r for r in rows if r.get("cmi_ref") is not None and r.get("n_ref_tokens")]
    if not usable:
        return
    by_len = sorted(usable, key=lambda r: r["n_ref_tokens"])
    step = len(by_len) / strata

    print(f"\n-- collapse rate by CMI, within length strata --")
    print(f"  {'length band':>16}  " + "".join(f"{'CMI cell ' + str(c + 1):>16}" for c in range(cells)))
    for s in range(strata):
        band = by_len[int(s * step) : int((s + 1) * step)]
        if not band:
            continue
        band = sorted(band, key=lambda r: r["cmi_ref"])
        cell_step = len(band) / cells
        lo, hi = band[0]["n_ref_tokens"], band[-1]["n_ref_tokens"]
        row = f"  {lo:6d} - {hi:5d}  "
        for c in range(cells):
            cell = band[int(c * cell_step) : int((c + 1) * cell_step)]
            if not cell:
                row += f"{'-':>16}"
                continue
            hits = sum(1 for r in cell if r.get("foreign_script"))
            row += f"{hits:>5d}/{len(cell):<4d}{hits / len(cell) * 100:5.1f}%"
        print(row)
    print("  (each row holds length roughly constant; read left to right)")


def concentration(rows: Sequence[dict], key: str, label: str, top: int = 8) -> None:
    """Are the collapses spread across the corpus or piled into a few groups?"""
    totals: Dict[str, int] = {}
    hits: Dict[str, int] = {}
    for r in rows:
        k = r.get(key)
        if k is None:
            continue
        totals[k] = totals.get(k, 0) + 1
        if r.get("foreign_script"):
            hits[k] = hits.get(k, 0) + 1
    if not hits:
        print(f"\n-- by {label} --\n  no collapses")
        return
    n_collapsed = sum(hits.values())
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    covered = sum(c for _, c in ranked[:top])
    print(f"\n-- by {label} --")
    print(f"  {len(hits)}/{len(totals)} {label}s produced at least one collapse")
    print(f"  top {top} account for {covered}/{n_collapsed} "
          f"({covered / n_collapsed * 100:.0f}%) of collapses")
    for k, c in ranked[:top]:
        print(f"    {k:<28} {c:3d}/{totals[k]:<4d}  ({c / totals[k] * 100:5.2f}%)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("utts", type=Path, help="per-utterance JSONL from score --out-utts")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest with topic/speaker, joined on id")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--strata", type=int, default=3, help="length strata for the grid")
    ap.add_argument("--cells", type=int, default=3, help="CMI cells within each stratum")
    ap.add_argument("--permutations", type=int, default=10000)
    args = ap.parse_args(argv)

    rows = read_jsonl(args.utts)
    if not any("foreign_script" in r for r in rows):
        raise SystemExit(
            f"{args.utts} has no 'foreign_script' field -- it predates the "
            f"third-language detector. Re-run `score --out-utts`."
        )

    meta: Dict[str, dict] = {}
    if args.manifest:
        meta = {str(r["id"]): r for r in read_jsonl(args.manifest)}

    for r in rows:
        lang_ref = r.get("lang_ref", {})
        r["en_tokens"] = lang_ref.get(EN, 0)
        r["zh_tokens"] = lang_ref.get(ZH, 0)
        r["n_ref_tokens"] = r.get("n_ref", 0)
        m = _SPEAKER_RE.match(str(r.get("id", "")))
        r["speaker"] = m.group(1) if m else None
        r["topic"] = meta.get(str(r.get("id")), {}).get("topic")

    hit = [r for r in rows if r.get("foreign_script")]
    miss = [r for r in rows if not r.get("foreign_script")]
    print(f"{len(rows)} utterances, {len(hit)} collapsed "
          f"({len(hit) / len(rows) * 100:.2f}%)")
    if not hit:
        print("nothing collapsed; no hypothesis to test.")
        return 0

    print("\n-- group means --")
    print(f"  {'variable':<22} {'collapsed':>10} {'normal':>10} {'ratio':>7}   p")
    for key, label in (
        ("cmi_ref", "code-mixing index"),
        ("en_tokens", "English tokens"),
        ("zh_tokens", "Chinese tokens"),
        ("n_ref_tokens", "utterance length"),
    ):
        a = [r[key] for r in hit if r.get(key) is not None]
        b = [r[key] for r in miss if r.get(key) is not None]
        if not a or not b:
            continue
        p = permutation_test(a, b, n=args.permutations)
        ratio = mean(a) / mean(b) if mean(b) else float("nan")
        print(f"  {label:<22} {mean(a):10.3f} {mean(b):10.3f} {ratio:7.2f}   "
              f"{'<0.0001' if p < 1e-4 else f'{p:.4f}'}")

    # English proportion is the variable the hypothesis is actually about;
    # absolute count is the rival explanation. Both get a dose-response curve,
    # and whichever is monotonic is the one doing the work.
    print_dose_response(rows, "cmi_ref", "code-mixing index", args.bins)
    print_dose_response(rows, "en_tokens", "absolute English token count", args.bins)
    print_dose_response(rows, "n_ref_tokens", "utterance length", args.bins)

    stratified_grid(rows, strata=args.strata, cells=args.cells)

    concentration(rows, "speaker", "speaker")
    if any(r.get("topic") for r in rows):
        concentration(rows, "topic", "topic")

    print("\n-- how to read this --")
    print("  The hypothesis survives if the CMI curve rises monotonically AND the")
    print("  length curve does not, AND the collapses are spread across speakers.")
    print("  If a handful of speakers own most of them, this is acoustic, not")
    print("  linguistic. If absolute English count tracks better than CMI, the")
    print("  trigger is 'enough English', not 'a high proportion of English' --")
    print("  a different mechanism with different implications.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
