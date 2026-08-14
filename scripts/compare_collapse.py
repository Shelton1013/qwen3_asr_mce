#!/usr/bin/env python
"""Compare language-identification collapse across models.

One model is an anecdote. The question this answers is whether the CMI threshold
is a property of code-switching data or a quirk of one checkpoint:

* does the threshold **move right** as the model gets stronger (a general
  mechanism, modulated by capability), or does it **disappear** (a weak-model
  bug worth one paragraph)?
* does each model collapse into the *same* language, or its own? A shared
  attractor points at something about the data; a per-model attractor points at
  the training distribution, which is the reading the mechanism predicts.

Usage::

    python scripts/compare_collapse.py \\
        exp/mce_stage0/qwen3-asr-1.7b.utts.jsonl \\
        exp/mce_stage0/qwen3-asr-0.6b.utts.jsonl \\
        exp/mce_stage0/whisper-large-v3-zh.utts.jsonl \\
        exp/mce_stage0/sensevoice.utts.jsonl
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mce.data import read_jsonl  # noqa: E402

#: Unicode name prefixes that identify a writing system, longest first so that
#: "CJK" does not swallow "CJK COMPATIBILITY".
_SCRIPT_PREFIXES = (
    "THAI", "CYRILLIC", "DEVANAGARI", "ARABIC", "HEBREW", "GREEK", "HANGUL",
    "HIRAGANA", "KATAKANA", "BENGALI", "TAMIL", "TELUGU", "GUJARATI", "KANNADA",
    "MALAYALAM", "SINHALA", "MYANMAR", "KHMER", "LAO", "TIBETAN", "GEORGIAN",
    "ARMENIAN", "ETHIOPIC", "CHEROKEE", "MONGOLIAN",
)


def dominant_foreign_script(text: str) -> Optional[str]:
    """Which writing system a collapsed hypothesis actually landed in."""
    counts: Counter = Counter()
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for prefix in _SCRIPT_PREFIXES:
            if name.startswith(prefix):
                counts[prefix] += 1
                break
    return counts.most_common(1)[0][0] if counts else None


def threshold(rows: Sequence[dict], bins: int = 10) -> Tuple[Optional[float], float]:
    """(lowest CMI at which collapse appears, collapse rate in the top bin).

    The threshold is read as the lower edge of the first quantile bin containing
    any collapse -- deliberately crude, because with 43 events across 4164
    utterances a fitted breakpoint would be false precision.
    """
    usable = sorted(
        (r for r in rows if r.get("cmi_ref") is not None), key=lambda r: r["cmi_ref"]
    )
    if not usable:
        return None, 0.0
    size = len(usable) / bins
    onset = None
    top_rate = 0.0
    for b in range(bins):
        chunk = usable[int(b * size) : int((b + 1) * size)]
        if not chunk:
            continue
        hits = sum(1 for r in chunk if r.get("foreign_script"))
        if hits and onset is None:
            onset = chunk[0]["cmi_ref"]
        if b == bins - 1:
            top_rate = hits / len(chunk)
    return onset, top_rate


def summarise(path: Path, bins: int) -> Dict:
    rows = read_jsonl(path)
    collapsed = [r for r in rows if r.get("foreign_script")]
    onset, top_rate = threshold(rows, bins)
    scripts: Counter = Counter()
    for r in collapsed:
        script = dominant_foreign_script(r.get("hyp", ""))
        if script:
            scripts[script] += 1
    attractor = scripts.most_common(1)[0] if scripts else None
    return {
        "name": path.stem.replace(".utts", ""),
        "n": len(rows),
        "collapsed": len(collapsed),
        "rate": len(collapsed) / len(rows) if rows else 0.0,
        "onset": onset,
        "top_rate": top_rate,
        "attractor": attractor[0] if attractor else "-",
        "attractor_share": (attractor[1] / len(collapsed)) if attractor and collapsed else 0.0,
        "scripts": scripts,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("utts", nargs="+", type=Path)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = ap.parse_args(argv)

    rows = []
    for path in args.utts:
        try:
            rows.append(summarise(path, args.bins))
        except FileNotFoundError:
            print(f"[warn] missing: {path}")
    if not rows:
        raise SystemExit("nothing to compare")

    header = ["model", "n", "collapsed", "rate %", "CMI onset", "top-decile %",
              "attractor", "share %"]
    body = [
        [
            r["name"],
            str(r["n"]),
            str(r["collapsed"]),
            f"{r['rate'] * 100:.2f}",
            f"{r['onset']:.3f}" if r["onset"] is not None else "-",
            f"{r['top_rate'] * 100:.2f}",
            r["attractor"],
            f"{r['attractor_share'] * 100:.0f}" if r["collapsed"] else "-",
        ]
        for r in rows
    ]

    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for line in body:
            print("| " + " | ".join(line) + " |")
    else:
        widths = [max(len(header[i]), *(len(b[i]) for b in body)) for i in range(len(header))]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        print("  ".join("-" * w for w in widths))
        for line in body:
            print("  ".join(c.ljust(w) for c, w in zip(line, widths)))

    print("\n-- collapse targets in full --")
    for r in rows:
        if r["scripts"]:
            detail = ", ".join(f"{k} x{v}" for k, v in r["scripts"].most_common())
            print(f"  {r['name']}: {detail}")
        else:
            print(f"  {r['name']}: none")

    print("\n-- how to read this --")
    print("  A threshold that moves right with model strength is a general")
    print("  mechanism; one that vanishes is a single checkpoint's bug. Different")
    print("  attractors across models support the reading that the collapse target")
    print("  comes from the training distribution rather than from the audio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
