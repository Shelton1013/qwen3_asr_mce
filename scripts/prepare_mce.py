#!/usr/bin/env python
"""Write train/test manifests for the MCE Cantonese-English dataset.

Expected layout::

    MCE_Dataset/
      Audio/{N}_MCE/{N}_{i}.wav      i = 1..K, one utterance per file
      Text/data_{N}.csv              header "Topic,Instance", K data rows

Folder ``N`` pairs with ``data_{N}.csv``, and row ``i-1`` of that CSV is the
transcript of ``{N}_{i}.wav``. The pairing is positional, so a folder whose
counts disagree is skipped rather than emitted: a silent off-by-one would
produce a manifest that looks fine and scores nonsense.

The split is **by folder**, not by utterance -- folders are speakers.

Usage::

    python scripts/prepare_mce.py /data/MCE_Dataset --out data/mce --train-folders 112

All the corpus logic lives in :mod:`mce.datasets`; this is the manifest-writing
front end. ``mce.cli --dataset mce --dataset-root ...`` calls the same code
in-process when you do not need the split written to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Import the package from the repo root when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mce.datasets import (  # noqa: E402
    ENCODINGS,
    check_split_balance,
    clean_transcript,
    decode_csv,
    discover,
    emit_path,
    prepare_mce,
    read_folder,
    split_folders,
    stratified_split,
    topic_signature,
)
from mce.tokenizer import EN, ZH, code_mixing_index, count_langs, tokenize  # noqa: E402

# Re-exported above so this script stays the single entry point people look at
# for anything MCE-shaped, even though the implementations moved.
__all__ = [
    "ENCODINGS",
    "check_split_balance",
    "clean_transcript",
    "decode_csv",
    "discover",
    "emit_path",
    "prepare_mce",
    "read_folder",
    "split_folders",
    "stratified_split",
    "topic_signature",
    "main",
]


def summarise(records: Sequence[dict], name: str) -> str:
    speakers = sorted({r["speaker"] for r in records})
    topics: Dict[str, int] = {}
    zh = en = 0
    cmi_sum = 0.0
    n_with_en = 0
    for r in records:
        topics[r["topic"]] = topics.get(r["topic"], 0) + 1
        toks = tokenize(r["text"])
        counts = count_langs(toks)
        zh += counts[ZH]
        en += counts[EN]
        cmi_sum += code_mixing_index(toks)
        if counts[EN]:
            n_with_en += 1
    total = zh + en
    return "\n".join(
        [
            f"  {name}: {len(records)} utterances from {len(speakers)} folders",
            f"    tokens: {zh} zh + {en} en"
            + (f"  ({en / total:.1%} English)" if total else ""),
            f"    utterances containing English: {n_with_en}/{len(records)}"
            + (f" ({n_with_en / len(records):.1%})" if records else ""),
            f"    mean CMI: {cmi_sum / len(records):.3f}" if records else "    mean CMI: n/a",
            f"    topics: {dict(sorted(topics.items(), key=lambda kv: -kv[1]))}",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path, help="MCE_Dataset directory")
    ap.add_argument("--out", type=Path, default=Path("data/mce"), help="output directory")
    ap.add_argument(
        "--train-folders",
        type=int,
        default=112,
        help="number of folders (speakers) in the training split; the rest are test",
    )
    ap.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="fallback split ratio used when fewer folders exist than --train-folders",
    )
    ap.add_argument(
        "--stratify",
        default="none",
        choices=["none", "topic"],
        help="'topic' takes --train-ratio of the folders in each topic group instead "
        "of the first --train-folders overall. Use this when the corpus is ordered "
        "by collection batch, or a positional split silently becomes a topic split.",
    )
    ap.add_argument(
        "--dev-ratio",
        type=float,
        default=0.0,
        help="carve this share of the TRAINING folders into a dev split. Do error "
        "analysis and DPO negative mining on dev, never on test -- feeding test-set "
        "failure patterns back into training inflates every number that follows.",
    )
    ap.add_argument("--encoding", default="auto", help=f"csv encoding; auto tries {ENCODINGS}")
    ap.add_argument(
        "--path-prefix",
        default=None,
        help="rewrite audio paths onto this root, e.g. /data/MCE_Dataset, "
        "so manifests built on one machine resolve on another",
    )
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any problem was found")
    args = ap.parse_args(argv)

    try:
        prepared = prepare_mce(
            args.root,
            train_folders=args.train_folders,
            train_ratio=args.train_ratio,
            encoding=args.encoding,
            path_prefix=args.path_prefix,
            stratify=args.stratify,
            dev_ratio=args.dev_ratio,
            warn=lambda m: print(f"[warn] {m}"),
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    meta = prepared["meta"]
    shown = ", ".join(str(i) for i in (meta["train_folders"] + meta["test_folders"])[:10])
    print(f"[info] found {meta['n_folders']} folder(s): {shown}"
          f"{' ...' if meta['n_folders'] > 10 else ''}")

    # Skip dev entirely when none was requested, rather than leaving an empty
    # dev.jsonl around for someone to mistake for a real split.
    names = [n for n in ("train", "dev", "test") if n != "dev" or args.dev_ratio > 0]

    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = args.out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in prepared[name]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[info] wrote {len(prepared[name]):5d} utterances -> {path}")

    with open(args.out / "split.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print()
    for name in names:
        print(summarise(prepared[name], name) if prepared[name] else f"  {name}: EMPTY")
    if args.dev_ratio <= 0:
        print("\n  NOTE: no dev split. Error analysis and DPO negative mining must "
              "not run on test.jsonl -- pass --dev-ratio 0.1 to carve one from train.")

    balance = meta.get("balance_warnings") or []
    if balance:
        print("\n[BALANCE] the two splits are not comparable:")
        for b in balance:
            print(f"    - {b}")

    problems = meta["problems"]
    if problems:
        print(f"\n[warn] {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"    - {p}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more (see split.json)")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
