#!/usr/bin/env python
"""Build train/test manifests from the MCE Cantonese-English dataset.

Expected layout::

    MCE_Dataset/
      Audio/{N}_MCE/{N}_{i}.wav      i = 1..K, one utterance per file
      Text/data_{N}.csv              header "Topic,Instance", K data rows

Folder ``N`` pairs with ``data_{N}.csv``, and row ``i-1`` of that CSV is the
transcript of ``{N}_{i}.wav``. The pairing is positional, so this script refuses
to emit anything when the counts disagree: a silent off-by-one would produce a
manifest that looks fine and scores nonsense.

The split is **by folder**, not by utterance. Folders are speakers, and splitting
utterances would put the same voice on both sides -- a fine-tuned model would
then be scored partly on speakers it trained on, and the test number would be
optimistic in a way no amount of later analysis can recover.

Usage::

    python scripts/prepare_mce.py F:/csasr/MCE_Dataset --out data/mce \\
        --train-folders 112 --path-prefix /data/MCE_Dataset
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Import the package from the repo root when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mce.tokenizer import EN, ZH, code_mixing_index, count_langs, tokenize  # noqa: E402

#: Tried in order. UTF-8 is strict so it fails fast on legacy bytes; GB18030 is a
#: superset of GBK and decodes the Traditional-Chinese Cantonese in this corpus.
#: Note the files are GBK-encoded despite holding Traditional characters, so a
#: Big5 guess would have to come after, not before.
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5hkscs", "big5", "cp950")

_FOLDER_RE = re.compile(r"^(\d+)_MCE$")
_WAV_RE = re.compile(r"^(\d+)_(\d+)\.wav$", re.IGNORECASE)
_CSV_RE = re.compile(r"^data_(\d+)\.csv$", re.IGNORECASE)


def decode_csv(path: Path, encoding: str = "auto") -> str:
    if encoding != "auto":
        return path.read_bytes().decode(encoding)
    raw = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "auto", raw, 0, 1, f"{path} decoded with none of {ENCODINGS}"
    )


def clean_transcript(text: str) -> str:
    r"""Strip the wrapping quotes the corpus stores around every instance.

    Each field is written with three doubled quote characters on each side in
    the raw file; ``csv`` unescapes that down to a single literal quote at each
    end, which is not part of the speech and would otherwise be scored as a
    token.
    """
    text = text.strip()
    while len(text) >= 2 and text[0] in '"“”' and text[-1] in '"“”':
        text = text[1:-1].strip()
    return " ".join(text.split())


def read_folder(
    audio_dir: Path, csv_path: Path, encoding: str = "auto"
) -> Tuple[List[dict], List[str]]:
    """Return (records, problems) for one speaker folder."""
    problems: List[str] = []

    wavs = sorted(
        (p for p in audio_dir.iterdir() if _WAV_RE.match(p.name)),
        key=lambda p: int(_WAV_RE.match(p.name).group(2)),
    )
    text = decode_csv(csv_path, encoding)
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return [], [f"{csv_path.name}: empty file"]

    header = [h.strip().lower() for h in rows[0]]
    try:
        topic_col = header.index("topic")
        inst_col = header.index("instance")
    except ValueError:
        return [], [f"{csv_path.name}: header {rows[0]} lacks Topic/Instance"]

    data = [r for r in rows[1:] if any(c.strip() for c in r)]

    if len(data) != len(wavs):
        problems.append(
            f"{audio_dir.name}: {len(wavs)} wav files but {len(data)} csv rows -- "
            f"positional pairing is unsafe, folder skipped"
        )
        return [], problems

    speaker = audio_dir.name
    records = []
    for i, (wav, row) in enumerate(zip(wavs, data), start=1):
        if len(row) <= max(topic_col, inst_col):
            problems.append(f"{csv_path.name}: row {i} has {len(row)} columns")
            continue
        transcript = clean_transcript(row[inst_col])
        if not transcript:
            problems.append(f"{csv_path.name}: row {i} has an empty transcript")
            continue
        records.append(
            {
                "id": f"{speaker}_{wav.stem}",
                "audio": wav,
                "text": transcript,
                "topic": row[topic_col].strip(),
                "speaker": speaker,
            }
        )
    return records, problems


def discover(root: Path) -> Tuple[List[Tuple[int, Path, Path]], List[str]]:
    """Find (index, audio_dir, csv_path) triples, sorted numerically."""
    problems: List[str] = []
    audio_root, text_root = root / "Audio", root / "Text"
    if not audio_root.is_dir():
        raise SystemExit(f"{audio_root} does not exist")
    if not text_root.is_dir():
        raise SystemExit(f"{text_root} does not exist")

    csvs: Dict[int, Path] = {}
    for p in text_root.iterdir():
        m = _CSV_RE.match(p.name)
        if m:
            csvs[int(m.group(1))] = p

    found = []
    for p in sorted(audio_root.iterdir()):
        m = _FOLDER_RE.match(p.name)
        if not (m and p.is_dir()):
            continue
        idx = int(m.group(1))
        if idx not in csvs:
            problems.append(f"Audio/{p.name}: no matching Text/data_{idx}.csv")
            continue
        found.append((idx, p, csvs[idx]))

    for idx, path in sorted(csvs.items()):
        if not any(i == idx for i, _, _ in found):
            problems.append(f"Text/{path.name}: no matching Audio/{idx}_MCE/")

    # Numeric sort, so 2_MCE precedes 10_MCE. Lexicographic order here would
    # silently change which folders land in the training split.
    found.sort(key=lambda t: t[0])
    return found, problems


def emit_path(wav: Path, root: Path, prefix: Optional[str]) -> str:
    """Absolute POSIX path, optionally rebased onto the server's dataset root."""
    if prefix:
        rel = wav.resolve().relative_to(root.resolve())
        return f"{prefix.rstrip('/')}/{rel.as_posix()}"
    return wav.resolve().as_posix()


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
    lines = [
        f"  {name}: {len(records)} utterances from {len(speakers)} folders",
        f"    tokens: {zh} zh + {en} en"
        + (f"  ({en / total:.1%} English)" if total else ""),
        f"    utterances containing English: {n_with_en}/{len(records)}"
        + (f" ({n_with_en / len(records):.1%})" if records else ""),
        f"    mean CMI: {cmi_sum / len(records):.3f}" if records else "    mean CMI: n/a",
        f"    topics: {dict(sorted(topics.items(), key=lambda kv: -kv[1]))}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--encoding", default="auto", help=f"csv encoding; auto tries {ENCODINGS}")
    ap.add_argument(
        "--path-prefix",
        default=None,
        help="rewrite audio paths onto this root, e.g. /data/MCE_Dataset, "
             "so manifests built on Windows work on the GPU server",
    )
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any problem was found")
    args = ap.parse_args(argv)

    folders, problems = discover(args.root)
    if not folders:
        raise SystemExit(f"no {{N}}_MCE folders paired with data_{{N}}.csv under {args.root}")

    print(f"[info] found {len(folders)} folder(s): "
          f"{', '.join(str(i) for i, _, _ in folders[:10])}"
          f"{' ...' if len(folders) > 10 else ''}")

    n_train = args.train_folders
    if len(folders) < args.train_folders:
        n_train = max(1, round(len(folders) * args.train_ratio))
        print(
            f"[warn] only {len(folders)} folder(s) present, fewer than "
            f"--train-folders {args.train_folders}. Falling back to a "
            f"{args.train_ratio:.0%} split -> {n_train} train / "
            f"{len(folders) - n_train} test."
        )
        if len(folders) == 1:
            print(
                "[warn] a single folder cannot be split by speaker. Everything "
                "goes to train and the test manifest will be EMPTY. Copy the "
                "remaining folders before using this for evaluation."
            )

    train_folders = folders[:n_train]
    test_folders = folders[n_train:]

    splits = {}
    for name, chosen in (("train", train_folders), ("test", test_folders)):
        records: List[dict] = []
        for _, audio_dir, csv_path in chosen:
            recs, probs = read_folder(audio_dir, csv_path, args.encoding)
            records.extend(recs)
            problems.extend(probs)
        splits[name] = records

    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        path = args.out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "audio": emit_path(r["audio"], args.root, args.path_prefix),
                            "text": r["text"],
                            "topic": r["topic"],
                            "speaker": r["speaker"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[info] wrote {len(records):5d} utterances -> {path}")

    meta = {
        "root": str(args.root),
        "n_folders": len(folders),
        "train_folders": [i for i, _, _ in train_folders],
        "test_folders": [i for i, _, _ in test_folders],
        "n_train_utts": len(splits["train"]),
        "n_test_utts": len(splits["test"]),
        "path_prefix": args.path_prefix,
        "problems": problems,
    }
    with open(args.out / "split.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print()
    for name, records in splits.items():
        if records:
            print(summarise(records, name))
        else:
            print(f"  {name}: EMPTY")

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
