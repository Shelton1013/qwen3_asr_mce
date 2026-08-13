"""Direct corpus readers, so a manifest file is optional rather than mandatory.

A manifest is still the right thing when a split has to be frozen and shared
across processes -- fine-tuning reads ``train.jsonl`` hours before evaluation
reads ``test.jsonl``, and the boundary between them must not be recomputed
differently in between.  But for a single evaluation pass, materialising a file
just to read it back is pure ceremony.

This module holds the corpus logic; ``scripts/prepare_mce.py`` wraps it to write
manifests, and ``mce.cli --dataset`` calls it in-process.  Both go through the
same code, so the two paths cannot drift apart and produce different splits.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Tried in order. UTF-8 is strict so it fails fast on legacy bytes; GB18030 is
#: a superset of GBK and decodes the Traditional-Chinese Cantonese in the MCE
#: corpus. The files are GBK-encoded *despite* holding Traditional characters,
#: so a Big5 guess must come after, not before -- it would decode to mojibake.
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5hkscs", "big5", "cp950")

_FOLDER_RE = re.compile(r"^(\d+)_MCE$")
_WAV_RE = re.compile(r"^(\d+)_(\d+)\.wav$", re.IGNORECASE)
_CSV_RE = re.compile(r"^data_(\d+)\.csv$", re.IGNORECASE)

SPLITS = ("train", "test", "all")


# --------------------------------------------------------------------------
# MCE corpus
# --------------------------------------------------------------------------


def decode_csv(path: Path, encoding: str = "auto") -> str:
    if encoding != "auto":
        return path.read_bytes().decode(encoding)
    raw = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("auto", raw, 0, 1, f"{path} decoded with none of {ENCODINGS}")


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
    """Return ``(records, problems)`` for one speaker folder.

    Records carry ``audio`` as a :class:`~pathlib.Path`; callers turn it into a
    string with :func:`emit_path` once they know whether to rebase it.
    """
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
    """Find ``(index, audio_dir, csv_path)`` triples, sorted numerically."""
    problems: List[str] = []
    audio_root, text_root = root / "Audio", root / "Text"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"{audio_root} does not exist")
    if not text_root.is_dir():
        raise FileNotFoundError(f"{text_root} does not exist")

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
    """Absolute POSIX path, optionally rebased onto another dataset root."""
    if prefix:
        rel = wav.resolve().relative_to(root.resolve())
        return f"{prefix.rstrip('/')}/{rel.as_posix()}"
    return wav.resolve().as_posix()


def split_folders(
    folders: Sequence[tuple],
    train_folders: int = 112,
    train_ratio: float = 0.7,
    warn: Optional[Callable[[str], None]] = None,
) -> Tuple[list, list]:
    """Split the folder list by speaker at ``train_folders``.

    Folders are speakers. Splitting utterances instead would put the same voice
    on both sides, and a fine-tuned model would then be scored partly on
    speakers it trained on -- optimism that no later analysis can undo.
    """
    say = warn or (lambda _m: None)
    n_train = train_folders
    if len(folders) < train_folders:
        n_train = max(1, round(len(folders) * train_ratio))
        say(
            f"only {len(folders)} folder(s) present, fewer than train_folders="
            f"{train_folders}. Falling back to a {train_ratio:.0%} split -> "
            f"{n_train} train / {len(folders) - n_train} test."
        )
        if len(folders) == 1:
            say(
                "a single folder cannot be split by speaker. Everything goes to "
                "train and the test split will be EMPTY. Copy the remaining "
                "folders before using this for evaluation."
            )
    return list(folders[:n_train]), list(folders[n_train:])


def prepare_mce(
    root: Path,
    train_folders: int = 112,
    train_ratio: float = 0.7,
    encoding: str = "auto",
    path_prefix: Optional[str] = None,
    warn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Read the whole corpus and return both splits plus metadata.

    Returns ``{"train": [...], "test": [...], "meta": {...}}`` with ``audio``
    already stringified.
    """
    root = Path(root)
    folders, problems = discover(root)
    if not folders:
        raise FileNotFoundError(
            f"no {{N}}_MCE folders paired with data_{{N}}.csv under {root}"
        )

    train, test = split_folders(folders, train_folders, train_ratio, warn=warn)

    splits: Dict[str, List[dict]] = {}
    for name, chosen in (("train", train), ("test", test)):
        records: List[dict] = []
        for _, audio_dir, csv_path in chosen:
            recs, probs = read_folder(audio_dir, csv_path, encoding)
            problems.extend(probs)
            for r in recs:
                records.append({**r, "audio": emit_path(r["audio"], root, path_prefix)})
        splits[name] = records

    splits["meta"] = {
        "root": str(root),
        "n_folders": len(folders),
        "train_folders": [i for i, _, _ in train],
        "test_folders": [i for i, _, _ in test],
        "n_train_utts": len(splits["train"]),
        "n_test_utts": len(splits["test"]),
        "path_prefix": path_prefix,
        "problems": problems,
    }
    return splits


def load_mce(root, split: str = "test", **kwargs) -> List[dict]:
    """One split of the MCE corpus, read straight from disk.

    Produces byte-identical records to the corresponding ``prepare_mce.py``
    manifest -- same ids, same order, same split boundary -- so switching
    between ``--manifest`` and ``--dataset mce`` never changes a score.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    prepared = prepare_mce(Path(root), **kwargs)
    if split == "all":
        return prepared["train"] + prepared["test"]
    return prepared[split]


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

#: name -> loader(root, split=..., **kwargs) -> list of manifest records
DATASETS: Dict[str, Callable[..., List[dict]]] = {
    "mce": load_mce,
}


def load_dataset(name: str, root, split: str = "test", **kwargs) -> List[dict]:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(DATASETS))}")
    return DATASETS[name](root, split=split, **kwargs)
