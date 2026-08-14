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

SPLITS = ("train", "dev", "test", "all")


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


def topic_signature(records: Sequence[dict]) -> frozenset:
    """The set of topics a folder covers.

    Corpora collected in batches give each batch its own topic list, so this
    signature identifies which batch a folder belongs to without needing a batch
    column. Folders sharing a signature are interchangeable; folders with
    different signatures are not.
    """
    return frozenset(r["topic"] for r in records)


def stratified_split(
    per_folder: Sequence[Tuple[int, List[dict]]],
    train_ratio: float = 0.7,
    warn: Optional[Callable[[str], None]] = None,
) -> Tuple[List[int], List[int]]:
    """Split folders by speaker, taking ``train_ratio`` of *each* topic group.

    A positional "first N folders" split silently becomes a topic split when the
    corpus is ordered by collection batch: the held-out folders then cover
    subjects the model never saw, at a different code-switching density. The
    resulting number measures cross-topic generalisation, not recognition, and
    no later analysis can separate the two.
    """
    say = warn or (lambda _m: None)
    groups: Dict[frozenset, List[int]] = {}
    for idx, records in per_folder:
        groups.setdefault(topic_signature(records), []).append(idx)

    if len(groups) > 1:
        say(
            f"corpus contains {len(groups)} distinct topic groups "
            f"({', '.join(str(len(v)) + ' folders' for v in groups.values())}); "
            f"stratifying so both splits see all of them."
        )

    train: List[int] = []
    test: List[int] = []
    # Sort groups by their smallest folder index so the assignment is
    # deterministic regardless of dict iteration order.
    for _, indices in sorted(groups.items(), key=lambda kv: min(kv[1])):
        indices = sorted(indices)
        n_train = max(1, round(len(indices) * train_ratio)) if len(indices) > 1 else len(indices)
        train.extend(indices[:n_train])
        test.extend(indices[n_train:])
        if len(indices) == 1:
            say(
                f"topic group with folders {indices} has a single folder; it "
                f"cannot appear in both splits and was assigned to train."
            )
    return sorted(train), sorted(test)


def check_split_balance(
    train: Sequence[dict],
    test: Sequence[dict],
    cmi_tolerance: float = 0.20,
    topic_tolerance: float = 0.05,
) -> List[str]:
    """Report ways the two splits are not comparable.

    A test set that differs from training in topic coverage or code-switching
    density will move the score for reasons that have nothing to do with the
    model, and the confound is invisible in the headline number.
    """
    from .tokenizer import EN, ZH, code_mixing_index, count_langs, tokenize

    if not train or not test:
        return []

    warnings: List[str] = []

    def profile(records):
        topics: Dict[str, int] = {}
        zh = en = 0
        cmi = 0.0
        for r in records:
            topics[r["topic"]] = topics.get(r["topic"], 0) + 1
            toks = tokenize(r["text"])
            counts = count_langs(toks)
            zh += counts[ZH]
            en += counts[EN]
            cmi += code_mixing_index(toks)
        n = len(records)
        shares = {k: v / n for k, v in topics.items()}
        return shares, cmi / n, (en / (zh + en) if zh + en else 0.0)

    train_topics, train_cmi, train_en = profile(train)
    test_topics, test_cmi, test_en = profile(test)

    missing = sorted(
        t for t, share in train_topics.items()
        if share >= topic_tolerance and test_topics.get(t, 0.0) < topic_tolerance / 5
    )
    unseen = sorted(
        t for t, share in test_topics.items()
        if share >= topic_tolerance and train_topics.get(t, 0.0) < topic_tolerance / 5
    )
    if missing:
        warnings.append(
            f"{len(missing)} topic(s) common in train are absent from test: "
            f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}"
        )
    if unseen:
        warnings.append(
            f"{len(unseen)} topic(s) common in test are absent from train: "
            f"{', '.join(unseen[:6])}{' ...' if len(unseen) > 6 else ''}"
        )

    if train_cmi and abs(train_cmi - test_cmi) / train_cmi > cmi_tolerance:
        warnings.append(
            f"code-switching density differs: mean CMI {train_cmi:.3f} (train) vs "
            f"{test_cmi:.3f} (test), a {abs(train_cmi - test_cmi) / train_cmi:.0%} gap. "
            f"The test set is {'easier' if test_cmi < train_cmi else 'harder'} than "
            f"training in the one dimension this task is about."
        )
    if train_en and abs(train_en - test_en) / train_en > cmi_tolerance:
        warnings.append(
            f"English token share differs: {train_en:.1%} (train) vs {test_en:.1%} (test)."
        )

    if warnings:
        warnings.append(
            "consider --stratify topic; a positional split becomes a topic split "
            "when the corpus is ordered by collection batch."
        )
    return warnings


def prepare_mce(
    root: Path,
    train_folders: int = 112,
    train_ratio: float = 0.7,
    encoding: str = "auto",
    path_prefix: Optional[str] = None,
    stratify: str = "none",
    dev_ratio: float = 0.0,
    warn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Read the whole corpus and return the splits plus metadata.

    Returns ``{"train": [...], "dev": [...], "test": [...], "meta": {...}}``
    with ``audio`` already stringified.

    ``stratify="topic"`` takes ``train_ratio`` of the folders in each topic
    group instead of the first ``train_folders`` overall, and ignores
    ``train_folders``. Balance is checked either way; the check is what tells
    you a positional split has quietly turned into a topic split.

    ``dev_ratio`` carves a development split out of the *training* folders,
    speaker-disjoint from both others. Without one there is nowhere to do error
    analysis: mining the test set for failure patterns and feeding them back
    into training is test-set contamination, and every number measured
    afterwards is inflated by an amount nobody can estimate.
    """
    if stratify not in ("none", "topic"):
        raise ValueError(f"stratify must be 'none' or 'topic', got {stratify!r}")
    say = warn or (lambda _m: None)
    root = Path(root)
    folders, problems = discover(root)
    if not folders:
        raise FileNotFoundError(
            f"no {{N}}_MCE folders paired with data_{{N}}.csv under {root}"
        )

    # Read everything up front: stratification needs each folder's topics, and
    # reading twice would let the two passes disagree.
    per_folder: List[Tuple[int, List[dict]]] = []
    for idx, audio_dir, csv_path in folders:
        recs, probs = read_folder(audio_dir, csv_path, encoding)
        problems.extend(probs)
        per_folder.append((idx, recs))

    if stratify == "topic":
        if train_folders is not None:
            say(f"--stratify topic uses train_ratio={train_ratio:.0%}; train_folders is ignored.")
        train_idx, test_idx = stratified_split(per_folder, train_ratio, warn=say)
    else:
        train_part, test_part = split_folders(per_folder, train_folders, train_ratio, warn=say)
        train_idx = [i for i, _ in train_part]
        test_idx = [i for i, _ in test_part]

    train_idx, dev_idx = _carve_dev(per_folder, train_idx, dev_ratio, stratify, say)

    by_idx = dict(per_folder)
    splits: Dict[str, List[dict]] = {}
    for name, chosen in (("train", train_idx), ("dev", dev_idx), ("test", test_idx)):
        records: List[dict] = []
        for idx in chosen:
            for r in by_idx[idx]:
                records.append({**r, "audio": emit_path(r["audio"], root, path_prefix)})
        splits[name] = records

    # Returned rather than warned: callers show it after the split summaries,
    # where the reader has the numbers in front of them.
    balance = check_split_balance(splits["train"], splits["test"])
    balance += [
        f"train/dev: {w}" for w in check_split_balance(splits["train"], splits["dev"])
    ]

    splits["meta"] = {
        "root": str(root),
        "n_folders": len(folders),
        "stratify": stratify,
        "dev_ratio": dev_ratio,
        "train_folders": train_idx,
        "dev_folders": dev_idx,
        "test_folders": test_idx,
        "n_train_utts": len(splits["train"]),
        "n_dev_utts": len(splits["dev"]),
        "n_test_utts": len(splits["test"]),
        "path_prefix": path_prefix,
        "problems": problems,
        "balance_warnings": balance,
    }
    return splits


def _carve_dev(
    per_folder: Sequence[Tuple[int, List[dict]]],
    train_idx: Sequence[int],
    dev_ratio: float,
    stratify: str,
    say: Callable[[str], None],
) -> Tuple[List[int], List[int]]:
    """Split a dev set off the training folders, keeping speakers disjoint."""
    if dev_ratio <= 0:
        return list(train_idx), []
    if not 0 < dev_ratio < 1:
        raise ValueError(f"dev_ratio must be in (0, 1), got {dev_ratio}")

    chosen = set(train_idx)
    train_only = [(i, recs) for i, recs in per_folder if i in chosen]
    keep = 1.0 - dev_ratio

    if stratify == "topic":
        train, dev = stratified_split(train_only, keep, warn=None)
    else:
        n_keep = max(1, round(len(train_only) * keep))
        train = [i for i, _ in train_only[:n_keep]]
        dev = [i for i, _ in train_only[n_keep:]]
    say(
        f"carved {len(dev)} dev folder(s) out of {len(train_only)} training folders "
        f"({dev_ratio:.0%}); use dev for error analysis so the test set stays unread."
    )
    return sorted(train), sorted(dev)


def load_mce(root, split: str = "test", **kwargs) -> List[dict]:
    """One split of the MCE corpus, read straight from disk.

    Produces byte-identical records to the corresponding ``prepare_mce.py``
    manifest -- same ids, same order, same split boundary -- so switching
    between ``--manifest`` and ``--dataset mce`` never changes a score.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    prepared = prepare_mce(Path(root), **kwargs)
    # Direct mode has no summary section to hang these off, so surface them
    # through the same channel as the other warnings rather than dropping them.
    warn = kwargs.get("warn")
    if warn:
        for message in prepared["meta"]["balance_warnings"]:
            warn(f"BALANCE: {message}")
    if split == "all":
        return prepared["train"] + prepared["dev"] + prepared["test"]
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
