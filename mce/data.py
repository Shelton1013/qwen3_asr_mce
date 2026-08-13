"""Manifest and hypothesis IO.

Everything downstream keys off a flat list of records::

    {"id": "utt_0001", "audio": "/data/sl2/yue_en/0001.wav", "text": "我今日好busy"}

Transcription writes ``{"id": ..., "hyp": ...}`` records; scoring joins the two
on ``id``.  Keeping the two steps separate means you can re-score with different
normalisation without paying for GPU inference again -- which matters, because
you will re-score more than once.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


#: Reads use utf-8-sig so a byte-order mark left by a Windows editor or by
#: PowerShell's Out-File does not turn into a JSON parse error on line 1.
#: It decodes plain UTF-8 unchanged, so this is strictly more permissive.
_READ_ENCODING = "utf-8-sig"


def read_jsonl(path: os.PathLike | str) -> List[dict]:
    records = []
    with open(path, "r", encoding=_READ_ENCODING) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


def write_jsonl(path: os.PathLike | str, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_manifest(
    path: os.PathLike | str,
    id_key: str = "id",
    audio_key: str = "audio",
    text_key: str = "text",
) -> List[dict]:
    """Load a manifest from JSONL, JSON array, or TSV.

    Column names are configurable because every corpus disagrees: SwitchLingua
    uses ``text``, WenetSpeech-Yue ships ``txt``, Kaldi-style dumps use ``wav``
    and ``trans``.  Missing ids are filled with the row index so that a manifest
    without an id column still works.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        rows: Sequence[dict] = read_jsonl(path)
    elif suffix == ".json":
        with open(path, "r", encoding=_READ_ENCODING) as fh:
            payload = json.load(fh)
        rows = payload if isinstance(payload, list) else payload.get("data", [])
    elif suffix in (".tsv", ".csv"):
        delim = "\t" if suffix == ".tsv" else ","
        with open(path, "r", encoding=_READ_ENCODING, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=delim))
    else:
        raise ValueError(
            f"Unsupported manifest format {suffix!r}; use .jsonl, .json, .tsv or .csv"
        )

    records = []
    for i, row in enumerate(rows):
        if audio_key not in row:
            raise KeyError(
                f"{path}: record {i} has no {audio_key!r} field "
                f"(available: {sorted(row)}). Pass --audio-key to override."
            )
        records.append(
            {
                "id": str(row.get(id_key, f"utt_{i:06d}")),
                "audio": str(row[audio_key]),
                "text": str(row.get(text_key, "")),
            }
        )
    _check_unique_ids(records, str(path))
    return records


def load_hf_dataset(
    dataset: str,
    split: str = "test",
    config: Optional[str] = None,
    audio_column: str = "audio",
    text_column: str = "text",
    id_column: Optional[str] = None,
    cache_audio_dir: Optional[str] = None,
) -> List[dict]:
    """Materialise a Hugging Face audio dataset into manifest records.

    Rows whose audio is only present in memory (no on-disk ``path``) are written
    out as 16 kHz mono wavs under ``cache_audio_dir``, because every ASR runner
    here takes a file path.  Rows that already point at a real file are used in
    place, so a second run costs nothing.
    """
    try:
        from datasets import Audio, load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "HF dataset loading needs `pip install datasets soundfile`"
        ) from exc

    ds = load_dataset(dataset, config, split=split)
    ds = ds.cast_column(audio_column, Audio(sampling_rate=16000))

    cache_dir = Path(cache_audio_dir or ".mce_audio_cache") / _slug(
        f"{dataset}-{config or 'default'}-{split}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    writer = None
    for i, row in enumerate(ds):
        audio = row[audio_column]
        path = audio.get("path") if isinstance(audio, dict) else None
        if not path or not os.path.isfile(path):
            if writer is None:
                import soundfile as sf  # type: ignore

                writer = sf
            path = str(cache_dir / f"{i:06d}.wav")
            if not os.path.isfile(path):
                writer.write(path, audio["array"], audio["sampling_rate"])
        utt_id = str(row[id_column]) if id_column else f"utt_{i:06d}"
        records.append({"id": utt_id, "audio": path, "text": str(row.get(text_column, ""))})
    _check_unique_ids(records, dataset)
    return records


def join_hyps(manifest: Sequence[dict], hyps: Sequence[dict]) -> List[tuple]:
    """Join manifest references with hypotheses on ``id``.

    A missing hypothesis becomes the empty string rather than being skipped: a
    model that crashed or returned nothing on an utterance should be charged
    for it, not quietly excluded from the denominator.
    """
    hyp_by_id: Dict[str, str] = {}
    for rec in hyps:
        if "id" not in rec:
            raise KeyError("hypothesis records must contain an 'id' field")
        hyp_by_id[str(rec["id"])] = str(rec.get("hyp", rec.get("text", "")))

    missing = 0
    pairs = []
    for rec in manifest:
        utt_id = str(rec["id"])
        if utt_id not in hyp_by_id:
            missing += 1
        pairs.append((utt_id, str(rec.get("text", "")), hyp_by_id.get(utt_id, "")))
    if missing:
        print(
            f"[warn] {missing}/{len(manifest)} utterances had no hypothesis; "
            f"scored as empty output."
        )
    return pairs


def _check_unique_ids(records: Sequence[dict], source: str) -> None:
    seen = set()
    for rec in records:
        if rec["id"] in seen:
            raise ValueError(
                f"{source}: duplicate utterance id {rec['id']!r}; ids must be unique "
                f"or the hypothesis join will silently drop rows"
            )
        seen.add(rec["id"])


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def batched(items: Sequence, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])
