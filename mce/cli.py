"""Command line entry point.

Three subcommands, deliberately separable:

``transcribe``  manifest -> hypothesis JSONL   (needs a GPU)
``score``       manifest + hypotheses -> metrics   (pure CPU, seconds)
``run``         both, for convenience

Splitting them is not cosmetic. Normalisation choices -- script direction,
punctuation, number folding -- change the numbers, and you will want to rescore
several ways. Re-running inference each time would make that expensive enough
that you would stop doing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .data import join_hyps, load_hf_dataset, load_manifest, read_jsonl, write_jsonl
from .datasets import DATASETS, SPLITS, load_dataset
from .metrics import score_corpus
from .normalize import NormalizeConfig, Normalizer
from .report import format_metrics, format_worst, markdown_table


def _add_manifest_args(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group("data source")
    src.add_argument("--manifest", help="JSONL/JSON/TSV manifest of audio + reference")
    src.add_argument("--id-key", default="id")
    src.add_argument("--audio-key", default="audio")
    src.add_argument("--text-key", default="text")
    src.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="read a corpus straight from disk, skipping the manifest step",
    )
    src.add_argument("--dataset-root", help="corpus directory for --dataset")
    src.add_argument(
        "--dataset-split",
        default="test",
        choices=list(SPLITS),
        help="which split --dataset yields (default: test)",
    )
    src.add_argument(
        "--train-folders",
        type=int,
        default=112,
        help="split boundary for --dataset; must match what prepare_mce.py used",
    )
    src.add_argument("--train-ratio", type=float, default=0.7)
    src.add_argument(
        "--dataset-stratify",
        default="none",
        choices=["none", "topic"],
        help="'topic' splits within each topic group; must match what "
        "prepare_mce.py used or the two paths disagree about the test set",
    )
    src.add_argument("--dataset-encoding", default="auto")
    src.add_argument("--hf-dataset", help="Hugging Face dataset id, e.g. Shelton1013/SwitchLingua_audio")
    src.add_argument("--hf-config", default=None, help="dataset config / subset name")
    src.add_argument("--hf-split", default="test")
    src.add_argument("--hf-audio-column", default="audio")
    src.add_argument("--hf-text-column", default="text")
    src.add_argument("--hf-id-column", default=None)
    src.add_argument("--audio-cache-dir", default=None)
    src.add_argument("--limit", type=int, default=None, help="score only the first N utterances")


def _add_norm_args(p: argparse.ArgumentParser) -> None:
    n = p.add_argument_group("normalisation (applied identically to ref and hyp)")
    n.add_argument(
        "--script",
        default="t2s",
        choices=["t2s", "s2t", "none"],
        help="unify Traditional/Simplified before scoring; 'none' to skip",
    )
    n.add_argument("--keep-case", action="store_true", help="do not lowercase")
    n.add_argument("--keep-punct", action="store_true", help="do not strip punctuation")
    n.add_argument("--keep-tags", action="store_true", help="do not strip <|...|> decoder tags")
    n.add_argument(
        "--normalize-numbers",
        action="store_true",
        help="fold number words to digits (experimental; changes semantics)",
    )
    n.add_argument(
        "--drop-string",
        action="append",
        default=[],
        help="literal string to delete from both sides; repeatable",
    )


def _add_metric_args(p: argparse.ArgumentParser) -> None:
    m = p.add_argument_group("metrics")
    m.add_argument(
        "--poi-window",
        type=int,
        default=1,
        help="tokens on each side of a language boundary counted as switch region",
    )
    m.add_argument(
        "--runaway-ratio",
        type=float,
        default=2.0,
        help="hyp/ref length ratio above which an utterance is flagged as runaway",
    )
    m.add_argument("--worst", type=int, default=20, help="print the N worst utterances")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    m = p.add_argument_group("model")
    m.add_argument(
        "--model",
        required=True,
        help="registry alias (qwen3-asr-1.7b), a Hub repo id, or a local checkpoint "
        "directory -- the runner family is read from its config.json",
    )
    m.add_argument("--model-id", default=None, help="override the checkpoint id/path")
    m.add_argument(
        "--language",
        default=None,
        help="language hint. Whisper: 'zh' (default) or 'yue' (known to collapse). "
             "Qwen3-ASR: leave unset so it can switch freely.",
    )
    m.add_argument(
        "--device",
        default="auto",
        help="'auto' (cuda:0 if available, else cpu), an explicit device like "
        "cuda:2, or 'shard' to split across GPUs. Sharding is opt-in because "
        "Qwen3-ASR's audio encoder reads a buffer directly and breaks under it.",
    )
    m.add_argument("--dtype", default="auto")
    m.add_argument("--batch-size", type=int, default=1)
    m.add_argument("--max-new-tokens", type=int, default=256)
    m.add_argument("--prompt", default=None, help="Qwen3-ASR context biasing text")


def build_normalizer(args) -> Normalizer:
    return Normalizer(
        NormalizeConfig(
            script=None if args.script == "none" else args.script,
            lowercase=not args.keep_case,
            remove_punct=not args.keep_punct,
            remove_tags=not args.keep_tags,
            normalize_numbers=args.normalize_numbers,
            drop_strings=list(args.drop_string),
        )
    )


def load_records(args) -> List[dict]:
    if args.manifest:
        records = load_manifest(
            args.manifest,
            id_key=args.id_key,
            audio_key=args.audio_key,
            text_key=args.text_key,
        )
    elif args.dataset:
        if not args.dataset_root:
            raise SystemExit("--dataset requires --dataset-root")
        records = load_dataset(
            args.dataset,
            args.dataset_root,
            split=args.dataset_split,
            train_folders=args.train_folders,
            train_ratio=args.train_ratio,
            encoding=args.dataset_encoding,
            stratify=args.dataset_stratify,
            warn=lambda m: print(f"[warn] {m}"),
        )
        boundary = (
            f"stratified by topic, {args.train_ratio:.0%} train"
            if args.dataset_stratify == "topic"
            else f"first {args.train_folders} folders are train"
        )
        print(
            f"[info] {args.dataset}:{args.dataset_split} -> {len(records)} utterances "
            f"({boundary})"
        )
        if not records:
            raise SystemExit(
                f"the {args.dataset_split!r} split is empty; check --dataset-root "
                f"and --train-folders"
            )
    elif args.hf_dataset:
        records = load_hf_dataset(
            args.hf_dataset,
            split=args.hf_split,
            config=args.hf_config,
            audio_column=args.hf_audio_column,
            text_column=args.hf_text_column,
            id_column=args.hf_id_column,
            cache_audio_dir=args.audio_cache_dir,
        )
    else:
        raise SystemExit(
            "one of --manifest, --dataset (with --dataset-root), or --hf-dataset is required"
        )
    if args.limit:
        records = records[: args.limit]
    return records


def cmd_transcribe(args) -> int:
    from .models import build_model

    records = load_records(args)
    model = build_model(
        args.model,
        model_id=args.model_id,
        language=args.language,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        **({"prompt": args.prompt} if args.prompt else {}),
    )
    print(f"[info] {args.model} -> {getattr(model, 'model_id', '?')} "
          f"on {len(records)} utterances")
    hyps = model.run(records)
    write_jsonl(args.out, hyps)
    print(f"[info] wrote {len(hyps)} hypotheses to {args.out}")
    return 0


def _serialisable_config(args) -> dict:
    """The run configuration, minus the argparse plumbing.

    ``args.func`` is the bound subcommand handler and is not JSON serialisable.
    Recording the rest matters: normalisation settings change the numbers, so a
    metrics file without them cannot be compared against another one.
    """
    out = {}
    for key, value in vars(args).items():
        if key == "func":
            continue
        if isinstance(value, (str, int, float, bool, type(None), list)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def cmd_score(args) -> int:
    records = load_records(args)
    hyps = read_jsonl(args.hyp)

    # Transcription records the exception on any utterance it could not decode.
    # Those become empty hypotheses, which score as full deletions -- honest per
    # utterance, but badly misleading in aggregate if half the run crashed.
    failed = [h for h in hyps if h.get("error")]
    if failed:
        example = failed[0]["error"].splitlines()[0]
        print(
            f"!! {len(failed)}/{len(hyps)} hypotheses came from FAILED transcriptions "
            f"and are empty.\n"
            f"!! They are scored as full deletions, so every rate below is inflated.\n"
            f"!! First error: {example}\n"
        )

    pairs = join_hyps(records, hyps)

    norm = build_normalizer(args)
    normed = [(utt_id, norm(ref), norm(hyp)) for utt_id, ref, hyp in pairs]

    metrics, results = score_corpus(
        normed, poi_window=args.poi_window, runaway_ratio=args.runaway_ratio
    )

    title = args.name or Path(args.hyp).stem
    print(format_metrics(metrics, title=title))
    if args.worst:
        print()
        print(format_worst(results, k=args.worst))

    if args.out_json:
        payload = {
            "name": title,
            "config": _serialisable_config(args),
            "metrics": metrics.to_dict(),
            "n_failed_transcriptions": len(failed),
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n[info] metrics -> {args.out_json}")
    if args.out_utts:
        write_jsonl(args.out_utts, (r.to_dict() for r in results))
        print(f"[info] per-utterance detail -> {args.out_utts}")
    return 0


def cmd_run(args) -> int:
    args.out = args.hyp
    rc = cmd_transcribe(args)
    if rc != 0:
        return rc
    print()
    return cmd_score(args)


def cmd_compare(args) -> int:
    """Render a markdown comparison table from several score --out-json files."""
    rows = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows.append((payload.get("name", Path(path).stem), _rehydrate(payload["metrics"])))
    print(markdown_table(rows))
    return 0


class _MetricsView:
    """Read-only stand-in exposing the attributes ``markdown_table`` needs."""

    def __init__(self, d: dict) -> None:
        self._d = d

    def __getattr__(self, name):
        if name in self._d:
            return self._d[name]
        raise AttributeError(name)


def _rehydrate(d: dict) -> _MetricsView:
    return _MetricsView(d)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mce",
        description="Code-switching ASR evaluation for Cantonese-English (MER/PIER/omission).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="run a model over a manifest")
    _add_manifest_args(t)
    _add_model_args(t)
    t.add_argument("--out", required=True, help="hypothesis JSONL to write")
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("score", help="score hypotheses against references")
    _add_manifest_args(s)
    _add_norm_args(s)
    _add_metric_args(s)
    s.add_argument("--hyp", required=True, help="hypothesis JSONL")
    s.add_argument("--name", default=None, help="label used in the report")
    s.add_argument("--out-json", default=None, help="write corpus metrics as JSON")
    s.add_argument("--out-utts", default=None, help="write per-utterance detail as JSONL")
    s.set_defaults(func=cmd_score)

    r = sub.add_parser("run", help="transcribe then score")
    _add_manifest_args(r)
    _add_model_args(r)
    _add_norm_args(r)
    _add_metric_args(r)
    r.add_argument("--hyp", required=True, help="hypothesis JSONL to write then score")
    r.add_argument("--name", default=None)
    r.add_argument("--out-json", default=None)
    r.add_argument("--out-utts", default=None)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="markdown table from several --out-json files")
    c.add_argument("results", nargs="+")
    c.set_defaults(func=cmd_compare)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
