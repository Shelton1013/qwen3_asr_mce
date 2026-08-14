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
        "--dev-ratio",
        type=float,
        default=0.0,
        help="share of training folders held out as dev; must match prepare_mce.py",
    )
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
        help="language hint. Whisper: 'zh' (default) or 'yue' (reportedly "
        "collapses). Use 'auto' to force the model's own detection on -- required "
        "when comparing language-identification behaviour, since pinning a "
        "language suppresses the failure being measured. Qwen3-ASR: leave unset.",
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
    m.add_argument(
        "--prompt",
        default=None,
        help="Qwen3-ASR context text. Pass 'anchor' for the built-in "
        "matrix-language anchor (mce.mitigation.CANTONESE_ANCHOR).",
    )
    m.add_argument(
        "--script-constraint",
        default=None,
        choices=["zh+en", "zh", "en"],
        help="forbid decoding tokens outside these scripts (collapse mitigation "
        "baseline). 'zh+en' admits both legitimate Cantonese-English scripts and "
        "excludes any third one.",
    )
    m.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Whisper only: forbid repeating an n-gram. 0/unset keeps the model's "
        "true behaviour; 5-8 suppresses the repetition loops that otherwise let a "
        "dozen utterances dominate the corpus figure. Report both runs.",
    )
    m.add_argument(
        "--chunk-length-s",
        type=float,
        default=None,
        help="Whisper only: chunk audio longer than this. Off by default -- the "
        "long-form path is experimental for seq2seq and these utterances fit in "
        "Whisper's native 30s window.",
    )


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
            dev_ratio=args.dev_ratio,
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
    from .mitigation import CANTONESE_ANCHOR
    from .models import build_model

    records = load_records(args)
    prompt = args.prompt
    if prompt == "anchor":
        prompt = CANTONESE_ANCHOR
        print(f"[info] anchoring with matrix-language context: {prompt}")

    language = args.language
    if language == "auto":
        # Distinct from "unset": unset falls through to each runner's default,
        # which for Whisper is a pinned 'zh'. This asks for detection explicitly.
        language = ""
        print("[info] language detection left to the model (no forced token)")

    extra = {}
    if prompt:
        extra["prompt"] = prompt
    if args.script_constraint:
        extra["script_constraint"] = args.script_constraint
    if args.chunk_length_s:
        extra["chunk_length_s"] = args.chunk_length_s
    if args.no_repeat_ngram_size:
        extra["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    model = build_model(
        args.model,
        model_id=args.model_id,
        language=language,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        **extra,
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

    # A handful of degenerate outputs can carry a third of the errors. Show what
    # the corpus looks like without them, so the reader can see how much of the
    # headline is a dozen utterances rather than the model.
    from .metrics import aggregate_clean

    clean = aggregate_clean(results, runaway_ratio=args.runaway_ratio)
    dropped = metrics.n_utts - clean.n_utts
    if dropped:
        print(f"\n-- excluding {dropped} degenerate utterances "
              f"({dropped / metrics.n_utts * 100:.2f}%: runaway or foreign-script) --")
        for label, full, trimmed in (
            ("MER", metrics.mer, clean.mer),
            ("CER_zh", metrics.cer_zh, clean.cer_zh),
            ("WER_en", metrics.wer_en, clean.wer_en),
            ("PIER", metrics.pier, clean.pier),
            ("insertions", metrics.ins_rate, clean.ins_rate),
        ):
            if full is None or trimmed is None:
                continue
            print(f"  {label:<12} {full * 100:7.2f} %  ->  {trimmed * 100:7.2f} %  "
                  f"({(trimmed - full) * 100:+.2f})")
        print("  Report both. The trimmed figure is not the model's score -- the "
              "dropped\n  utterances are real failures -- but it separates 'bad at "
              "this task' from\n  'catastrophic on a dozen inputs'.")
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


def cmd_analyze(args) -> int:
    """Tabulate what actually went wrong, from a --out-utts dump."""
    from .analyze import build_inventory, format_inventory, inventory_to_dict

    rows = read_jsonl(args.utts)
    missing = [r for r in rows if "ref" not in r or "hyp" not in r]
    if missing:
        raise SystemExit(
            f"{args.utts}: {len(missing)} rows lack 'ref'/'hyp'. This command reads "
            f"the per-utterance dump written by `score --out-utts`, not a hypothesis "
            f"file."
        )
    if "test" in Path(args.utts).name and not args.allow_test:
        print(
            "!! This dump looks like it came from the TEST set. Mining test-set "
            "failures\n!! and feeding them into training is contamination -- use dev. "
            "Pass\n!! --allow-test if you really mean to read it.\n"
        )
        return 1

    inv = build_inventory(
        [(r["ref"], r["hyp"]) for r in rows], poi_window=args.poi_window
    )
    report = format_inventory(inv, top=args.top, min_count=args.min_count)
    print(report)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(inventory_to_dict(inv, args.min_count), fh, ensure_ascii=False, indent=2)
        print(f"\n[info] pairs -> {args.out_json}")
    if args.out_text:
        Path(args.out_text).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_text).write_text(report, encoding="utf-8")
        print(f"[info] report -> {args.out_text}")
    return 0


def cmd_check_alignment(args) -> int:
    """Find utterances paired with the wrong reference."""
    from .alignment_check import (
        find_misalignments,
        find_swapped_pairs,
        format_report,
    )

    rows = read_jsonl(args.utts)
    if any("ref" not in r or "hyp" not in r for r in rows):
        raise SystemExit(
            f"{args.utts}: rows must contain 'ref' and 'hyp' -- pass the dump from "
            f"`score --out-utts`."
        )
    found = find_misalignments(
        rows,
        window=args.window,
        max_own_mer=args.min_own_mer,
        ratio=args.ratio,
        max_best_mer=args.max_match_mer,
    )
    report = format_report(rows, found, top=args.top)
    print(report)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        pairs = {(a.index, b.index) for a, b in find_swapped_pairs(found)}
        payload = [
            {
                "id": m.utt_id,
                "matches": m.best_id,
                "own_mer": m.own_mer,
                "match_mer": m.best_mer,
                "mutual": any(m.index in p for p in pairs),
                "ref": m.ref,
                "hyp": m.hyp,
            }
            for m in found
        ]
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n[info] {len(payload)} suspect rows -> {args.out_json}")
    return 0


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

    a = sub.add_parser(
        "analyze",
        help="tabulate substitution pairs from a score --out-utts dump (run it on dev)",
    )
    a.add_argument("--utts", required=True, help="per-utterance JSONL from score --out-utts")
    a.add_argument("--top", type=int, default=30, help="rows per table")
    a.add_argument("--min-count", type=int, default=2, help="hide pairs seen fewer times")
    a.add_argument("--poi-window", type=int, default=1)
    a.add_argument("--out-json", default=None, help="write the pairs for downstream use")
    a.add_argument("--out-text", default=None, help="write the report")
    a.add_argument(
        "--allow-test",
        action="store_true",
        help="permit analysing a dump whose filename says 'test'",
    )
    a.set_defaults(func=cmd_analyze)

    k = sub.add_parser(
        "check-alignment",
        help="find utterances whose audio is paired with the wrong reference",
    )
    k.add_argument("--utts", required=True, help="per-utterance JSONL from score --out-utts")
    k.add_argument("--window", type=int, default=5, help="rows either side to compare")
    k.add_argument("--min-own-mer", type=float, default=0.5,
                   help="only inspect rows scoring at least this badly")
    k.add_argument("--ratio", type=float, default=0.5,
                   help="a neighbour must fit at least this much better, proportionally")
    k.add_argument("--max-match-mer", type=float, default=0.5,
                   help="and this well in absolute terms")
    k.add_argument("--top", type=int, default=20)
    k.add_argument("--out-json", default=None, help="write the suspect rows")
    k.set_defaults(func=cmd_check_alignment)

    c = sub.add_parser("compare", help="markdown table from several --out-json files")
    c.add_argument("results", nargs="+")
    c.set_defaults(func=cmd_compare)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
