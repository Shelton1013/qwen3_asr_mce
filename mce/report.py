"""Rendering of corpus metrics, plus the diagnostic reading of them.

The tables here always print the I/D/S breakdown next to the rates.  That is
deliberate: the breakdown is what turns "the number is bad" into "the number is
bad *for this reason*", and it is the fastest route from a broken evaluation to
a fixed one.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .metrics import CorpusMetrics, UtteranceResult
from .tokenizer import EN, NUM, OTHER, ZH


def _pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}"


def _num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_metrics(metrics: CorpusMetrics, title: str = "") -> str:
    """A single model's full metric block, as plain text."""
    m = metrics
    lines: List[str] = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(
        f"utterances={m.n_utts}  ref_tokens={m.n_ref_tokens}  hyp_tokens={m.n_hyp_tokens}"
    )
    lines.append("")
    lines.append("-- primary --")
    lines.append(f"  MER            {_pct(m.mer)} %")
    lines.append(f"  CER_zh         {_pct(m.cer_zh)} %   (over {m.lang_ref[ZH]} zh tokens)")
    lines.append(f"  WER_en         {_pct(m.wer_en)} %   (over {m.lang_ref[EN]} en tokens)")
    lines.append(f"  PIER           {_pct(m.pier)} %   (over {m.n_poi} switch-region tokens)")
    lines.append("")
    lines.append("-- error decomposition (share of reference tokens) --")
    lines.append(f"  substitutions  {_pct(m.sub_rate)} %   ({m.subs})")
    lines.append(f"  deletions      {_pct(m.del_rate)} %   ({m.dels})")
    lines.append(f"  insertions     {_pct(m.ins_rate)} %   ({m.ins})")
    lines.append("")
    lines.append("-- code-switching behaviour --")
    lines.append(f"  en omission rate     {_pct(m.en_omission_rate)} %   "
                 f"({m.utts_en_omitted}/{m.utts_with_en_ref} utts lost all English)")
    lines.append(f"  zh omission rate     {_pct(m.zh_omission_rate)} %   "
                 f"({m.utts_zh_omitted}/{m.utts_with_zh_ref} utts lost all Chinese)")
    lines.append(f"  en->zh substitution  {_pct(m.en_sub_by_zh_rate)} %   "
                 f"({m.en_sub_by_zh} en tokens replaced by Chinese)")
    lines.append(f"  en lost (del+sub)    {_pct(m.en_lost_rate)} %   ({m.en_lost} tokens)")
    lines.append(f"  switch ratio         {_num(m.switch_ratio)}   "
                 f"({m.n_switch_hyp} produced / {m.n_switch_ref} expected)")
    lines.append("")
    lines.append("-- sanity --")
    lines.append(f"  length ratio         {_num(m.length_ratio)}   (hyp tokens / ref tokens)")
    lines.append(f"  runaway rate         {_pct(m.runaway_rate)} %   "
                 f"({m.utts_runaway} utts far longer than reference)")
    lines.append(f"  mean CMI (reference) {_num(m.mean_cmi_ref)}   "
                 f"(0 = monolingual, 0.5 = balanced mix)")
    lines.append("")
    lines.append(diagnose(m))
    return "\n".join(lines)


def diagnose(m: CorpusMetrics) -> str:
    """Turn the decomposition into the first thing to go check.

    Every branch here corresponds to a concrete, previously observed failure in
    Cantonese-English evaluation. Nothing is inferred beyond the counts.
    """
    notes: List[str] = []

    if m.n_ref_tokens and m.n_hyp_tokens == 0:
        # Everything below would describe this as a language failure. It is not
        # one: the system emitted nothing, which is a broken run, not a score.
        return (
            "-- diagnosis --\n"
            "  Every hypothesis is empty: the system produced no output at all for "
            "any\n  utterance. This is a broken run, not a recognition result -- the "
            "100% rates\n  above are an artefact of scoring emptiness. Check the "
            "transcription step for\n  load or decode errors before reading any of "
            "these numbers."
        )
    if m.length_ratio is not None and 0 < m.length_ratio < 0.2:
        notes.append(
            f"- Hypotheses are {m.length_ratio:.2f}x the reference length. Before "
            f"reading this as deletion errors, check how many utterances came back "
            f"empty from a failed decode."
        )

    if m.runaway_rate is not None and m.runaway_rate > 0.05:
        notes.append(
            f"- {_pct(m.runaway_rate)}% of utterances are >2x the reference length, and "
            f"insertions are {_pct(m.ins_rate)}% of reference tokens. That is decoder "
            f"collapse / repetition, not recognition error. For Whisper this is the "
            f"known consequence of forcing language='yue'; try language='zh'."
        )
    if m.ins_rate is not None and m.ins_rate > 0.30 and (m.runaway_rate or 0) <= 0.05:
        notes.append(
            f"- Insertion rate {_pct(m.ins_rate)}% is high without runaway outputs. "
            f"Check for leftover decoder tags or punctuation surviving normalisation."
        )
    subs_dominate = m.subs > m.ins and m.subs > m.dels
    if m.cer_zh is not None and m.cer_zh > 0.45 and subs_dominate:
        notes.append(
            f"- CER_zh {_pct(m.cer_zh)}% with substitutions dominating usually means a "
            f"script mismatch (Traditional reference vs Simplified hypothesis). Set "
            f"--script t2s (or s2t) so both sides are unified before scoring."
        )
    if m.en_omission_rate is not None and m.en_omission_rate > 0.10:
        notes.append(
            f"- {_pct(m.en_omission_rate)}% of English-bearing utterances came back with "
            f"no English at all. This is the 'language omission' failure mode; CER alone "
            f"will not show it because Chinese dominates the character count."
        )
    if m.en_sub_by_zh_rate is not None and m.en_sub_by_zh_rate > 0.15:
        notes.append(
            f"- {_pct(m.en_sub_by_zh_rate)}% of English tokens were replaced by Chinese "
            f"material: the model is translating instead of transcribing."
        )
    if m.switch_ratio is not None and m.switch_ratio < 0.6:
        notes.append(
            f"- Only {_num(m.switch_ratio)} of the expected language switches were "
            f"produced. The model is flattening utterances into one language."
        )
    if m.pier is not None and m.mer is not None and m.pier > m.mer * 1.3:
        notes.append(
            f"- PIER ({_pct(m.pier)}%) is {m.pier / m.mer:.2f}x MER ({_pct(m.mer)}%): "
            f"errors are concentrated at switch points, which is the expected shape for "
            f"a model that is good at both languages but bad at switching between them."
        )
    if m.mer is not None and m.mer > 1.0:
        notes.append(
            "- MER above 1.0 means errors outnumber reference tokens. On clean read "
            "speech that is almost always a scoring bug, not a model result. Read the "
            "decomposition above before believing this number."
        )

    if not notes:
        return "-- diagnosis --\n  nothing anomalous in the decomposition."
    return "-- diagnosis --\n" + "\n".join("  " + n[2:] for n in notes)


def markdown_table(rows: Sequence[tuple]) -> str:
    """Comparison table across models.

    ``rows`` is a sequence of ``(name, CorpusMetrics)``.
    """
    header = (
        "| Model | MER % | CER_zh % | WER_en % | PIER % | EN omit % | EN->ZH % | "
        "Ins % | Del % | Sub % | Runaway % | Len ratio |"
    )
    sep = "|" + "---|" * 12
    lines = [header, sep]
    for name, m in rows:
        lines.append(
            "| {name} | {mer} | {cer} | {wer} | {pier} | {omit} | {tr} | {ins} | "
            "{dele} | {sub} | {run} | {lr} |".format(
                name=name,
                mer=_pct(m.mer),
                cer=_pct(m.cer_zh),
                wer=_pct(m.wer_en),
                pier=_pct(m.pier),
                omit=_pct(m.en_omission_rate),
                tr=_pct(m.en_sub_by_zh_rate),
                ins=_pct(m.ins_rate),
                dele=_pct(m.del_rate),
                sub=_pct(m.sub_rate),
                run=_pct(m.runaway_rate),
                lr=_num(m.length_ratio),
            )
        )
    return "\n".join(lines)


def worst_utterances(
    results: Sequence[UtteranceResult], k: int = 20, min_ref_tokens: int = 3
) -> List[UtteranceResult]:
    """The k worst utterances by MER, for eyeballing.

    Utterances shorter than ``min_ref_tokens`` are excluded: a two-token
    reference with one error scores 50% and would otherwise fill the whole list
    without telling you anything.
    """
    eligible = [r for r in results if r.n_ref >= min_ref_tokens]
    return sorted(eligible, key=lambda r: (r.mer or 0.0), reverse=True)[:k]


def format_worst(results: Sequence[UtteranceResult], k: int = 20) -> str:
    lines = ["-- worst utterances --"]
    for r in worst_utterances(results, k=k):
        lines.append(
            f"  [{r.id}] MER={_pct(r.mer)}%  S={r.subs} D={r.dels} I={r.ins}"
        )
        lines.append(f"    REF: {r.ref}")
        lines.append(f"    HYP: {r.hyp}")
    return "\n".join(lines)
