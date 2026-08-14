"""Error analysis over a scored run: which tokens go wrong, and into what.

The corpus metrics say *how much* is wrong. This says *what*, by re-aligning the
per-utterance dump and tabulating the actual substitution pairs. Two questions
it exists to answer:

**What should DPO negatives look like?** ``en -> zh`` substitutions are the
"translating instead of transcribing" failure, but that label covers two things
that need different negatives: semantic translation (``typhoon`` -> 台风, the
model knows the word and renders its meaning) and phonetic collapse
(``the sun`` -> 特新, the model heard it and decoded into Chinese character
space anyway). Ranked pairs let you see the split and build both kinds.

**Which "errors" are really reference noise?** Frequent single-character
``zh -> zh`` substitutions between homophones -- 佢地/佢哋, 钟意/中意 -- are
orthographic variants, not recognition failures. They inflate CER_zh for free.
Seeing them ranked tells you whether to add a normalisation rule.

Run this on **dev**, never on test. Mining test-set failures and feeding them
back into training is contamination, and it is invisible in the resulting score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .align import DELETE, EQUAL, INSERT, SUB, align
from .tokenizer import EN, NUM, OTHER, ZH, Token, poi_indices, token_texts, tokenize


@dataclass
class ErrorInventory:
    """Every edit in the run, grouped by what kind of token it touched."""

    #: (ref_lang, hyp_lang) -> Counter of (ref_text, hyp_text)
    substitutions: Dict[Tuple[str, str], Counter]
    #: lang -> Counter of deleted reference tokens
    deletions: Dict[str, Counter]
    #: lang -> Counter of inserted hypothesis tokens
    insertions: Dict[str, Counter]
    #: same as ``substitutions`` but restricted to switch-point regions
    poi_substitutions: Dict[Tuple[str, str], Counter]
    #: every English reference token that was deleted or substituted, counted
    #: individually. Spans are better for reading, but an English word swallowed
    #: inside a long mixed-language deletion would vanish from a span table --
    #: and "which English words does the model lose" is the DPO vocabulary.
    lost_en_tokens: Counter = None  # type: ignore[assignment]
    n_utts: int = 0

    def __post_init__(self) -> None:
        if self.lost_en_tokens is None:
            self.lost_en_tokens = Counter()

    def pairs(self, ref_lang: str, hyp_lang: str) -> Counter:
        return self.substitutions.get((ref_lang, hyp_lang), Counter())

    def total(self, ref_lang: str, hyp_lang: str) -> int:
        return sum(self.pairs(ref_lang, hyp_lang).values())


MIXED = "mixed"


def join_tokens(tokens: Sequence[Token]) -> str:
    """Render a token span back into readable text.

    Chinese characters butt together; anything else gets a space. This is for
    human reading, not round-tripping.
    """
    out = ""
    for i, tok in enumerate(tokens):
        if i and not (tok.lang == ZH and tokens[i - 1].lang == ZH):
            out += " "
        out += tok.text
    return out


def span_lang(tokens: Sequence[Token]) -> str:
    langs = {t.lang for t in tokens}
    return langs.pop() if len(langs) == 1 else MIXED


def edit_spans(ops, ref: Sequence[Token], hyp: Sequence[Token]):
    """Group consecutive edits into ``(ref_tokens, hyp_tokens, ref_indices)``.

    A single edit is often only part of one mistake. ``typhoon`` -> ``台风``
    costs one substitution plus one insertion, and which token the substitution
    lands on is an arbitrary tie-break inside the aligner -- reporting
    ``typhoon -> 风`` would be both wrong and useless as a DPO negative.
    Merging the run recovers ``typhoon -> 台风``.
    """
    spans = []
    run_ref: List[Token] = []
    run_hyp: List[Token] = []
    run_idx: List[int] = []

    def flush():
        if run_ref or run_hyp:
            spans.append((list(run_ref), list(run_hyp), list(run_idx)))
        run_ref.clear()
        run_hyp.clear()
        run_idx.clear()

    for op in ops:
        if op.op == EQUAL:
            flush()
            continue
        if op.op in (SUB, DELETE):
            run_ref.append(ref[op.ref_idx])
            run_idx.append(op.ref_idx)
        if op.op in (SUB, INSERT):
            run_hyp.append(hyp[op.hyp_idx])
        if op.op == INSERT and not run_idx:
            run_idx.append(op.ref_pos)
    flush()
    return spans


def build_inventory(
    pairs: Iterable[Tuple[str, str]], poi_window: int = 1
) -> ErrorInventory:
    """Tabulate every edit across ``(reference, hypothesis)`` pairs.

    Both strings must already be normalised -- pass the ``ref``/``hyp`` fields
    from a ``--out-utts`` dump, which are exactly what was scored.
    """
    subs: Dict[Tuple[str, str], Counter] = {}
    poi_subs: Dict[Tuple[str, str], Counter] = {}
    dels: Dict[str, Counter] = {}
    ins: Dict[str, Counter] = {}
    lost_en: Counter = Counter()
    n = 0

    for ref_text, hyp_text in pairs:
        n += 1
        ref: List[Token] = tokenize(ref_text)
        hyp: List[Token] = tokenize(hyp_text)
        poi = poi_indices(ref, window=poi_window)
        ops = align(token_texts(ref), token_texts(hyp))

        for span_ref, span_hyp, indices in edit_spans(ops, ref, hyp):
            for tok in span_ref:
                if tok.lang == EN:
                    lost_en[tok.text] += 1
            if span_ref and span_hyp:
                key = (span_lang(span_ref), span_lang(span_hyp))
                value = (join_tokens(span_ref), join_tokens(span_hyp))
                subs.setdefault(key, Counter())[value] += 1
                if any(i in poi for i in indices):
                    poi_subs.setdefault(key, Counter())[value] += 1
            elif span_ref:
                dels.setdefault(span_lang(span_ref), Counter())[join_tokens(span_ref)] += 1
            else:
                ins.setdefault(span_lang(span_hyp), Counter())[join_tokens(span_hyp)] += 1

    return ErrorInventory(subs, dels, ins, poi_subs, lost_en_tokens=lost_en, n_utts=n)


def looks_orthographic(ref: str, hyp: str) -> bool:
    """Heuristic: a single-character zh->zh swap is a candidate spelling variant.

    Deliberately weak. It flags candidates for a human to confirm -- deciding
    that 地/哋 are the same word and 疯/风 are not needs someone who reads
    Cantonese, and guessing wrong would silently normalise away real errors.
    """
    return len(ref) == 1 and len(hyp) == 1


def format_inventory(
    inv: ErrorInventory, top: int = 30, min_count: int = 2
) -> str:
    """A readable report. Sections are ordered by how actionable they are."""
    lines: List[str] = []

    def table(title: str, counter: Counter, note: str = "") -> None:
        total = sum(counter.values())
        distinct = len(counter)
        lines.append("")
        lines.append(f"== {title} ==")
        lines.append(f"   {total} occurrences over {distinct} distinct pairs")
        if note:
            lines.append(f"   {note}")
        shown = [(k, v) for k, v in counter.most_common(top) if v >= min_count]
        if not shown:
            lines.append("   (nothing above the minimum count)")
            return
        width = max(len(str(k[0])) for k, _ in shown)
        for (ref, hyp), count in shown:
            lines.append(f"   {count:5d}  {ref:<{width}}  ->  {hyp}")

    lines.append(f"error inventory over {inv.n_utts} utterances")

    table(
        "English replaced by Chinese",
        inv.pairs(EN, ZH),
        "translation (typhoon->台风) and phonetic collapse (the sun->特新) both "
        "land here;\n   they need different DPO negatives, so read the pairs.",
    )
    table(
        "English replaced by English",
        inv.pairs(EN, EN),
        "plain misrecognition, unrelated to code-switching.",
    )

    lost = inv.lost_en_tokens
    lines.append("")
    lines.append("== English tokens lost (deleted or substituted) ==")
    lines.append(f"   {sum(lost.values())} occurrences over {len(lost)} distinct tokens")
    lines.append("   counted per token, so a word swallowed inside a longer deletion")
    lines.append("   still shows up. This is the DPO negative vocabulary.")
    shown = [(t, c) for t, c in lost.most_common(top) if c >= min_count]
    for tok, count in shown:
        lines.append(f"   {count:5d}  {tok}")
    if not shown:
        lines.append("   (nothing above the minimum count)")

    zh_pairs = inv.pairs(ZH, ZH)
    ortho = Counter({k: v for k, v in zh_pairs.items() if looks_orthographic(*k)})
    table(
        "Chinese single-character swaps (orthographic variant candidates)",
        ortho,
        "frequent homophone pairs here are reference noise, not recognition "
        "errors;\n   confirm by ear, then add them to the normaliser.",
    )

    poi_en_zh = inv.poi_substitutions.get((EN, ZH), Counter())
    table(
        "English->Chinese at switch points only",
        poi_en_zh,
        "the subset that sits on a language boundary -- the highest-value "
        "negatives.",
    )

    lines.append("")
    lines.append("== totals ==")
    for (rl, hl), counter in sorted(
        inv.substitutions.items(), key=lambda kv: -sum(kv[1].values())
    ):
        lines.append(f"   sub {rl:>5} -> {hl:<5} {sum(counter.values()):6d}")
    for lang, counter in sorted(inv.deletions.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(f"   del {lang:>5} {'':<9}{sum(counter.values()):6d}")
    for lang, counter in sorted(inv.insertions.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(f"   ins {lang:>5} {'':<9}{sum(counter.values()):6d}")
    return "\n".join(lines)


def inventory_to_dict(inv: ErrorInventory, min_count: int = 1) -> dict:
    """Machine-readable form, for generating DPO negatives downstream."""

    def dump(counter: Counter) -> List[dict]:
        return [
            {"ref": r, "hyp": h, "count": c}
            for (r, h), c in counter.most_common()
            if c >= min_count
        ]

    return {
        "n_utts": inv.n_utts,
        "lost_en_tokens": [
            {"token": t, "count": c}
            for t, c in inv.lost_en_tokens.most_common()
            if c >= min_count
        ],
        "substitutions": {
            f"{rl}->{hl}": dump(c) for (rl, hl), c in inv.substitutions.items()
        },
        "poi_substitutions": {
            f"{rl}->{hl}": dump(c) for (rl, hl), c in inv.poi_substitutions.items()
        },
        "deletions": {
            lang: [{"token": t, "count": c} for t, c in counter.most_common() if c >= min_count]
            for lang, counter in inv.deletions.items()
        },
        "insertions": {
            lang: [{"token": t, "count": c} for t, c in counter.most_common() if c >= min_count]
            for lang, counter in inv.insertions.items()
        },
    }
