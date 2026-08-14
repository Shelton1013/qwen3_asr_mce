"""Find utterances whose audio is paired with the wrong reference.

Observed in the MCE corpus: adjacent utterances swapped.

    97_40  REF running好重要 因为可以improve一下cardiovascular fitness
           HYP 作为一个footballer 要够improve ball control同passing skills
    97_41  REF 作为一个footballer 要够improve ball control同passing skills
           HYP running好重要 因为可以improve一下cardiovascular fitness

The model transcribed both correctly. The labels are crossed. Each such pair
scores ~100% MER twice over, so a handful of them moves the corpus number while
saying nothing about the model -- and worse, training on them teaches the model
to map audio to unrelated text.

The test is cheap: if a hypothesis matches some *other* reference far better
than its own, the pairing is suspect. Only nearby rows are compared, because
mislabelling comes from a shifted or swapped index, not from a random draw
across the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .align import align
from .tokenizer import token_texts, tokenize


@dataclass
class Misalignment:
    """One utterance that matches a neighbour's reference better than its own."""

    index: int
    utt_id: str
    own_mer: float
    best_id: str
    best_index: int
    best_mer: float
    ref: str
    hyp: str

    @property
    def improvement(self) -> float:
        """How much better the neighbour's reference fits, in MER points."""
        return self.own_mer - self.best_mer


def _mer(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else float("inf")
    errors = sum(1 for op in align(ref_tokens, hyp_tokens) if op.is_error)
    return errors / len(ref_tokens)


def find_misalignments(
    rows: Sequence[dict],
    window: int = 5,
    max_own_mer: float = 0.5,
    ratio: float = 0.5,
    max_best_mer: float = 0.5,
) -> List[Misalignment]:
    """Flag rows whose hypothesis fits a nearby reference much better.

    ``window``       how many rows either side to compare against. Rows arrive in
                     manifest order, which groups by speaker, so a small window
                     stays inside one speaker's block.
    ``max_own_mer``  only consider rows that scored badly against their own
                     reference; a row that already matches well is not mislabelled.
    ``ratio``        the neighbour must fit at least this much better,
                     proportionally (0.5 = less than half the error).
    ``max_best_mer`` and it must fit well in absolute terms, so two equally bad
                     rows do not "explain" each other.
    """
    tokens = [token_texts(tokenize(r.get("ref", ""))) for r in rows]
    hyps = [token_texts(tokenize(r.get("hyp", ""))) for r in rows]

    found: List[Misalignment] = []
    for i, row in enumerate(rows):
        if not tokens[i] or not hyps[i]:
            continue
        own = _mer(tokens[i], hyps[i])
        if own < max_own_mer:
            continue

        best_mer = own
        best_j = None
        lo, hi = max(0, i - window), min(len(rows), i + window + 1)
        for j in range(lo, hi):
            if j == i or not tokens[j]:
                continue
            candidate = _mer(tokens[j], hyps[i])
            if candidate < best_mer:
                best_mer, best_j = candidate, j

        if best_j is None or best_mer > max_best_mer or best_mer > own * ratio:
            continue
        found.append(
            Misalignment(
                index=i,
                utt_id=str(row.get("id", i)),
                own_mer=own,
                best_id=str(rows[best_j].get("id", best_j)),
                best_index=best_j,
                best_mer=best_mer,
                ref=row.get("ref", ""),
                hyp=row.get("hyp", ""),
            )
        )
    return found


def find_swapped_pairs(
    misalignments: Sequence[Misalignment],
) -> List[Tuple[Misalignment, Misalignment]]:
    """Pairs that point at each other -- the clearest possible evidence.

    A one-way match can be coincidence (two utterances on the same topic). A
    mutual match means the two references are simply the wrong way round.
    """
    by_index = {m.index: m for m in misalignments}
    pairs = []
    seen = set()
    for m in misalignments:
        other = by_index.get(m.best_index)
        if other is None or other.best_index != m.index:
            continue
        key = tuple(sorted((m.index, other.index)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((m, other))
    return pairs


def estimate_mer_impact(
    rows: Sequence[dict], misalignments: Sequence[Misalignment]
) -> Optional[float]:
    """MER points the flagged rows contribute, as a share of all reference tokens.

    An upper bound on what fixing the labels could recover: it credits each
    flagged row with the full difference between its own score and the better
    match, which assumes the better match is the correct pairing.
    """
    total_ref = sum(len(token_texts(tokenize(r.get("ref", "")))) for r in rows)
    if not total_ref:
        return None
    recovered = 0.0
    for m in misalignments:
        n_ref = len(token_texts(tokenize(m.ref)))
        recovered += m.improvement * n_ref
    return recovered / total_ref


def format_report(
    rows: Sequence[dict], misalignments: Sequence[Misalignment], top: int = 20
) -> str:
    lines = [f"alignment check over {len(rows)} utterances"]
    if not misalignments:
        lines.append("  no misalignments found.")
        return "\n".join(lines)

    pairs = find_swapped_pairs(misalignments)
    impact = estimate_mer_impact(rows, misalignments)

    lines.append("")
    lines.append(f"  {len(misalignments)} utterances match a neighbour's reference "
                 f"better than their own")
    lines.append(f"  {len(pairs)} of them form mutual pairs -- two references the "
                 f"wrong way round")
    if impact is not None:
        lines.append(f"  up to {impact * 100:.2f} MER points are attributable to "
                     f"these labels, not the model")
    lines.append("")
    lines.append("  These are corpus defects. Fix or drop them before trusting the")
    lines.append("  headline number, and before training on the affected rows --")
    lines.append("  a crossed pair teaches the model to map audio to unrelated text.")

    if pairs:
        lines.append("")
        lines.append("== mutually swapped pairs ==")
        for a, b in pairs[:top]:
            lines.append(f"  {a.utt_id}  <->  {b.utt_id}")
            lines.append(f"    {a.utt_id} own MER {a.own_mer * 100:6.2f}%  "
                         f"against {b.utt_id}: {a.best_mer * 100:6.2f}%")
            lines.append(f"      REF: {a.ref}")
            lines.append(f"      HYP: {a.hyp}")

    one_way = [m for m in misalignments if all(m is not x for pair in pairs for x in pair)]
    if one_way:
        lines.append("")
        lines.append("== one-way matches (weaker evidence; check by ear) ==")
        for m in sorted(one_way, key=lambda x: -x.improvement)[:top]:
            lines.append(f"  {m.utt_id} -> {m.best_id}   "
                         f"own {m.own_mer * 100:.2f}% vs {m.best_mer * 100:.2f}%")
            lines.append(f"      REF: {m.ref}")
            lines.append(f"      HYP: {m.hyp}")
    return "\n".join(lines)
