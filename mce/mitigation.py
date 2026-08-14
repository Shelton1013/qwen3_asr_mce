"""Interventions against language-identification collapse.

Two of them, and the contrast between them is the point.

**Script constraint** (:func:`build_script_logits_processor`) masks any token
carrying a character outside the scripts that may legitimately appear. It is the
obvious fix, and it is the one a reviewer will propose, so it belongs in the
paper as a baseline rather than as a contribution.

Note what it can and cannot do *in a code-switching setting*. For monolingual
ASR the constraint is decisive: the target is Cantonese, so mask Latin and the
model cannot answer in English. Here both scripts are legitimate, so the mask
can only exclude a third script. It stops Thai; it does nothing about English
being rendered as Chinese, which is the larger failure by an order of magnitude
(en->zh substitution 10.7% versus foreign-script 1.0%). The trivial fix is
strictly weaker precisely where code-switching lives.

**Context anchoring** is the intervention the collapse mechanism actually
predicts. Collapse rises as the embedded language's share rises and falls as the
utterance lengthens -- longer utterances accumulate enough matrix-language
evidence to hold the language prior steady. If that reading is right, supplying
matrix-language context should suppress collapse without touching the output
distribution's support. Qwen3-ASR takes such context through its transcription
prompt, so this needs no new machinery: see :data:`CANTONESE_ANCHOR` and pass it
with ``--prompt``.

The two make different predictions, which is what makes the comparison worth
running. A constraint suppresses the symptom whatever the cause. Anchoring only
works if the prior-flip account is correct -- and if it works, the mechanism and
the intervention corroborate each other.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, Optional, Sequence, Set

from .tokenizer import EN, NUM, OTHER, ZH, is_cjk

#: Script classes a token may carry. ``neutral`` covers punctuation, spaces and
#: control characters, which never identify a language and are always allowed.
NEUTRAL = "neutral"

#: Named constraint sets. ``zh+en`` is the Cantonese-English one: it admits both
#: legitimate scripts and excludes everything else.
SCRIPT_SETS: Dict[str, Set[str]] = {
    "zh+en": {ZH, EN, NUM, NEUTRAL},
    "zh": {ZH, NUM, NEUTRAL},
    "en": {EN, NUM, NEUTRAL},
}

#: Matrix-language context for the anchoring intervention. Deliberately generic:
#: it supplies Cantonese evidence without naming any content word that could
#: bias recognition of a particular utterance.
CANTONESE_ANCHOR = "以下係一段香港人講嘅粵語，中間會夾雜英文詞。"

# Byte-level and sentencepiece markers that stand for whitespace rather than
# script. Stripping them keeps a leading-space token from looking neutral-only.
_MARKERS = "ĠĊĉ▁▁"


def classify_char(ch: str) -> str:
    """Which script class a single character belongs to."""
    if ch.isspace() or ch in _MARKERS:
        return NEUTRAL
    if is_cjk(ch):
        return ZH
    if ch.isdigit():
        return NUM
    category = unicodedata.category(ch)
    if category.startswith("P") or category.startswith("S") or category.startswith("C"):
        return NEUTRAL
    if ch.isalpha():
        # Latin, including accented forms; anything else alphabetic is a third
        # script and is exactly what we are trying to exclude.
        try:
            return EN if "LATIN" in unicodedata.name(ch) else OTHER
        except ValueError:  # pragma: no cover - unnamed codepoint
            return OTHER
    return OTHER


def token_scripts(text: str) -> Set[str]:
    return {classify_char(ch) for ch in text} or {NEUTRAL}


def allowed_token_ids(
    tokenizer, allowed: Iterable[str], keep_special: bool = True
) -> Set[int]:
    """Token ids whose surface form stays inside ``allowed``.

    Special tokens are kept regardless: they carry no script and the decoder
    needs them to terminate.
    """
    allowed = set(allowed)
    try:
        vocab_size = len(tokenizer)
    except TypeError:  # pragma: no cover - unusual tokenizer
        vocab_size = tokenizer.vocab_size

    ids = list(range(vocab_size))
    # One batched call; per-id decoding over a 150k vocabulary is minutes.
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])

    keep: Set[int] = set()
    for i, tok in zip(ids, tokens):
        if tok is None:
            continue
        if keep_special and i in special:
            keep.add(i)
            continue
        if isinstance(tok, bytes):  # pragma: no cover - byte tokenizers
            tok = tok.decode("utf-8", errors="replace")
        if token_scripts(tok) <= allowed:
            keep.add(i)
    return keep


def build_script_logits_processor(tokenizer, constraint: str = "zh+en"):
    """A ``LogitsProcessor`` that forbids tokens outside ``constraint``.

    Built once at load time; the vocabulary scan is O(V) and the per-step cost
    is a single masked add.
    """
    import torch
    from transformers import LogitsProcessor

    if constraint not in SCRIPT_SETS:
        raise ValueError(
            f"unknown script constraint {constraint!r}; "
            f"choose from {', '.join(sorted(SCRIPT_SETS))}"
        )
    keep = allowed_token_ids(tokenizer, SCRIPT_SETS[constraint])
    if not keep:
        raise RuntimeError("script constraint would block the entire vocabulary")

    try:
        vocab_size = len(tokenizer)
    except TypeError:  # pragma: no cover
        vocab_size = tokenizer.vocab_size
    mask = torch.ones(vocab_size, dtype=torch.bool)
    mask[list(keep)] = False  # True marks a *blocked* id

    blocked = int(mask.sum())
    print(
        f"[info] script constraint '{constraint}': blocking {blocked}/{vocab_size} "
        f"tokens ({blocked / vocab_size:.1%})"
    )

    class ScriptConstraintProcessor(LogitsProcessor):
        def __init__(self, blocked_mask):
            self.blocked = blocked_mask

        def __call__(self, input_ids, scores):
            blocked = self.blocked.to(scores.device)
            # Slice rather than assume equal width: generation configs sometimes
            # pad the logit dimension past the tokenizer's length.
            width = min(scores.shape[-1], blocked.shape[-1])
            scores[..., :width] = scores[..., :width].masked_fill(
                blocked[:width], float("-inf")
            )
            if scores.shape[-1] > width:
                scores[..., width:] = float("-inf")
            return scores

    return ScriptConstraintProcessor(mask)
