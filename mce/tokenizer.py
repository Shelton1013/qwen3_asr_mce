"""Code-switching aware tokenisation.

The whole point of MER (Mixed Error Rate) is that a single tokenisation rule is
wrong for code-switched text:

* Chinese has no whitespace, so whitespace splitting collapses an entire
  Cantonese span into one or two giant "words".  The reference token count
  implodes and the error rate can easily exceed 1.0 -- which is exactly the
  failure mode this package exists to stop you from shipping.
* Character splitting English inflates the token count and under-reports real
  word errors.

So we tokenise per script: CJK ideographs become one token each, Latin runs
become one token per word.  Every token carries a language tag, which is what
makes the per-language and switch-point metrics possible downstream.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence

# Language tags used throughout the package.
ZH = "zh"
EN = "en"
NUM = "num"
OTHER = "other"

#: Languages that participate in code-switching boundary detection.
CS_LANGS = (ZH, EN)

# CJK ideographs, including the common extensions and compatibility blocks.
# Bopomofo / kana are deliberately excluded: they are not part of the
# Cantonese-English setting and would silently change the denominator.
_CJK_PATTERN = (
    r"㐀-䶿"      # CJK Ext A
    r"一-鿿"      # CJK Unified Ideographs
    r"豈-﫿"      # CJK Compatibility Ideographs
    r"\U00020000-\U0002a6df"  # CJK Ext B
    r"\U0002a700-\U0002ebef"  # CJK Ext C-F
    r"\U0002f800-\U0002fa1f"  # CJK Compatibility Supplement
)
_CJK_RE = re.compile(f"[{_CJK_PATTERN}]")

# A Latin "word": letters, optionally with internal apostrophes / hyphens.
# Normalisation usually strips those already, but the tokenizer must not depend
# on normalisation having run.
_LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’-][A-Za-zÀ-ɏ]+)*")
_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Token:
    """A single scoring unit together with the script it came from."""

    text: str
    lang: str

    def __str__(self) -> str:  # pragma: no cover - debugging convenience
        return f"{self.text}/{self.lang}"


def is_cjk(ch: str) -> bool:
    """True if ``ch`` is a CJK ideograph we score at character level."""
    return bool(_CJK_RE.match(ch))


def tokenize(text: str) -> List[Token]:
    """Split ``text`` into MER scoring units.

    Rules, applied left to right:

    * one CJK ideograph -> one ``zh`` token
    * one Latin word    -> one ``en`` token
    * one digit run     -> one ``num`` token
    * anything else that is not whitespace -> one ``other`` token per character

    ``num`` and ``other`` tokens still count towards the overall MER denominator
    (they are real content the model has to get right) but they are excluded
    from the language-restricted rates and from switch-point detection, because
    "2026" is not evidence about either language.
    """
    tokens: List[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if is_cjk(ch):
            tokens.append(Token(ch, ZH))
            i += 1
            continue
        m = _LATIN_RE.match(text, i)
        if m:
            tokens.append(Token(m.group(0), EN))
            i = m.end()
            continue
        m = _DIGIT_RE.match(text, i)
        if m:
            tokens.append(Token(m.group(0), NUM))
            i = m.end()
            continue
        tokens.append(Token(ch, OTHER))
        i += 1
    return tokens


def token_texts(tokens: Sequence[Token]) -> List[str]:
    """Extract the surface strings, which is what the aligner compares."""
    return [t.text for t in tokens]


def count_langs(tokens: Iterable[Token]) -> dict:
    """Token count per language tag."""
    counts = {ZH: 0, EN: 0, NUM: 0, OTHER: 0}
    for t in tokens:
        counts[t.lang] = counts.get(t.lang, 0) + 1
    return counts


def switch_points(tokens: Sequence[Token]) -> List[int]:
    """Indices where the reference switches between Cantonese and English.

    A returned index ``i`` means "the boundary sits immediately before
    ``tokens[i]``", i.e. ``tokens[i]`` is the first token of the new language.
    Tokens tagged ``num`` / ``other`` are transparent: they neither create nor
    break a switch, since a digit between Chinese and English is not itself a
    switch point.
    """
    points: List[int] = []
    prev_idx = None
    for i, tok in enumerate(tokens):
        if tok.lang not in CS_LANGS:
            continue
        if prev_idx is not None and tok.lang != tokens[prev_idx].lang:
            points.append(i)
        prev_idx = i
    return points


def poi_indices(tokens: Sequence[Token], window: int = 1) -> set:
    """Reference token indices inside the point-of-interest (switch) regions.

    ``window=1`` covers the last token of the outgoing language and the first
    token of the incoming one -- the minimal region where a code-switching
    model actually has to do something a monolingual model cannot.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(tokens)
    idx: set = set()
    for boundary in switch_points(tokens):
        for k in range(boundary - window, boundary + window):
            if 0 <= k < n:
                idx.add(k)
    return idx


def code_mixing_index(tokens: Sequence[Token]) -> float:
    """Utterance-level Code Mixing Index over the zh/en tokens only.

    ``CMI = (N - max(N_zh, N_en)) / N``.  0.0 means monolingual, 0.5 means a
    perfectly balanced mix.  Reported on the *reference* so you can tell whether
    a low error rate came from an easy, barely-mixed test set.
    """
    zh = sum(1 for t in tokens if t.lang == ZH)
    en = sum(1 for t in tokens if t.lang == EN)
    total = zh + en
    if total == 0:
        return 0.0
    return (total - max(zh, en)) / total


def strip_accents(text: str) -> str:
    """NFKD fold that drops combining marks; used by the normaliser."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def collapse_whitespace(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()
