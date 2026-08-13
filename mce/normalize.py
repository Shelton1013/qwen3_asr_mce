"""Text normalisation applied identically to references and hypotheses.

Most of the "WER > 1.0" pathologies in Cantonese-English evaluation are scoring
artefacts, not model failures.  The three that matter here:

1. **Script mismatch.**  Qwen3-ASR is trained mostly on Simplified Chinese;
   Hong Kong references are usually Traditional.  Without unification *every*
   Chinese character is a substitution and CER jumps above 0.5.
2. **Decoder tags.**  SenseVoice emits ``<|yue|><|NEUTRAL|><|Speech|><|woitn|>``
   inline, Whisper emits ``<|startoftranscript|>``-family tokens if you decode
   without skipping specials.  Left in, they are pure insertions.
3. **Punctuation and full-width forms.**  Whisper punctuates, Qwen3-ASR
   punctuates differently, references often do not punctuate at all.

What this module deliberately does *not* try to fix is the 書面語 / 口語
register gap (Whisper writing 「是」 for an audible 「係」).  That is a genuine
model behaviour, not a formatting artefact, and silently normalising it away
would hide the single largest source of Whisper's apparent Cantonese error.
Measure it instead -- see ``docs`` in the README.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from .tokenizer import collapse_whitespace

# Inline decoder tags: <|yue|>, <|NEUTRAL|>, <|woitn|>, <|endoftext|> ...
_TAG_RE = re.compile(r"<\|[^|]*\|>")
# Bracketed non-speech annotations frequently present in references.
_BRACKET_RE = re.compile(r"[\[\(（【][^\]\)）】]{0,32}[\]\)）】]")
# Placeholder tokens some corpora use for unintelligible speech.
_PLACEHOLDER_RE = re.compile(r"<\s*(unk|UNK|noise|NOISE|sil|SIL)\s*>")

# Punctuation replaced by a space (it marks a token boundary).
_PUNCT_TO_SPACE = (
    "!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~"
    "。，、；：？！“”‘’（）《》〈〉【】〔〕—…·「」『』～｜＂＇"
)
# Punctuation deleted outright because it lives *inside* a word.
# Removing rather than splitting keeps "don't" -> "dont" on both sides.
_PUNCT_TO_DELETE = "'’-‐‑–"

_TRANS_SPACE = {ord(c): " " for c in _PUNCT_TO_SPACE}
_TRANS_DELETE = {ord(c): None for c in _PUNCT_TO_DELETE}

_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}


@dataclass
class NormalizeConfig:
    """Every switch here is applied to reference and hypothesis alike.

    Defaults are the conservative set: fix the artefacts, touch nothing that
    could be a real model difference.
    """

    #: ``"t2s"`` Traditional->Simplified, ``"s2t"`` the reverse, ``None`` to skip.
    #: Pick one and apply it to *both* sides; which direction you pick does not
    #: matter for the score, only that it is consistent.
    script: Optional[str] = "t2s"
    lowercase: bool = True
    remove_tags: bool = True
    remove_brackets: bool = True
    to_halfwidth: bool = True
    remove_punct: bool = True
    strip_accents: bool = False
    #: Experimental: fold English number words and Chinese numerals to digits.
    #: Off by default -- it is a real semantic rewrite and can fire on "one of".
    normalize_numbers: bool = False
    #: Extra literal strings removed before anything else (corpus-specific junk).
    drop_strings: List[str] = field(default_factory=list)


class Normalizer:
    """Callable normalisation pipeline.

    >>> n = Normalizer(NormalizeConfig(script="t2s"))
    >>> n("我 today 好 Busy 啊！")
    '我 today 好 busy 啊'
    """

    def __init__(self, config: Optional[NormalizeConfig] = None) -> None:
        self.config = config or NormalizeConfig()
        self._converter = self._build_converter(self.config.script)
        self._cn2an = None
        if self.config.normalize_numbers:
            try:  # pragma: no cover - optional dependency
                import cn2an  # type: ignore

                self._cn2an = cn2an
            except ImportError:
                self._cn2an = None

    @staticmethod
    def _build_converter(script: Optional[str]):
        if not script:
            return None
        if script not in ("t2s", "s2t"):
            raise ValueError(f"script must be 't2s', 's2t' or None, got {script!r}")
        try:  # pragma: no cover - optional dependency
            from opencc import OpenCC  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Script unification requires OpenCC. Install it with "
                "`pip install opencc-python-reimplemented`, or pass script=None "
                "if your references and hypotheses are already in one script."
            ) from exc
        return OpenCC(script)

    # -- individual steps -------------------------------------------------

    @staticmethod
    def _halfwidth(text: str) -> str:
        # NFKC maps full-width ASCII and most compatibility forms to half-width.
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _strip_tags(text: str) -> str:
        text = _TAG_RE.sub(" ", text)
        text = _PLACEHOLDER_RE.sub(" ", text)
        return text

    def _numbers(self, text: str) -> str:
        text = english_number_words_to_digits(text)
        if self._cn2an is not None:  # pragma: no cover - optional dependency
            try:
                text = self._cn2an.transform(text, "cn2an")
            except Exception:
                # cn2an raises on plenty of harmless inputs; a failed rewrite
                # must never take down an evaluation run.
                pass
        return text

    # -- pipeline ---------------------------------------------------------

    def __call__(self, text: str) -> str:
        if text is None:
            return ""
        cfg = self.config
        out = str(text)
        for junk in cfg.drop_strings:
            out = out.replace(junk, " ")
        if cfg.remove_tags:
            out = self._strip_tags(out)
        if cfg.remove_brackets:
            out = _BRACKET_RE.sub(" ", out)
        if cfg.to_halfwidth:
            out = self._halfwidth(out)
        if self._converter is not None:
            out = self._converter.convert(out)
        if cfg.normalize_numbers:
            out = self._numbers(out)
        if cfg.lowercase:
            out = out.lower()
        if cfg.strip_accents:
            from .tokenizer import strip_accents

            out = strip_accents(out)
        if cfg.remove_punct:
            out = out.translate(_TRANS_DELETE)
            out = out.translate(_TRANS_SPACE)
        return collapse_whitespace(out)


def english_number_words_to_digits(text: str) -> str:
    """Fold runs of English number words into Arabic digits.

    Handles the common newsreader forms -- "twenty twenty six",
    "three hundred and fifty", "two thousand". It is intentionally simple: it
    only ever rewrites a maximal run of number words, and a run it cannot parse
    is left untouched rather than guessed at.
    """
    parts = re.split(r"(\W+)", text)
    out: List[str] = []
    buf: List[str] = []
    # Text seen since the last number word. If more number words follow it is
    # interior to the run and gets absorbed; if the run ends it is emitted
    # verbatim, so "five and apples" does not lose its "and".
    pending: List[str] = []

    def flush() -> None:
        if buf:
            value = _parse_en_number(buf)
            out.append(str(value) if value is not None else " ".join(buf))
            buf.clear()
        out.extend(pending)
        pending.clear()

    for tok in parts:
        low = tok.lower()
        if low in _EN_UNITS or low in _EN_TENS or low in _EN_SCALES:
            pending.clear()
            buf.append(low)
        elif buf and (tok.strip() == "" or low == "and"):
            pending.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return "".join(out)


def _parse_en_number(words: List[str]) -> Optional[int]:
    """Parse a run of number words, or return None if the run is not a number.

    "twenty twenty six" is a year, not 20 + 20 + 6, so a run that restarts at a
    tens/unit boundary is concatenated rather than summed.
    """
    if not words:
        return None
    total = 0
    current = 0
    seen = False
    for w in words:
        if w in _EN_UNITS:
            current += _EN_UNITS[w]
            seen = True
        elif w in _EN_TENS:
            if current and current % 100 != 0:
                # "twenty twenty" -- a new group started, concatenate instead.
                return None
            current += _EN_TENS[w]
            seen = True
        elif w in _EN_SCALES:
            scale = _EN_SCALES[w]
            if current == 0:
                current = 1
            if scale >= 1000:
                total += current * scale
                current = 0
            else:
                current *= scale
            seen = True
        else:
            return None
    return total + current if seen else None
