"""Code-switching ASR metrics.

The headline number is MER, but MER alone cannot distinguish the two things you
most need to tell apart in Cantonese-English:

* a model that transcribes Cantonese well and silently drops the English, and
* a model that handles both.

Cantonese is typically 70-85% of the characters in a CS utterance, so the first
model still scores a respectable MER (and an excellent CER).  The per-language
rates, the switch-point rate and the omission rates exist to expose that.

Metric definitions
------------------
``MER``     (S+D+I)/N over the mixed tokenisation: one token per CJK character,
            one per English word.  Digits and residual symbols count towards N.
``CER_zh``  Same formula restricted to Chinese: errors whose reference token is
            Chinese, plus insertions of Chinese material, over the number of
            Chinese reference tokens.
``WER_en``  The same, restricted to English.
``PIER``    Point-of-Interest Error Rate.  Errors falling inside a +/-``window``
            token region around each Cantonese<->English boundary in the
            reference, over the size of that region.  This is the number that
            actually measures code-switching ability; expect it to be markedly
            higher than MER.
``en_omission_rate``   Fraction of utterances whose reference contains English
            but whose hypothesis contains none at all.  The clearest signature
            of the "language omission" failure mode.
``en_sub_by_zh_rate``  Fraction of English reference tokens replaced by Chinese
            material.  The signature of "translation instead of transcription".
``runaway_rate``       Fraction of utterances where the hypothesis is more than
            ``runaway_ratio`` times the reference length -- a direct detector for
            decoder collapse / repetition loops, which is the usual reason a
            reported WER exceeds 1.0.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .align import DELETE, EQUAL, INSERT, SUB, Op, align
from .tokenizer import (
    CS_LANGS,
    EN,
    NUM,
    OTHER,
    ZH,
    Token,
    code_mixing_index,
    count_langs,
    poi_indices,
    switch_points,
    token_texts,
    tokenize,
)

_LANGS = (ZH, EN, NUM, OTHER)


@dataclass
class UtteranceResult:
    """Per-utterance scoring detail, dumped to JSONL for error analysis."""

    id: str
    ref: str
    hyp: str
    n_ref: int
    n_hyp: int
    hits: int
    subs: int
    dels: int
    ins: int
    lang_ref: Dict[str, int]
    lang_err: Dict[str, int]
    lang_sub: Dict[str, int]
    lang_del: Dict[str, int]
    lang_ins: Dict[str, int]
    n_poi: int
    poi_err: int
    n_switch_ref: int
    n_switch_hyp: int
    en_omitted: bool
    zh_omitted: bool
    en_sub_by_zh: int
    en_lost: int
    cmi_ref: float
    length_ratio: float

    @property
    def errors(self) -> int:
        return self.subs + self.dels + self.ins

    @property
    def mer(self) -> Optional[float]:
        return _safe_div(self.errors, self.n_ref)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mer"] = self.mer
        return d


@dataclass
class CorpusMetrics:
    """Aggregated over the corpus by summing counts, never by averaging rates.

    Averaging per-utterance rates would let a two-token utterance outweigh a
    fifty-token one; every rate below is a corpus-level ratio of summed counts.
    The two exceptions are explicitly named ``mean_*`` / ``*_rate`` over
    utterances, where the utterance genuinely is the unit of interest.
    """

    n_utts: int = 0
    n_ref_tokens: int = 0
    n_hyp_tokens: int = 0
    hits: int = 0
    subs: int = 0
    dels: int = 0
    ins: int = 0
    lang_ref: Counter = field(default_factory=Counter)
    lang_err: Counter = field(default_factory=Counter)
    lang_sub: Counter = field(default_factory=Counter)
    lang_del: Counter = field(default_factory=Counter)
    lang_ins: Counter = field(default_factory=Counter)
    n_poi: int = 0
    poi_err: int = 0
    n_switch_ref: int = 0
    n_switch_hyp: int = 0
    utts_with_en_ref: int = 0
    utts_with_zh_ref: int = 0
    utts_en_omitted: int = 0
    utts_zh_omitted: int = 0
    en_sub_by_zh: int = 0
    en_lost: int = 0
    utts_runaway: int = 0
    sum_cmi_ref: float = 0.0

    # -- derived ----------------------------------------------------------

    @property
    def errors(self) -> int:
        return self.subs + self.dels + self.ins

    @property
    def mer(self) -> Optional[float]:
        return _safe_div(self.errors, self.n_ref_tokens)

    @property
    def cer_zh(self) -> Optional[float]:
        return _safe_div(self.lang_err[ZH], self.lang_ref[ZH])

    @property
    def wer_en(self) -> Optional[float]:
        return _safe_div(self.lang_err[EN], self.lang_ref[EN])

    @property
    def pier(self) -> Optional[float]:
        return _safe_div(self.poi_err, self.n_poi)

    @property
    def sub_rate(self) -> Optional[float]:
        return _safe_div(self.subs, self.n_ref_tokens)

    @property
    def del_rate(self) -> Optional[float]:
        return _safe_div(self.dels, self.n_ref_tokens)

    @property
    def ins_rate(self) -> Optional[float]:
        return _safe_div(self.ins, self.n_ref_tokens)

    @property
    def en_omission_rate(self) -> Optional[float]:
        return _safe_div(self.utts_en_omitted, self.utts_with_en_ref)

    @property
    def zh_omission_rate(self) -> Optional[float]:
        return _safe_div(self.utts_zh_omitted, self.utts_with_zh_ref)

    @property
    def en_sub_by_zh_rate(self) -> Optional[float]:
        return _safe_div(self.en_sub_by_zh, self.lang_ref[EN])

    @property
    def en_lost_rate(self) -> Optional[float]:
        return _safe_div(self.en_lost, self.lang_ref[EN])

    @property
    def runaway_rate(self) -> Optional[float]:
        return _safe_div(self.utts_runaway, self.n_utts)

    @property
    def switch_ratio(self) -> Optional[float]:
        """Switches produced vs switches expected. Well below 1.0 means the
        model is flattening the utterance into a single language."""
        return _safe_div(self.n_switch_hyp, self.n_switch_ref)

    @property
    def length_ratio(self) -> Optional[float]:
        return _safe_div(self.n_hyp_tokens, self.n_ref_tokens)

    @property
    def mean_cmi_ref(self) -> Optional[float]:
        return _safe_div(self.sum_cmi_ref, self.n_utts)

    def lang_rate(self, lang: str) -> Optional[float]:
        return _safe_div(self.lang_err[lang], self.lang_ref[lang])

    def to_dict(self) -> dict:
        return {
            "n_utts": self.n_utts,
            "n_ref_tokens": self.n_ref_tokens,
            "n_hyp_tokens": self.n_hyp_tokens,
            "mer": self.mer,
            "cer_zh": self.cer_zh,
            "wer_en": self.wer_en,
            "pier": self.pier,
            "sub_rate": self.sub_rate,
            "del_rate": self.del_rate,
            "ins_rate": self.ins_rate,
            "subs": self.subs,
            "dels": self.dels,
            "ins": self.ins,
            "hits": self.hits,
            "en_omission_rate": self.en_omission_rate,
            "zh_omission_rate": self.zh_omission_rate,
            "en_sub_by_zh_rate": self.en_sub_by_zh_rate,
            "en_lost_rate": self.en_lost_rate,
            "runaway_rate": self.runaway_rate,
            "switch_ratio": self.switch_ratio,
            "length_ratio": self.length_ratio,
            "mean_cmi_ref": self.mean_cmi_ref,
            "n_poi": self.n_poi,
            "n_switch_ref": self.n_switch_ref,
            "n_switch_hyp": self.n_switch_hyp,
            "lang_ref_tokens": dict(self.lang_ref),
            "lang_error_rate": {l: self.lang_rate(l) for l in _LANGS},
        }


def _safe_div(num: float, den: float) -> Optional[float]:
    """Ratios with an empty denominator are undefined, not zero.

    Reporting 0.0 for "English error rate on a corpus with no English" would be
    read as a perfect score, which is the opposite of the truth.
    """
    if not den:
        return None
    return num / den


def score_utterance(
    utt_id: str,
    ref_text: str,
    hyp_text: str,
    poi_window: int = 1,
    runaway_ratio: float = 2.0,
) -> UtteranceResult:
    """Tokenise, align and count a single (reference, hypothesis) pair.

    Both strings are expected to be normalised already -- see
    :class:`mce.normalize.Normalizer`.  Scoring never normalises implicitly,
    so that what you scored is exactly what you can inspect in the dump.
    """
    ref_tokens: List[Token] = tokenize(ref_text)
    hyp_tokens: List[Token] = tokenize(hyp_text)
    ops = align(token_texts(ref_tokens), token_texts(hyp_tokens))

    lang_ref = count_langs(ref_tokens)
    lang_hyp = count_langs(hyp_tokens)
    lang_sub: Counter = Counter()
    lang_del: Counter = Counter()
    lang_ins: Counter = Counter()

    hits = subs = dels = ins = 0
    en_sub_by_zh = 0
    en_lost = 0

    poi = poi_indices(ref_tokens, window=poi_window)
    poi_err = 0

    for op in ops:
        if op.op == EQUAL:
            hits += 1
            continue
        if op.op == SUB:
            subs += 1
            lang = ref_tokens[op.ref_idx].lang
            lang_sub[lang] += 1
            if lang == EN:
                en_lost += 1
                if hyp_tokens[op.hyp_idx].lang == ZH:
                    en_sub_by_zh += 1
        elif op.op == DELETE:
            dels += 1
            lang = ref_tokens[op.ref_idx].lang
            lang_del[lang] += 1
            if lang == EN:
                en_lost += 1
        else:  # INSERT
            ins += 1
            lang_ins[hyp_tokens[op.hyp_idx].lang] += 1

        if _in_poi(op, poi):
            poi_err += 1

    lang_err = Counter()
    for lang in _LANGS:
        lang_err[lang] = lang_sub[lang] + lang_del[lang] + lang_ins[lang]

    n_ref = len(ref_tokens)
    n_hyp = len(hyp_tokens)
    return UtteranceResult(
        id=utt_id,
        ref=ref_text,
        hyp=hyp_text,
        n_ref=n_ref,
        n_hyp=n_hyp,
        hits=hits,
        subs=subs,
        dels=dels,
        ins=ins,
        lang_ref={l: lang_ref.get(l, 0) for l in _LANGS},
        lang_err={l: lang_err[l] for l in _LANGS},
        lang_sub={l: lang_sub[l] for l in _LANGS},
        lang_del={l: lang_del[l] for l in _LANGS},
        lang_ins={l: lang_ins[l] for l in _LANGS},
        n_poi=len(poi),
        poi_err=poi_err,
        n_switch_ref=len(switch_points(ref_tokens)),
        n_switch_hyp=len(switch_points(hyp_tokens)),
        en_omitted=lang_ref.get(EN, 0) > 0 and lang_hyp.get(EN, 0) == 0,
        zh_omitted=lang_ref.get(ZH, 0) > 0 and lang_hyp.get(ZH, 0) == 0,
        en_sub_by_zh=en_sub_by_zh,
        en_lost=en_lost,
        cmi_ref=code_mixing_index(ref_tokens),
        length_ratio=(n_hyp / n_ref) if n_ref else float("inf") if n_hyp else 1.0,
    )


def _in_poi(op: Op, poi: set) -> bool:
    """Does this edit fall inside a switch-point window?

    Substitutions and deletions are tested on their own reference index.  An
    insertion has no reference token, so it counts if it sits immediately
    before or after a POI token -- inserted material at a boundary is precisely
    the "model hesitates at the switch" error we want to catch.
    """
    if op.op == INSERT:
        return op.ref_pos in poi or (op.ref_pos - 1) in poi
    return op.ref_idx in poi


def aggregate(
    results: Sequence[UtteranceResult], runaway_ratio: float = 2.0
) -> CorpusMetrics:
    """Sum per-utterance counts into corpus metrics."""
    m = CorpusMetrics()
    for r in results:
        m.n_utts += 1
        m.n_ref_tokens += r.n_ref
        m.n_hyp_tokens += r.n_hyp
        m.hits += r.hits
        m.subs += r.subs
        m.dels += r.dels
        m.ins += r.ins
        for lang in _LANGS:
            m.lang_ref[lang] += r.lang_ref.get(lang, 0)
            m.lang_err[lang] += r.lang_err.get(lang, 0)
            m.lang_sub[lang] += r.lang_sub.get(lang, 0)
            m.lang_del[lang] += r.lang_del.get(lang, 0)
            m.lang_ins[lang] += r.lang_ins.get(lang, 0)
        m.n_poi += r.n_poi
        m.poi_err += r.poi_err
        m.n_switch_ref += r.n_switch_ref
        m.n_switch_hyp += r.n_switch_hyp
        if r.lang_ref.get(EN, 0) > 0:
            m.utts_with_en_ref += 1
            if r.en_omitted:
                m.utts_en_omitted += 1
        if r.lang_ref.get(ZH, 0) > 0:
            m.utts_with_zh_ref += 1
            if r.zh_omitted:
                m.utts_zh_omitted += 1
        m.en_sub_by_zh += r.en_sub_by_zh
        m.en_lost += r.en_lost
        if r.n_ref and r.length_ratio > runaway_ratio:
            m.utts_runaway += 1
        m.sum_cmi_ref += r.cmi_ref
    return m


def score_corpus(
    pairs: Sequence,
    poi_window: int = 1,
    runaway_ratio: float = 2.0,
):
    """Score an iterable of ``(id, reference, hypothesis)`` triples.

    Returns ``(CorpusMetrics, list[UtteranceResult])``.
    """
    results = [
        score_utterance(
            utt_id, ref, hyp, poi_window=poi_window, runaway_ratio=runaway_ratio
        )
        for utt_id, ref, hyp in pairs
    ]
    return aggregate(results, runaway_ratio=runaway_ratio), results
