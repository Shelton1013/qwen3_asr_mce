"""Detecting a hypothesis answered in a third language.

Observed on Qwen3-ASR-0.6B: a Cantonese-English utterance came back in Thai.

    REF  寻找你嘅optimal study environment 有啲人喜欢在quiet environments学习
    HYP  คําแนะนําในการ optimal study environment อย่าที่อ่านให้ฟุ้งใจ quiet environment

Every per-language metric is keyed off the *reference*, which contains no Thai
to be wrong about, so CER_zh and WER_en stay quiet and the damage surfaces only
as insertions. Language identification failing outright is a different kind of
defect from mis-recognising words, and it needs its own signal.
"""

from mce.metrics import (
    FOREIGN_SCRIPT_MIN_TOKENS,
    FOREIGN_SCRIPT_THRESHOLD,
    score_corpus,
    score_utterance,
)
from mce.report import diagnose, format_metrics
from mce.tokenizer import OTHER, tokenize

THAI = "คําแนะนําในการ optimal study environment อย่าที่อ่านให้ฟุ้งใจ"
REF = "寻找你嘅optimal study environment 有啲人喜欢在quiet environments学习"


class TestTokenisation:
    def test_thai_characters_are_tagged_other(self):
        langs = {t.lang for t in tokenize("คําแนะนํา")}
        assert langs == {OTHER}

    def test_latin_inside_thai_is_still_english(self):
        toks = tokenize(THAI)
        assert any(t.text == "optimal" and t.lang == "en" for t in toks)
        assert any(t.lang == OTHER for t in toks)


class TestUtteranceFlag:
    def test_a_thai_hypothesis_is_flagged(self):
        r = score_utterance("u", REF, THAI)
        assert r.foreign_script

    def test_a_normal_hypothesis_is_not(self):
        r = score_utterance("u", REF, "寻找你嘅optimal study environment 有啲人喜欢")
        assert not r.foreign_script

    def test_hypothesis_language_counts_are_recorded(self):
        r = score_utterance("u", REF, THAI)
        assert r.lang_hyp[OTHER] > 0
        assert r.lang_hyp["en"] > 0

    def test_short_hypotheses_are_not_judged(self):
        # Too few tokens for the share to mean anything.
        r = score_utterance("u", "我好攰", "คํา")
        assert r.n_hyp < FOREIGN_SCRIPT_MIN_TOKENS
        assert not r.foreign_script

    def test_a_little_foreign_text_is_below_the_threshold(self):
        hyp = "我今日好攰真係好攰啊唔想返工คํา"
        r = score_utterance("u", REF, hyp)
        assert r.lang_hyp[OTHER] / r.n_hyp < FOREIGN_SCRIPT_THRESHOLD
        assert not r.foreign_script

    def test_an_empty_hypothesis_is_not_foreign(self):
        assert not score_utterance("u", REF, "").foreign_script


class TestCorpusMetrics:
    def test_rate_counts_affected_utterances(self):
        pairs = [(f"u{i}", REF, THAI if i < 3 else REF) for i in range(10)]
        m, _ = score_corpus(pairs)
        assert m.utts_foreign_script == 3
        assert m.foreign_script_rate == 0.3

    def test_zero_when_nothing_is_foreign(self):
        m, _ = score_corpus([("u", REF, REF)])
        assert m.foreign_script_rate == 0.0

    def test_token_level_rate_is_also_reported(self):
        m, _ = score_corpus([("u", REF, THAI)])
        assert (m.hyp_foreign_token_rate or 0) > 0.3

    def test_the_per_language_rates_stay_blind_to_it(self):
        # The point of the new signal: the old ones cannot see this.
        m, _ = score_corpus([("u", REF, THAI)])
        assert m.foreign_script_rate == 1.0
        # WER_en looks almost healthy because the English words survived
        assert (m.wer_en or 1.0) < 0.5

    def test_it_appears_in_the_serialised_metrics(self):
        m, _ = score_corpus([("u", REF, THAI)])
        d = m.to_dict()
        assert d["foreign_script_rate"] == 1.0
        assert "hyp_foreign_token_rate" in d


class TestReporting:
    def test_diagnosis_names_language_identification(self):
        m, _ = score_corpus([(f"u{i}", REF, THAI) for i in range(5)])
        report = diagnose(m)
        assert "language identification failing outright" in report
        assert "CER_zh and WER_en cannot see it" in report

    def test_the_sanity_block_shows_the_count(self):
        m, _ = score_corpus([(f"u{i}", REF, THAI if i else REF) for i in range(4)])
        text = format_metrics(m, title="x")
        assert "foreign script rate" in text
        assert "answered in a third language" in text

    def test_clean_runs_say_nothing_about_it(self):
        m, _ = score_corpus([("u", REF, REF)])
        assert "third language" not in diagnose(m)

    def test_identifying_the_cause_suppresses_the_misleading_generic_notes(self):
        # A Thai hypothesis is long and error-dense, so the runaway and
        # "MER > 1 is probably a scoring bug" notes both trip -- and both point
        # somewhere wrong, one of them at a Whisper flag Qwen does not have.
        m, _ = score_corpus([(f"u{i}", REF, THAI) for i in range(5)])
        report = diagnose(m)
        assert m.mer > 1.0 and m.runaway_rate > 0.05
        assert "language='yue'" not in report
        assert "scoring bug" not in report
        assert "downstream of this" in report

    def test_those_notes_still_fire_when_the_cause_is_unknown(self):
        m, _ = score_corpus([("u", "我好攰", "我好攰我好攰我好攰")])
        report = diagnose(m)
        assert "language='yue'" in report
        assert "scoring bug" in report

    def test_comparison_table_has_the_column(self):
        from mce.report import markdown_table

        m, _ = score_corpus([("u", REF, THAI)])
        table = markdown_table([("qwen3-asr-0.6b", m)])
        assert "Foreign %" in table
        header_cols = table.splitlines()[0].count("|") - 1
        row_cols = table.splitlines()[2].count("|") - 1
        assert header_cols == row_cols
