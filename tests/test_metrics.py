"""Behavioural tests for the metrics.

Each case is a failure mode that has actually been observed in Cantonese-English
evaluation, encoded as the smallest example that exhibits it.
"""

from mce.metrics import aggregate, score_corpus, score_utterance
from mce.report import diagnose
from mce.tokenizer import EN, ZH


def one(ref, hyp, **kw):
    return score_utterance("u1", ref, hyp, **kw)


def corpus(*pairs, **kw):
    return score_corpus([(f"u{i}", r, h) for i, (r, h) in enumerate(pairs)], **kw)


class TestPerfect:
    def test_identical_transcription_scores_zero(self):
        m, _ = corpus(("我今日好busy", "我今日好busy"))
        assert m.mer == 0.0
        assert m.cer_zh == 0.0
        assert m.wer_en == 0.0
        assert m.pier == 0.0
        assert m.runaway_rate == 0.0

    def test_switch_ratio_is_one_when_switches_are_reproduced(self):
        m, _ = corpus(("我好busy啊", "我好busy啊"))
        assert m.switch_ratio == 1.0


class TestLanguageOmission:
    """The failure CER alone cannot see."""

    REF = "我今日好busy要開meeting"
    HYP = "我今日好要開"

    def test_english_is_deleted_wholesale(self):
        r = one(self.REF, self.HYP)
        assert r.n_ref == 8
        assert r.lang_ref[ZH] == 6
        assert r.lang_ref[EN] == 2
        assert r.dels == 2
        assert r.subs == 0
        assert r.ins == 0

    def test_cer_looks_perfect_while_english_is_completely_lost(self):
        m, _ = corpus((self.REF, self.HYP))
        assert m.cer_zh == 0.0          # Chinese is flawless...
        assert m.wer_en == 1.0          # ...and every English token is gone
        assert m.mer == 2 / 8           # MER only mildly penalised: 25%
        assert m.en_omission_rate == 1.0
        assert m.en_lost_rate == 1.0

    def test_diagnosis_names_the_failure(self):
        m, _ = corpus((self.REF, self.HYP))
        assert "language omission" in diagnose(m)


class TestTranslationInsteadOfTranscription:
    REF = "我要開meeting"
    HYP = "我要開會議"

    def test_english_token_replaced_by_chinese_is_counted(self):
        r = one(self.REF, self.HYP)
        assert r.en_sub_by_zh == 1
        assert r.subs == 1
        assert r.ins == 1

    def test_rate_and_diagnosis(self):
        m, _ = corpus((self.REF, self.HYP))
        assert m.en_sub_by_zh_rate == 1.0
        assert "translating instead of transcribing" in diagnose(m)


class TestSwitchPointConcentration:
    def test_pier_exceeds_mer_when_errors_sit_on_the_boundary(self):
        # 我 好 busy 啊 -> deleting 'busy' destroys both switch points
        m, _ = corpus(("我好busy啊", "我好啊"))
        assert m.mer == 1 / 4
        assert m.n_poi == 3
        assert m.pier == 1 / 3
        assert m.pier > m.mer

    def test_errors_away_from_the_boundary_do_not_inflate_pier(self):
        # 我今日好busy : switch region is {好, busy}; the error is on 今
        m, _ = corpus(("我今日好busy", "我尋日好busy"))
        assert m.mer == 1 / 5
        assert m.pier == 0.0

    def test_wider_window_captures_more_context(self):
        m, _ = corpus(("我今日好busy", "我尋日好busy"), poi_window=3)
        assert m.pier > 0.0


class TestRunawayDecoding:
    """The shape that produces a reported error rate above 1.0."""

    def test_repetition_is_insertions_not_substitutions(self):
        r = one("我好攰", "我好攰我好攰我好攰")
        assert r.ins == 6
        assert r.subs == 0
        assert r.dels == 0

    def test_mer_exceeds_one_and_is_flagged(self):
        m, _ = corpus(("我好攰", "我好攰我好攰我好攰"))
        assert m.mer == 2.0
        assert m.length_ratio == 3.0
        assert m.runaway_rate == 1.0
        report = diagnose(m)
        assert "decoder collapse" in report
        assert "language='yue'" in report

    def test_mer_above_one_is_called_out_as_probably_a_scoring_bug(self):
        m, _ = corpus(("我好攰", "我好攰我好攰我好攰"))
        assert "scoring bug" in diagnose(m)


class TestEmptyAndDegenerate:
    def test_empty_hypothesis_is_charged_as_full_deletion(self):
        m, _ = corpus(("我今日好busy要開meeting", ""))
        assert m.mer == 1.0
        assert m.dels == 8
        assert m.en_omission_rate == 1.0

    def test_monolingual_corpus_reports_english_rate_as_undefined_not_zero(self):
        m, _ = corpus(("我今日好攰", "我今日好攰"))
        assert m.lang_ref[EN] == 0
        assert m.wer_en is None      # not 0.0, which would read as perfect
        assert m.pier is None        # no switch points exist

    def test_empty_reference_and_hypothesis(self):
        m, _ = corpus(("", ""))
        assert m.mer is None
        assert m.n_ref_tokens == 0


class TestAggregation:
    def test_corpus_rate_is_ratio_of_sums_not_mean_of_rates(self):
        # A 1-token utterance with 1 error and a 9-token utterance with 0.
        # Mean of per-utterance rates would be 0.5; the honest answer is 0.1.
        m, _ = corpus(("我", "你"), ("我今日好攰真係好攰啊", "我今日好攰真係好攰啊"))
        assert m.n_ref_tokens == 11
        assert m.mer == 1 / 11

    def test_aggregate_matches_manual_sum(self):
        results = [
            score_utterance("a", "我好busy", "我好"),
            score_utterance("b", "開meeting", "開meeting"),
        ]
        m = aggregate(results)
        assert m.n_utts == 2
        assert m.subs + m.dels + m.ins == sum(r.errors for r in results)
        assert m.n_ref_tokens == sum(r.n_ref for r in results)

    def test_utterances_with_no_english_do_not_dilute_the_omission_rate(self):
        m, _ = corpus(
            ("我今日好busy", "我今日好"),      # English lost
            ("我今日好攰", "我今日好攰"),      # no English at all
        )
        # denominator is utterances that *had* English, not all utterances
        assert m.utts_with_en_ref == 1
        assert m.en_omission_rate == 1.0


class TestUtteranceDetail:
    def test_per_utterance_dict_round_trips(self):
        r = one("我好busy", "我好")
        d = r.to_dict()
        assert d["id"] == "u1"
        assert d["mer"] == r.mer
        assert d["lang_ref"][EN] == 1

    def test_cmi_is_reported_from_the_reference(self):
        r = one("我好 busy today ok", "whatever")
        assert 0.0 < r.cmi_ref <= 0.5
