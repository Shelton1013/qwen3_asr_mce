"""Detecting audio paired with the wrong reference.

Real case from the MCE corpus, speaker 97: four adjacent pairs whose references
are crossed. The model transcribed both sides correctly and was charged ~100%
MER twice for it. Left in the training set, a crossed pair teaches the model to
map audio to unrelated text.
"""

import json

import pytest

from mce.alignment_check import (
    estimate_mer_impact,
    find_misalignments,
    find_swapped_pairs,
    format_report,
)

# Verbatim from the qwen3-asr-1.7b run.
SWAPPED = [
    {"id": "97_40", "ref": "running好重要 因为可以improve一下cardiovascular fitness",
     "hyp": "作为一个footballer 要够improve ball control同passing skills"},
    {"id": "97_41", "ref": "作为一个footballer 要够improve ball control同passing skills",
     "hyp": "running好重要 因为可以improve一下cardiovascular fitness"},
]

CLEAN = [
    {"id": "a1", "ref": "我今日好busy要開meeting", "hyp": "我今日好busy要開meeting"},
    {"id": "a2", "ref": "食呢個japanese sushi真係refreshing", "hyp": "食呢個japanese sushi真係refreshing"},
]


class TestSwapDetection:
    def test_a_crossed_pair_is_found(self):
        found = find_misalignments(SWAPPED)
        assert len(found) == 2
        assert {m.utt_id for m in found} == {"97_40", "97_41"}

    def test_each_side_points_at_the_other(self):
        found = find_misalignments(SWAPPED)
        by_id = {m.utt_id: m for m in found}
        assert by_id["97_40"].best_id == "97_41"
        assert by_id["97_41"].best_id == "97_40"

    def test_the_neighbour_fits_perfectly(self):
        found = find_misalignments(SWAPPED)
        assert all(m.best_mer == 0.0 for m in found)
        assert all(m.own_mer > 0.9 for m in found)

    def test_mutual_pairs_are_reported_once(self):
        pairs = find_swapped_pairs(find_misalignments(SWAPPED))
        assert len(pairs) == 1
        assert {pairs[0][0].utt_id, pairs[0][1].utt_id} == {"97_40", "97_41"}


class TestNoFalsePositives:
    def test_correct_transcriptions_are_not_flagged(self):
        assert find_misalignments(CLEAN) == []

    def test_ordinary_recognition_errors_are_not_flagged(self):
        rows = [
            {"id": "a", "ref": "我今日好busy要開meeting", "hyp": "我今日好busy要開會議"},
            {"id": "b", "ref": "食嘢真係開心", "hyp": "食野真係開心"},
        ]
        assert find_misalignments(rows) == []

    def test_two_equally_bad_rows_do_not_explain_each_other(self):
        # Both hypotheses are garbage; neither fits the other's reference well.
        rows = [
            {"id": "a", "ref": "我今日好busy要開meeting", "hyp": "aaa bbb ccc ddd"},
            {"id": "b", "ref": "食呢個japanese sushi", "hyp": "eee fff ggg hhh"},
        ]
        assert find_misalignments(rows) == []

    def test_similar_utterances_on_the_same_topic_need_a_real_gap(self):
        rows = [
            {"id": "a", "ref": "我好鍾意食sushi", "hyp": "我好鍾意飲coffee"},
            {"id": "b", "ref": "我好鍾意飲tea", "hyp": "我好鍾意食noodle"},
        ]
        found = find_misalignments(rows, ratio=0.2, max_best_mer=0.2)
        assert found == []


class TestWindowing:
    def test_matches_outside_the_window_are_ignored(self):
        rows = [SWAPPED[0]] + [dict(CLEAN[0], id=f"pad{i}") for i in range(8)] + [SWAPPED[1]]
        assert find_misalignments(rows, window=2) == []

    def test_a_wide_enough_window_finds_them(self):
        rows = [SWAPPED[0]] + [dict(CLEAN[0], id=f"pad{i}") for i in range(8)] + [SWAPPED[1]]
        found = find_misalignments(rows, window=20)
        assert len(found) == 2


class TestThresholds:
    def test_rows_that_already_match_well_are_skipped(self):
        found = find_misalignments(SWAPPED, max_own_mer=1.5)
        assert found == []       # own MER never reaches the (absurd) bar

    def test_a_perfect_match_survives_any_ratio(self):
        # best_mer == 0 is the strongest evidence available; no proportional
        # threshold should be able to filter it out.
        assert len(find_misalignments(SWAPPED, ratio=0.0)) == 2

    def test_the_ratio_gates_partial_matches(self):
        # own 100%, neighbour 12.5% -- flagged at the default, excluded once the
        # neighbour is required to fit nearly ten times better.
        rows = [
            {"id": "a", "ref": "我今日好攰要開meeting", "hyp": "聽日去銅鑼灣食麵"},
            {"id": "b", "ref": "聽日去銅鑼灣食飯", "hyp": "完全唔同"},
        ]
        assert [m.utt_id for m in find_misalignments(rows, ratio=0.5)] == ["a"]
        assert find_misalignments(rows, ratio=0.1) == []

    def test_empty_rows_are_ignored(self):
        rows = [{"id": "a", "ref": "", "hyp": ""}, {"id": "b", "ref": "我好攰", "hyp": ""}]
        assert find_misalignments(rows) == []


class TestImpact:
    def test_impact_is_a_share_of_all_reference_tokens(self):
        impact = estimate_mer_impact(SWAPPED, find_misalignments(SWAPPED))
        assert 0.9 < impact <= 1.0     # every token in this two-row corpus

    def test_impact_shrinks_as_the_corpus_grows(self):
        rows = SWAPPED + [dict(CLEAN[0], id=f"c{i}") for i in range(50)]
        impact = estimate_mer_impact(rows, find_misalignments(rows))
        assert 0 < impact < 0.1

    def test_impact_of_nothing_is_zero(self):
        assert estimate_mer_impact(CLEAN, []) == 0.0

    def test_empty_corpus_has_no_impact(self):
        assert estimate_mer_impact([], []) is None


class TestReport:
    def test_clean_corpus_says_so(self):
        assert "no misalignments found" in format_report(CLEAN, [])

    def test_swapped_pairs_are_shown_with_both_scores(self):
        found = find_misalignments(SWAPPED)
        text = format_report(SWAPPED, found)
        assert "mutually swapped pairs" in text
        assert "97_40" in text and "97_41" in text
        assert "corpus defects" in text

    def test_report_warns_against_training_on_them(self):
        text = format_report(SWAPPED, find_misalignments(SWAPPED))
        assert "map audio to unrelated text" in text


class TestCli:
    def test_end_to_end(self, tmp_path, capsys):
        from mce.cli import main

        utts = tmp_path / "dev.utts.jsonl"
        with open(utts, "w", encoding="utf-8") as fh:
            for row in SWAPPED:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        out = tmp_path / "suspect.json"
        assert main(["check-alignment", "--utts", str(utts), "--out-json", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "mutually swapped pairs" in printed
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload) == 2
        assert all(row["mutual"] for row in payload)

    def test_wrong_file_type_is_rejected(self, tmp_path):
        from mce.cli import main

        path = tmp_path / "h.jsonl"
        path.write_text(json.dumps({"id": "a", "hyp": "x"}) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="--out-utts"):
            main(["check-alignment", "--utts", str(path)])
