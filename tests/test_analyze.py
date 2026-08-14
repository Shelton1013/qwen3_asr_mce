"""Error inventory: which tokens go wrong, and into what.

The corpus metrics say how much is wrong; this says what. Its two jobs are to
supply the vocabulary for DPO negatives and to separate reference noise
(homophone spellings) from real recognition errors.
"""

import json

import pytest

from mce.analyze import (
    build_inventory,
    format_inventory,
    inventory_to_dict,
    looks_orthographic,
)
from mce.tokenizer import EN, ZH


def inv(*pairs, **kw):
    return build_inventory(list(pairs), **kw)


class TestSubstitutionPairs:
    def test_english_replaced_by_chinese_is_captured(self):
        # The aligner splits this into sub(typhoon->台) + ins(风); merging the
        # run is what makes the pair usable as a DPO negative.
        i = inv(("上week的typhoon好猛", "上week的台风好猛"))
        assert i.pairs(EN, ZH)[("typhoon", "台风")] == 1

    def test_multi_word_english_spans_are_kept_whole(self):
        i = inv(("the sun真令人feel good", "特新真令人feel good"))
        assert i.pairs(EN, ZH)[("the sun", "特新")] == 1

    def test_semantic_and_phonetic_failures_are_both_visible(self):
        i = inv(
            ("上week的typhoon好猛", "上week的台风好猛"),      # translation
            ("the sun真令人feel good", "特新真令人feel good"),  # transliteration
        )
        assert set(i.pairs(EN, ZH)) == {("typhoon", "台风"), ("the sun", "特新")}

    def test_english_replaced_by_english(self):
        i = inv(("crispy嘅外皮", "whisper嘅外皮"))
        assert i.pairs(EN, EN)[("crispy", "whisper")] == 1

    def test_chinese_single_char_swaps_are_separated(self):
        i = inv(("我钟意呢个", "我中意呢个"))
        assert i.pairs(ZH, ZH)[("钟", "中")] == 1

    def test_counts_accumulate_across_utterances(self):
        i = inv(
            ("我好like呢间", "我好钟意呢间"),
            ("我好like个个", "我好钟意个个"),
        )
        assert i.pairs(EN, ZH)[("like", "钟意")] == 2
        assert i.n_utts == 2

    def test_directions_are_kept_apart(self):
        i = inv(("食meeting", "食会议"), ("食meeting", "食meating"))
        assert i.total(EN, ZH) == 1
        assert i.total(EN, EN) == 1


class TestDeletionsAndInsertions:
    def test_deleted_english_is_tallied_by_token(self):
        i = inv(("最近的air quality好差", "最近的quality好差"))
        assert i.deletions[EN]["air"] == 1
        assert i.lost_en_tokens["air"] == 1

    def test_inserted_material_is_tallied_by_language(self):
        i = inv(("我好攰", "我好攰啊"))
        assert i.insertions[ZH]["啊"] == 1

    def test_empty_hypothesis_is_one_mixed_deletion_span(self):
        i = inv(("我好busy", ""))
        assert i.deletions["mixed"]["我好 busy"] == 1
        assert not i.substitutions

    def test_english_inside_a_long_deletion_is_still_counted(self):
        # The span table records one mixed deletion; the per-token tally is what
        # keeps 'busy' and 'meeting' visible.
        i = inv(("我今日好busy要開meeting", ""))
        assert i.lost_en_tokens["busy"] == 1
        assert i.lost_en_tokens["meeting"] == 1

    def test_substituted_english_counts_as_lost_too(self):
        i = inv(("上week的typhoon好猛", "上week的台风好猛"))
        assert i.lost_en_tokens["typhoon"] == 1
        assert "week" not in i.lost_en_tokens


class TestPoiRestriction:
    def test_boundary_substitutions_are_a_subset(self):
        # 我 好 busy 啊 -> the English sits on both switch points
        i = inv(("我好busy啊", "我好忙啊"))
        assert i.poi_substitutions[(EN, ZH)][("busy", "忙")] == 1

    def test_substitutions_away_from_a_boundary_are_excluded(self):
        # error on 今, far from the single switch before 'busy'
        i = inv(("我今日好攰真係好busy", "我尋日好攰真係好busy"), poi_window=1)
        assert i.pairs(ZH, ZH)[("今", "尋")] == 1
        assert i.poi_substitutions.get((ZH, ZH), {}) == {}


class TestOrthographicHeuristic:
    def test_single_character_pairs_are_flagged(self):
        assert looks_orthographic("地", "哋")
        assert looks_orthographic("钟", "中")

    def test_multi_character_pairs_are_not(self):
        assert not looks_orthographic("台风", "typhoon")
        assert not looks_orthographic("busy", "忙")

    def test_it_is_only_a_candidate_filter(self):
        # 疯/风 is a real error, not a spelling variant -- the heuristic cannot
        # tell, and the report says so rather than normalising it away.
        assert looks_orthographic("疯", "风")


class TestReport:
    PAIRS = [
        ("上week的typhoon好猛", "上week的台风好猛"),
        ("我钟意饮咖啡", "我中意饮咖啡"),
        ("我钟意食嘢", "我中意食嘢"),
        ("crispy嘅外皮", "whisper嘅外皮"),
        ("最近的air quality好差", "最近的quality好差"),
    ]

    def test_every_section_renders(self):
        text = format_inventory(build_inventory(self.PAIRS), min_count=1)
        for heading in (
            "English replaced by Chinese",
            "English replaced by English",
            "English tokens lost (deleted or substituted)",
            "orthographic variant candidates",
            "switch points only",
            "totals",
        ):
            assert heading in text

    def test_repeated_pairs_surface_in_the_orthographic_table(self):
        text = format_inventory(build_inventory(self.PAIRS), min_count=2)
        assert "钟" in text and "中" in text

    def test_min_count_hides_singletons(self):
        text = format_inventory(build_inventory(self.PAIRS), min_count=99)
        assert "nothing above the minimum count" in text

    def test_utterance_count_is_reported(self):
        assert "over 5 utterances" in format_inventory(build_inventory(self.PAIRS))

    def test_empty_input_does_not_crash(self):
        assert "over 0 utterances" in format_inventory(build_inventory([]))


class TestJsonExport:
    def test_shape_is_serialisable_and_keyed_by_direction(self):
        i = inv(("我好like呢间", "我好钟意呢间"), ("食meeting", "食meating"))
        d = inventory_to_dict(i)
        payload = json.loads(json.dumps(d, ensure_ascii=False))
        assert payload["n_utts"] == 2
        assert "en->zh" in payload["substitutions"]
        assert payload["substitutions"]["en->zh"][0]["ref"] == "like"

    def test_min_count_filters_the_export(self):
        i = inv(("食meeting", "食会议"))
        assert inventory_to_dict(i, min_count=2)["substitutions"]["en->zh"] == []


class TestCliGuardrails:
    def _dump(self, tmp_path, name):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(3):
                fh.write(json.dumps(
                    {"id": f"u{i}", "ref": "我好like呢间", "hyp": "我好钟意呢间"},
                    ensure_ascii=False) + "\n")
        return path

    def test_analysing_a_test_dump_is_refused_by_default(self, tmp_path, capsys):
        from mce.cli import main

        path = self._dump(tmp_path, "qwen.test.utts.jsonl")
        assert main(["analyze", "--utts", str(path)]) == 1
        assert "contamination" in capsys.readouterr().out

    def test_the_refusal_can_be_overridden_deliberately(self, tmp_path):
        from mce.cli import main

        path = self._dump(tmp_path, "qwen.test.utts.jsonl")
        assert main(["analyze", "--utts", str(path), "--allow-test"]) == 0

    def test_a_dev_dump_runs_without_complaint(self, tmp_path, capsys):
        from mce.cli import main

        path = self._dump(tmp_path, "qwen.dev.utts.jsonl")
        assert main(["analyze", "--utts", str(path), "--min-count", "1"]) == 0
        assert "contamination" not in capsys.readouterr().out

    def test_wrong_file_type_is_a_clear_error(self, tmp_path):
        from mce.cli import main

        path = tmp_path / "dev.hyp.jsonl"
        path.write_text(json.dumps({"id": "u0", "hyp": "我好"}) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="--out-utts"):
            main(["analyze", "--utts", str(path)])

    def test_json_and_text_outputs_are_written(self, tmp_path):
        from mce.cli import main

        path = self._dump(tmp_path, "dev.utts.jsonl")
        out_json = tmp_path / "pairs.json"
        out_text = tmp_path / "report.txt"
        main([
            "analyze", "--utts", str(path), "--min-count", "1",
            "--out-json", str(out_json), "--out-text", str(out_text),
        ])
        assert json.loads(out_json.read_text(encoding="utf-8"))["n_utts"] == 3
        assert "English replaced by Chinese" in out_text.read_text(encoding="utf-8")
