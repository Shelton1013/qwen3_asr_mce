"""Tests for the MCE dataset preparation script.

The two things that must never fail silently are positional pairing (row i of
the CSV really is wav i) and folder ordering (numeric, so folder 2 does not sort
after folder 10 and land in the wrong split). Both are covered here.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepare_mce.py"
_spec = importlib.util.spec_from_file_location("prepare_mce", _SCRIPT)
prepare_mce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_mce)


def make_dataset(tmp_path, folders, encoding="gb18030"):
    """folders: {index: [(topic, transcript), ...]}; creates wavs + csvs."""
    root = tmp_path / "MCE_Dataset"
    (root / "Audio").mkdir(parents=True)
    (root / "Text").mkdir(parents=True)
    for idx, rows in folders.items():
        adir = root / "Audio" / f"{idx}_MCE"
        adir.mkdir()
        for i in range(1, len(rows) + 1):
            (adir / f"{idx}_{i}.wav").write_bytes(b"RIFF fake wav")
        lines = ["Topic,Instance"]
        for topic, text in rows:
            # the corpus wraps every instance in doubled quotes
            lines.append(f'{topic},"""{text}"""')
        (root / "Text" / f"data_{idx}.csv").write_bytes(
            ("\n".join(lines) + "\n").encode(encoding)
        )
    return root


class TestCleanTranscript:
    def test_strips_the_wrapping_quote(self):
        assert prepare_mce.clean_transcript('"我好busy"') == "我好busy"

    def test_strips_curly_quotes_too(self):
        assert prepare_mce.clean_transcript("“我好busy”") == "我好busy"

    def test_collapses_internal_whitespace(self):
        assert prepare_mce.clean_transcript('"我好   busy\n啊"') == "我好 busy 啊"

    def test_leaves_unquoted_text_alone(self):
        assert prepare_mce.clean_transcript("我好busy") == "我好busy"

    def test_does_not_eat_an_unbalanced_quote(self):
        assert prepare_mce.clean_transcript('我好"busy') == '我好"busy'


class TestDecoding:
    def test_gbk_encoded_traditional_chinese_is_recovered(self, tmp_path):
        # This corpus stores Traditional characters in GBK, which is why the
        # encoding probe must try UTF-8 first and Big5 only after GB18030.
        path = tmp_path / "data_1.csv"
        path.write_bytes("Topic,Instance\n天氣,\"\"\"今日好凍\"\"\"\n".encode("gbk"))
        assert "今日好凍" in prepare_mce.decode_csv(path)

    def test_utf8_is_preferred_when_valid(self, tmp_path):
        path = tmp_path / "data_1.csv"
        path.write_text("Topic,Instance\n天氣,x\n", encoding="utf-8")
        assert "天氣" in prepare_mce.decode_csv(path)

    def test_explicit_encoding_is_honoured(self, tmp_path):
        path = tmp_path / "data_1.csv"
        path.write_bytes("天氣".encode("gbk"))
        assert prepare_mce.decode_csv(path, "gbk") == "天氣"


class TestReadFolder:
    def test_pairs_rows_with_wavs_in_numeric_order(self, tmp_path):
        rows = [("天氣", f"utterance {i}") for i in range(1, 13)]
        root = make_dataset(tmp_path, {1: rows})
        records, problems = prepare_mce.read_folder(
            root / "Audio" / "1_MCE", root / "Text" / "data_1.csv"
        )
        assert problems == []
        assert len(records) == 12
        # 1_2.wav must be the 2nd record, not sort after 1_10.wav
        assert records[1]["audio"].name == "1_2.wav"
        assert records[1]["text"] == "utterance 2"
        assert records[9]["audio"].name == "1_10.wav"
        assert records[9]["text"] == "utterance 10"

    def test_ids_and_metadata_are_populated(self, tmp_path):
        root = make_dataset(tmp_path, {7: [("食物", "我好like呢間餐廳")]})
        records, _ = prepare_mce.read_folder(
            root / "Audio" / "7_MCE", root / "Text" / "data_7.csv"
        )
        assert records[0]["id"] == "7_MCE_7_1"
        assert records[0]["speaker"] == "7_MCE"
        assert records[0]["topic"] == "食物"
        assert records[0]["text"] == "我好like呢間餐廳"

    def test_count_mismatch_skips_the_folder_instead_of_misaligning(self, tmp_path):
        root = make_dataset(tmp_path, {1: [("天氣", "a"), ("天氣", "b")]})
        # delete one wav so counts disagree
        (root / "Audio" / "1_MCE" / "1_2.wav").unlink()
        records, problems = prepare_mce.read_folder(
            root / "Audio" / "1_MCE", root / "Text" / "data_1.csv"
        )
        assert records == []
        assert len(problems) == 1
        assert "unsafe" in problems[0]

    def test_bad_header_is_reported(self, tmp_path):
        root = make_dataset(tmp_path, {1: [("天氣", "a")]})
        (root / "Text" / "data_1.csv").write_text("foo,bar\nx,y\n", encoding="utf-8")
        records, problems = prepare_mce.read_folder(
            root / "Audio" / "1_MCE", root / "Text" / "data_1.csv"
        )
        assert records == []
        assert "Topic/Instance" in problems[0]


class TestDiscover:
    def test_folders_are_sorted_numerically_not_lexicographically(self, tmp_path):
        rows = [("天氣", "a")]
        root = make_dataset(tmp_path, {i: rows for i in (1, 2, 10, 11, 100)})
        found, problems = prepare_mce.discover(root)
        assert [i for i, _, _ in found] == [1, 2, 10, 11, 100]
        assert problems == []

    def test_audio_without_csv_is_reported(self, tmp_path):
        root = make_dataset(tmp_path, {1: [("天氣", "a")]})
        (root / "Audio" / "2_MCE").mkdir()
        found, problems = prepare_mce.discover(root)
        assert [i for i, _, _ in found] == [1]
        assert any("no matching Text/data_2.csv" in p for p in problems)

    def test_csv_without_audio_is_reported(self, tmp_path):
        root = make_dataset(tmp_path, {1: [("天氣", "a")]})
        (root / "Text" / "data_5.csv").write_text("Topic,Instance\n", encoding="utf-8")
        _, problems = prepare_mce.discover(root)
        assert any("no matching Audio/5_MCE" in p for p in problems)

    def test_missing_tree_is_a_hard_error(self, tmp_path):
        (tmp_path / "Audio").mkdir()
        with pytest.raises(SystemExit):
            prepare_mce.discover(tmp_path)


class TestEmitPath:
    def test_absolute_posix_by_default(self, tmp_path):
        root = make_dataset(tmp_path, {1: [("天氣", "a")]})
        wav = root / "Audio" / "1_MCE" / "1_1.wav"
        out = prepare_mce.emit_path(wav, root, None)
        assert out.endswith("Audio/1_MCE/1_1.wav")
        assert "\\" not in out

    def test_prefix_rebases_onto_the_server_root(self, tmp_path):
        root = make_dataset(tmp_path, {1: [("天氣", "a")]})
        wav = root / "Audio" / "1_MCE" / "1_1.wav"
        out = prepare_mce.emit_path(wav, root, "/data/MCE_Dataset")
        assert out == "/data/MCE_Dataset/Audio/1_MCE/1_1.wav"

    def test_trailing_slash_in_prefix_is_tolerated(self, tmp_path):
        root = make_dataset(tmp_path, {1: [("天氣", "a")]})
        wav = root / "Audio" / "1_MCE" / "1_1.wav"
        assert prepare_mce.emit_path(wav, root, "/data/") == "/data/Audio/1_MCE/1_1.wav"


class TestSplit:
    def _run(self, tmp_path, n_folders, out, extra=()):
        rows = [("天氣", "我好busy")] * 3
        root = make_dataset(tmp_path, {i: rows for i in range(1, n_folders + 1)})
        rc = prepare_mce.main([str(root), "--out", str(out), *extra])
        return rc, root

    def test_folder_level_split_at_the_requested_boundary(self, tmp_path, capsys):
        out = tmp_path / "out"
        rc, _ = self._run(tmp_path, 10, out, extra=["--train-folders", "7"])
        assert rc == 0
        import json

        meta = json.loads((out / "split.json").read_text(encoding="utf-8"))
        assert meta["train_folders"] == [1, 2, 3, 4, 5, 6, 7]
        assert meta["test_folders"] == [8, 9, 10]
        assert meta["n_train_utts"] == 21
        assert meta["n_test_utts"] == 9

    def test_no_speaker_appears_in_both_splits(self, tmp_path):
        out = tmp_path / "out"
        self._run(tmp_path, 10, out, extra=["--train-folders", "7"])
        from mce.data import read_jsonl

        train = {r["speaker"] for r in read_jsonl(out / "train.jsonl")}
        test = {r["speaker"] for r in read_jsonl(out / "test.jsonl")}
        assert train & test == set()

    def test_falls_back_to_a_ratio_when_too_few_folders(self, tmp_path, capsys):
        out = tmp_path / "out"
        rc, _ = self._run(tmp_path, 10, out, extra=["--train-folders", "112"])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "Falling back" in captured
        import json

        meta = json.loads((out / "split.json").read_text(encoding="utf-8"))
        assert len(meta["train_folders"]) == 7   # 70% of 10

    def test_single_folder_warns_that_test_will_be_empty(self, tmp_path, capsys):
        out = tmp_path / "out"
        rc, _ = self._run(tmp_path, 1, out)
        assert rc == 0
        captured = capsys.readouterr().out
        assert "test manifest will be EMPTY" in captured
        assert (out / "test.jsonl").read_text(encoding="utf-8") == ""

    def test_strict_mode_fails_when_a_folder_was_skipped(self, tmp_path):
        out = tmp_path / "out"
        rows = [("天氣", "a"), ("天氣", "b")]
        root = make_dataset(tmp_path, {1: rows, 2: rows})
        (root / "Audio" / "2_MCE" / "2_2.wav").unlink()
        rc = prepare_mce.main(
            [str(root), "--out", str(out), "--train-folders", "1", "--strict"]
        )
        assert rc == 1
