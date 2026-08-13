import json

import pytest

from mce.data import join_hyps, load_manifest, read_jsonl, write_jsonl


@pytest.fixture
def manifest_file(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_jsonl(
        path,
        [
            {"id": "a", "audio": "/x/a.wav", "text": "我好busy"},
            {"id": "b", "audio": "/x/b.wav", "text": "開meeting"},
        ],
    )
    return path


def test_load_jsonl_manifest(manifest_file):
    records = load_manifest(manifest_file)
    assert [r["id"] for r in records] == ["a", "b"]
    assert records[0]["audio"] == "/x/a.wav"


def test_load_tsv_manifest(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("id\twav\ttrans\nu1\t/x/1.wav\t我好busy\n", encoding="utf-8")
    records = load_manifest(path, audio_key="wav", text_key="trans")
    assert records[0]["text"] == "我好busy"


def test_missing_audio_column_names_the_available_fields(tmp_path):
    path = tmp_path / "m.jsonl"
    write_jsonl(path, [{"id": "a", "wav": "/x/a.wav"}])
    with pytest.raises(KeyError, match="audio"):
        load_manifest(path)


def test_ids_default_to_row_index(tmp_path):
    path = tmp_path / "m.jsonl"
    write_jsonl(path, [{"audio": "/x/a.wav", "text": "x"}])
    assert load_manifest(path)[0]["id"] == "utt_000000"


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "m.jsonl"
    write_jsonl(
        path,
        [{"id": "a", "audio": "1.wav", "text": "x"}, {"id": "a", "audio": "2.wav", "text": "y"}],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)


def test_unsupported_format(tmp_path):
    path = tmp_path / "m.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_manifest(path)


def test_utf8_bom_is_tolerated(tmp_path):
    # Windows editors and PowerShell's Out-File prepend a BOM; without
    # utf-8-sig this fails on line 1 of every manifest written on Windows.
    path = tmp_path / "m.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf" + json.dumps(
            {"id": "a", "audio": "1.wav", "text": "我好busy"}, ensure_ascii=False
        ).encode("utf-8") + b"\n"
    )
    assert load_manifest(path)[0]["text"] == "我好busy"


def test_invalid_json_reports_the_line_number(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(path)


class TestJoin:
    manifest = [
        {"id": "a", "audio": "1.wav", "text": "我好busy"},
        {"id": "b", "audio": "2.wav", "text": "開meeting"},
    ]

    def test_join_on_id(self):
        pairs = join_hyps(self.manifest, [{"id": "b", "hyp": "開會"}, {"id": "a", "hyp": "我好"}])
        assert pairs == [("a", "我好busy", "我好"), ("b", "開meeting", "開會")]

    def test_missing_hypothesis_is_scored_as_empty_not_dropped(self, capsys):
        pairs = join_hyps(self.manifest, [{"id": "a", "hyp": "我好"}])
        assert len(pairs) == 2
        assert pairs[1] == ("b", "開meeting", "")
        assert "no hypothesis" in capsys.readouterr().out

    def test_text_key_is_accepted_as_an_alias_for_hyp(self):
        pairs = join_hyps(self.manifest[:1], [{"id": "a", "text": "我好"}])
        assert pairs[0][2] == "我好"

    def test_hypothesis_without_id_is_an_error(self):
        with pytest.raises(KeyError):
            join_hyps(self.manifest, [{"hyp": "我好"}])


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "out" / "hyp.jsonl"
    write_jsonl(path, [{"id": "a", "hyp": "我好busy"}])
    assert read_jsonl(path) == [{"id": "a", "hyp": "我好busy"}]
    # non-ASCII must survive verbatim, not as \uXXXX escapes
    assert "我好busy" in path.read_text(encoding="utf-8")
