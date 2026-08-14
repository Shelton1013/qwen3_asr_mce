"""Direct corpus reading, and its equivalence with the manifest path.

The point of `--dataset` is convenience, not a second implementation. If the
two paths could disagree about the split boundary or the ids, a score would
depend on which flag you happened to use -- so equivalence is the main thing
tested here.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from mce.data import read_jsonl
from mce.datasets import DATASETS, load_dataset, load_mce, prepare_mce

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepare_mce.py"
_spec = importlib.util.spec_from_file_location("prepare_mce_equiv", _SCRIPT)
prepare_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_script)


def make_dataset(tmp_path, n_folders, n_utts=4):
    root = tmp_path / "MCE_Dataset"
    (root / "Audio").mkdir(parents=True)
    (root / "Text").mkdir(parents=True)
    topics = ["天氣", "食物", "旅遊"]
    for n in range(1, n_folders + 1):
        adir = root / "Audio" / f"{n}_MCE"
        adir.mkdir()
        rows = ["Topic,Instance"]
        for i in range(1, n_utts + 1):
            (adir / f"{n}_{i}.wav").write_bytes(b"RIFF")
            rows.append(
                f'{topics[i % len(topics)]},"""我好busy要開meeting第{n}_{i}條"""'
            )
        (root / "Text" / f"data_{n}.csv").write_bytes(
            ("\n".join(rows) + "\n").encode("gb18030")
        )
    return root


class TestRegistry:
    def test_mce_is_registered(self):
        assert "mce" in DATASETS

    def test_unknown_dataset_lists_the_options(self, tmp_path):
        with pytest.raises(KeyError, match="Available"):
            load_dataset("nope", tmp_path)

    def test_invalid_split_is_rejected(self, tmp_path):
        root = make_dataset(tmp_path, 2)
        with pytest.raises(ValueError, match="split must be"):
            load_mce(root, split="validation")


class TestSplitSelection:
    def test_train_and_test_partition_the_corpus(self, tmp_path):
        root = make_dataset(tmp_path, 10, n_utts=3)
        train = load_mce(root, "train", train_folders=7)
        test = load_mce(root, "test", train_folders=7)
        assert len(train) == 21
        assert len(test) == 9
        assert {r["speaker"] for r in train} & {r["speaker"] for r in test} == set()

    def test_all_split_is_train_then_test(self, tmp_path):
        root = make_dataset(tmp_path, 5, n_utts=2)
        every = load_mce(root, "all", train_folders=3)
        assert [r["id"] for r in every] == [
            r["id"] for r in load_mce(root, "train", train_folders=3)
        ] + [r["id"] for r in load_mce(root, "test", train_folders=3)]

    def test_default_split_is_test(self, tmp_path):
        root = make_dataset(tmp_path, 10, n_utts=2)
        assert load_mce(root, train_folders=7) == load_mce(root, "test", train_folders=7)

    def test_boundary_is_numeric_not_lexicographic(self, tmp_path):
        root = make_dataset(tmp_path, 12, n_utts=1)
        test = load_mce(root, "test", train_folders=9)
        # folders 10, 11, 12 -- not 1, 11, 12 as a string sort would give
        assert sorted(int(r["speaker"].split("_")[0]) for r in test) == [10, 11, 12]


class TestEquivalenceWithManifests:
    """`--dataset mce` and a prepared manifest must be interchangeable."""

    def test_records_are_identical(self, tmp_path):
        root = make_dataset(tmp_path, 10, n_utts=3)
        out = tmp_path / "out"
        rc = prepare_script.main(
            [str(root), "--out", str(out), "--train-folders", "7"]
        )
        assert rc == 0

        for split in ("train", "test"):
            from_manifest = read_jsonl(out / f"{split}.jsonl")
            direct = load_mce(root, split, train_folders=7)
            assert from_manifest == direct, f"{split} split diverged"

    def test_split_metadata_matches(self, tmp_path):
        root = make_dataset(tmp_path, 10, n_utts=2)
        out = tmp_path / "out"
        prepare_script.main([str(root), "--out", str(out), "--train-folders", "7"])
        on_disk = json.loads((out / "split.json").read_text(encoding="utf-8"))
        in_memory = prepare_mce(root, train_folders=7)["meta"]
        assert on_disk["train_folders"] == in_memory["train_folders"]
        assert on_disk["test_folders"] == in_memory["test_folders"]
        assert on_disk["n_test_utts"] == in_memory["n_test_utts"]

    def test_path_prefix_is_applied_the_same_way(self, tmp_path):
        root = make_dataset(tmp_path, 4, n_utts=1)
        direct = load_mce(root, "all", train_folders=2, path_prefix="/data/MCE_Dataset")
        assert all(r["audio"].startswith("/data/MCE_Dataset/Audio/") for r in direct)


class TestWarnings:
    def test_too_few_folders_warns_through_the_callback(self, tmp_path):
        root = make_dataset(tmp_path, 5, n_utts=1)
        messages = []
        load_mce(root, "test", train_folders=112, warn=messages.append)
        assert any("Falling back" in m for m in messages)

    def test_single_folder_warns_that_test_is_empty(self, tmp_path):
        root = make_dataset(tmp_path, 1, n_utts=2)
        messages = []
        records = load_mce(root, "test", train_folders=112, warn=messages.append)
        assert records == []
        assert any("test split will be EMPTY" in m for m in messages)

    def test_warn_is_optional(self, tmp_path):
        root = make_dataset(tmp_path, 3, n_utts=1)
        assert load_mce(root, "test", train_folders=2)  # no warn= kwarg, no crash


class TestCliWiring:
    def test_dataset_without_root_is_rejected(self, tmp_path):
        from mce.cli import main

        with pytest.raises(SystemExit, match="--dataset-root"):
            main(["score", "--dataset", "mce", "--hyp", str(tmp_path / "h.jsonl")])

    def test_score_reads_the_corpus_directly(self, tmp_path, capsys):
        from mce.cli import main

        root = make_dataset(tmp_path, 10, n_utts=2)
        test = load_mce(root, "test", train_folders=7)
        hyp = tmp_path / "hyp.jsonl"
        with open(hyp, "w", encoding="utf-8") as fh:
            for r in test:
                fh.write(json.dumps({"id": r["id"], "hyp": r["text"]}, ensure_ascii=False) + "\n")

        rc = main(
            [
                "score",
                "--dataset", "mce",
                "--dataset-root", str(root),
                "--dataset-split", "test",
                "--train-folders", "7",
                "--hyp", str(hyp),
                "--script", "none",
                "--worst", "0",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "6 utterances" in out          # 3 folders x 2 utts
        assert "MER            0.00 %" in out  # hypotheses are the references

    def test_empty_split_is_a_clean_error_not_a_zero_denominator(self, tmp_path):
        from mce.cli import main

        root = make_dataset(tmp_path, 1, n_utts=2)
        with pytest.raises(SystemExit, match="empty"):
            main(
                [
                    "score",
                    "--dataset", "mce",
                    "--dataset-root", str(root),
                    "--hyp", str(tmp_path / "h.jsonl"),
                ]
            )
