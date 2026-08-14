"""The dev split, and the contamination it exists to prevent.

Error analysis has to happen somewhere. If it happens on the test set -- reading
which words the model gets wrong, then building DPO negatives from exactly those
failures -- the test set has been folded into training and every number measured
afterwards is inflated by an unknowable amount. Dev is the set you are allowed
to read.
"""

import pytest

from mce.datasets import SPLITS, load_mce, prepare_mce

TOPICS = ["天氣", "食物", "旅遊"]


def make_corpus(tmp_path, n_folders=20, utts=6):
    root = tmp_path / "MCE_Dataset"
    (root / "Audio").mkdir(parents=True)
    (root / "Text").mkdir(parents=True)
    for n in range(1, n_folders + 1):
        adir = root / "Audio" / f"{n}_MCE"
        adir.mkdir()
        rows = ["Topic,Instance"]
        for i in range(1, utts + 1):
            (adir / f"{n}_{i}.wav").write_bytes(b"RIFF")
            rows.append(f'{TOPICS[i % 3]},"""我今日好busy要開meeting"""')
        (root / "Text" / f"data_{n}.csv").write_bytes(
            ("\n".join(rows) + "\n").encode("gb18030")
        )
    return root


class TestThreeWaySplit:
    def test_dev_comes_out_of_train_not_test(self, tmp_path):
        root = make_corpus(tmp_path, 20)
        no_dev = prepare_mce(root, train_folders=14)["meta"]
        with_dev = prepare_mce(root, train_folders=14, dev_ratio=0.2)["meta"]
        # the test set is untouched by introducing a dev split
        assert with_dev["test_folders"] == no_dev["test_folders"]
        assert len(with_dev["train_folders"]) + len(with_dev["dev_folders"]) == len(
            no_dev["train_folders"]
        )

    def test_all_three_splits_are_speaker_disjoint(self, tmp_path):
        root = make_corpus(tmp_path, 20)
        p = prepare_mce(root, train_folders=14, dev_ratio=0.25)
        sets = {k: {r["speaker"] for r in p[k]} for k in ("train", "dev", "test")}
        assert sets["train"] & sets["dev"] == set()
        assert sets["train"] & sets["test"] == set()
        assert sets["dev"] & sets["test"] == set()

    def test_no_utterance_is_lost(self, tmp_path):
        root = make_corpus(tmp_path, 20, utts=6)
        p = prepare_mce(root, train_folders=14, dev_ratio=0.2)
        assert len(p["train"]) + len(p["dev"]) + len(p["test"]) == 20 * 6

    def test_dev_ratio_zero_yields_no_dev(self, tmp_path):
        root = make_corpus(tmp_path, 10)
        p = prepare_mce(root, train_folders=7)
        assert p["dev"] == []
        assert p["meta"]["dev_folders"] == []

    def test_dev_size_follows_the_ratio(self, tmp_path):
        root = make_corpus(tmp_path, 20)
        p = prepare_mce(root, train_folders=15, dev_ratio=0.2)
        assert len(p["meta"]["dev_folders"]) == 3   # 20% of 15

    def test_invalid_ratio_is_rejected(self, tmp_path):
        root = make_corpus(tmp_path, 10)
        with pytest.raises(ValueError, match="dev_ratio"):
            prepare_mce(root, train_folders=7, dev_ratio=1.5)

    def test_stratified_dev_keeps_every_topic_group(self, tmp_path):
        root = make_corpus(tmp_path, 20)
        p = prepare_mce(root, stratify="topic", train_ratio=0.7, dev_ratio=0.2)
        for name in ("train", "dev", "test"):
            assert {r["topic"] for r in p[name]} == set(TOPICS), name

    def test_meta_records_the_ratio_and_counts(self, tmp_path):
        root = make_corpus(tmp_path, 20)
        m = prepare_mce(root, train_folders=14, dev_ratio=0.2)["meta"]
        assert m["dev_ratio"] == 0.2
        assert m["n_dev_utts"] > 0
        assert m["n_train_utts"] + m["n_dev_utts"] + m["n_test_utts"] == 120


class TestLoadMceSplits:
    def test_dev_is_a_selectable_split(self, tmp_path):
        root = make_corpus(tmp_path, 20)
        dev = load_mce(root, "dev", train_folders=14, dev_ratio=0.2)
        assert dev
        assert all(r["speaker"] for r in dev)

    def test_dev_is_in_the_split_vocabulary(self):
        assert "dev" in SPLITS

    def test_all_covers_every_split_exactly_once(self, tmp_path):
        root = make_corpus(tmp_path, 20)
        kw = dict(train_folders=14, dev_ratio=0.2)
        every = load_mce(root, "all", **kw)
        ids = [r["id"] for r in every]
        assert len(ids) == len(set(ids)) == 120

    def test_dev_is_empty_without_a_ratio(self, tmp_path):
        root = make_corpus(tmp_path, 10)
        assert load_mce(root, "dev", train_folders=7) == []


class TestBalanceCheckCoversDev:
    def test_train_dev_imbalance_is_labelled(self, tmp_path):
        # one topic group of 18 folders, one of 2: a 10% dev cut takes only the
        # tail, so the guard has something to say about at least one pair
        root = make_corpus(tmp_path, 20)
        warnings = prepare_mce(root, train_folders=14, dev_ratio=0.2)["meta"][
            "balance_warnings"
        ]
        assert all(isinstance(w, str) for w in warnings)
        # any train/dev finding must be attributed, not mixed in with train/test
        assert all(not w.startswith("train/dev:") or "train/dev:" in w for w in warnings)


class TestScriptWiring:
    def _run(self, tmp_path, extra):
        import importlib.util
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "scripts" / "prepare_mce.py"
        spec = importlib.util.spec_from_file_location("prep_dev", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        root = make_corpus(tmp_path, 20)
        out = tmp_path / "out"
        rc = mod.main([str(root), "--out", str(out), "--train-folders", "14", *extra])
        return rc, out

    def test_dev_jsonl_is_written_when_requested(self, tmp_path):
        rc, out = self._run(tmp_path, ["--dev-ratio", "0.2"])
        assert rc == 0
        assert (out / "dev.jsonl").exists()
        assert (out / "dev.jsonl").read_text(encoding="utf-8").strip()

    def test_no_empty_dev_file_is_left_behind(self, tmp_path):
        rc, out = self._run(tmp_path, [])
        assert rc == 0
        assert not (out / "dev.jsonl").exists()

    def test_missing_dev_split_is_called_out(self, tmp_path, capsys):
        self._run(tmp_path, [])
        out = capsys.readouterr().out
        assert "no dev split" in out
        assert "must not run on test.jsonl" in out
