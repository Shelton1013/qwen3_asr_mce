"""Stratified splitting and the split-balance guard.

Motivated by a real failure: the MCE corpus is ordered by collection batch, each
batch has its own topic list, and a positional "first 112 folders" split
therefore put nine topics entirely in train and nine entirely in test -- at a
35% lower code-switching density. The headline number would have measured
cross-topic generalisation while looking like a recognition score.
"""

import pytest

from mce.datasets import (
    check_split_balance,
    load_mce,
    prepare_mce,
    stratified_split,
    topic_signature,
)

BATCH_A = ["天氣", "工作", "體育"]
BATCH_B = ["宠物", "环境", "音乐同艺术"]


def make_batched(tmp_path, n_a=10, n_b=6, utts=6, a_en=3, b_en=1):
    """Two collection batches: disjoint topics and different English density."""
    root = tmp_path / "MCE_Dataset"
    (root / "Audio").mkdir(parents=True)
    (root / "Text").mkdir(parents=True)
    n = 0
    for topics, count, n_en in ((BATCH_A, n_a, a_en), (BATCH_B, n_b, b_en)):
        for _ in range(count):
            n += 1
            adir = root / "Audio" / f"{n}_MCE"
            adir.mkdir()
            rows = ["Topic,Instance"]
            for i in range(1, utts + 1):
                (adir / f"{n}_{i}.wav").write_bytes(b"RIFF")
                english = " ".join(["meeting"] * n_en)
                rows.append(f'{topics[i % len(topics)]},"""我今日好攰要開{english}啊"""')
            (root / "Text" / f"data_{n}.csv").write_bytes(
                ("\n".join(rows) + "\n").encode("gb18030")
            )
    return root


class TestTopicSignature:
    def test_identifies_the_batch(self, tmp_path):
        root = make_batched(tmp_path)
        prepared = prepare_mce(root, train_folders=1)
        every = prepared["train"] + prepared["test"]
        a = [r for r in every if r["speaker"] == "1_MCE"]
        b = [r for r in every if r["speaker"] == "16_MCE"]
        assert topic_signature(a) != topic_signature(b)
        assert topic_signature(a) == frozenset(BATCH_A)


class TestStratifiedSplit:
    def test_each_group_is_split_by_the_ratio(self):
        per_folder = [
            (i, [{"topic": "A"}]) for i in range(1, 11)
        ] + [(i, [{"topic": "B"}]) for i in range(11, 17)]
        train, test = stratified_split(per_folder, train_ratio=0.7)
        assert len(train) == 7 + 4       # 70% of 10, 70% of 6
        assert len(test) == 3 + 2
        # both groups present on both sides
        assert set(train) & set(range(11, 17))
        assert set(test) & set(range(1, 11))

    def test_no_folder_lands_in_both_splits(self):
        per_folder = [(i, [{"topic": "A" if i <= 8 else "B"}]) for i in range(1, 15)]
        train, test = stratified_split(per_folder, 0.7)
        assert set(train) & set(test) == set()
        assert sorted(train + test) == list(range(1, 15))

    def test_singleton_group_goes_to_train_with_a_warning(self):
        per_folder = [(1, [{"topic": "A"}]), (2, [{"topic": "A"}]), (3, [{"topic": "B"}])]
        messages = []
        train, test = stratified_split(per_folder, 0.7, warn=messages.append)
        assert 3 in train
        assert any("single folder" in m for m in messages)

    def test_result_is_deterministic(self):
        per_folder = [(i, [{"topic": "B" if i % 2 else "A"}]) for i in range(1, 21)]
        first = stratified_split(per_folder, 0.7)
        assert first == stratified_split(list(reversed(per_folder)), 0.7)


class TestBalanceGuard:
    def test_positional_split_of_a_batched_corpus_is_flagged(self, tmp_path):
        root = make_batched(tmp_path)
        prepared = prepare_mce(root, train_folders=12)  # cuts into batch B
        warnings = prepared["meta"]["balance_warnings"]
        assert warnings, "the guard must fire on a topic-disjoint split"
        joined = " ".join(warnings)
        assert "absent from test" in joined or "absent from train" in joined
        assert "--stratify topic" in joined

    def test_density_gap_is_reported(self, tmp_path):
        root = make_batched(tmp_path, a_en=4, b_en=0)
        prepared = prepare_mce(root, train_folders=10)
        joined = " ".join(prepared["meta"]["balance_warnings"])
        assert "code-switching density differs" in joined
        assert "easier" in joined

    def test_stratified_split_of_the_same_corpus_is_clean(self, tmp_path):
        root = make_batched(tmp_path)
        prepared = prepare_mce(root, stratify="topic", train_ratio=0.7)
        assert prepared["meta"]["balance_warnings"] == []

    def test_guard_is_silent_on_a_homogeneous_corpus(self, tmp_path):
        root = make_batched(tmp_path, n_a=10, n_b=0)
        prepared = prepare_mce(root, train_folders=7)
        assert prepared["meta"]["balance_warnings"] == []

    def test_guard_handles_an_empty_split(self, tmp_path):
        root = make_batched(tmp_path, n_a=1, n_b=0)
        assert check_split_balance([{"topic": "A", "text": "我"}], []) == []


class TestStratifyPlumbing:
    def test_meta_records_the_mode(self, tmp_path):
        root = make_batched(tmp_path)
        assert prepare_mce(root, stratify="topic")["meta"]["stratify"] == "topic"
        assert prepare_mce(root, train_folders=10)["meta"]["stratify"] == "none"

    def test_invalid_mode_is_rejected(self, tmp_path):
        root = make_batched(tmp_path)
        with pytest.raises(ValueError, match="stratify must be"):
            prepare_mce(root, stratify="speaker")

    def test_load_mce_accepts_stratify(self, tmp_path):
        root = make_batched(tmp_path)
        test = load_mce(root, "test", stratify="topic", train_ratio=0.7)
        topics = {r["topic"] for r in test}
        # both batches represented in the held-out split
        assert topics & set(BATCH_A)
        assert topics & set(BATCH_B)

    def test_load_mce_surfaces_balance_warnings(self, tmp_path):
        root = make_batched(tmp_path)
        messages = []
        load_mce(root, "test", train_folders=12, warn=messages.append)
        assert any(m.startswith("BALANCE:") for m in messages)

    def test_speakers_still_never_cross_splits_when_stratified(self, tmp_path):
        root = make_batched(tmp_path)
        prepared = prepare_mce(root, stratify="topic")
        train = {r["speaker"] for r in prepared["train"]}
        test = {r["speaker"] for r in prepared["test"]}
        assert train & test == set()
