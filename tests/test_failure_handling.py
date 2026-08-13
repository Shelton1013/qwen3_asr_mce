"""Systematic failures must stop the run, not become a score.

A wrong checkpoint variant loads with every tensor randomly initialised,
transformers carries on, and every decode raises. If the runner swallows that
per batch, the evaluation ends with a full set of empty hypotheses which score
as 100% MER and 100% "language omission" -- indistinguishable from a real, very
bad model. That is the single most dangerous failure mode in this pipeline, so
it is tested from three directions: the runner aborts, the scorer flags it, and
the diagnosis refuses to call it a language problem.
"""

from dataclasses import dataclass
from typing import List, Sequence

import pytest

from mce.metrics import score_corpus
from mce.models.base import ASRModel
from mce.report import diagnose, format_metrics


@dataclass
class FlakyModel(ASRModel):
    """Fails every batch, or only the 1-indexed batches named in ``fail_batches``.

    *Where* the failures fall matters: three failures in the first three batches
    is a configuration problem, while three scattered through a hundred is a
    handful of bad wavs. The abort rule has to tell those apart.
    """

    model_id: str = "dummy"
    fail_all: bool = False
    fail_batches: frozenset = frozenset()
    _seen: int = 0

    def _load(self) -> None:
        pass

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        self._seen += 1
        if self.fail_all or self._seen in self.fail_batches:
            raise ValueError(f"decode failed on batch {self._seen}")
        return ["我好busy"] * len(audio_paths)


def records(n):
    return [{"id": f"u{i}", "audio": f"{i}.wav"} for i in range(n)]


class TestRunnerAbort:
    def test_total_failure_aborts_instead_of_emitting_empties(self):
        model = FlakyModel(fail_all=True)
        with pytest.raises(RuntimeError) as exc:
            model.run(records(20), progress=False)
        message = str(exc.value)
        assert "configuration problem, not a model result" in message
        assert "decode failed" in message

    def test_abort_happens_early_not_after_the_whole_corpus(self):
        model = FlakyModel(fail_all=True, batch_size=1)
        with pytest.raises(RuntimeError) as exc:
            model.run(records(4000), progress=False)
        # min_failures_to_abort=3, so it must not have chewed through 4000
        assert "of the first 3 utterances" in str(exc.value)

    def test_isolated_failure_is_tolerated(self):
        model = FlakyModel(fail_batches=frozenset({1}), batch_size=1)
        out = model.run(records(50), progress=False)
        assert len(out) == 50
        assert out[0]["hyp"] == ""
        assert "error" in out[0]
        assert all("error" not in r for r in out[1:])

    def test_scattered_failures_below_the_threshold_do_not_abort(self):
        # 3 failures, but only after 49 successes -- 6% at the worst moment
        model = FlakyModel(fail_batches=frozenset({50, 51, 52}), batch_size=1)
        out = model.run(records(100), progress=False)
        assert sum(1 for r in out if r.get("error")) == 3
        assert len(out) == 100

    def test_three_failures_at_the_very_start_do_abort(self):
        # Same count as above, different position: nothing has succeeded yet.
        model = FlakyModel(fail_batches=frozenset({1, 2, 3}), batch_size=1)
        with pytest.raises(RuntimeError):
            model.run(records(100), progress=False)

    def test_threshold_is_configurable(self):
        model = FlakyModel(fail_all=True, batch_size=1, max_failure_rate=1.0)
        out = model.run(records(10), progress=False)   # never aborts
        assert all(r["hyp"] == "" for r in out)


class TestCheckpointVerification:
    HINT = "use the -hf conversion"

    def test_many_missing_weights_is_fatal(self):
        info = {"missing_keys": [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(28)]}
        with pytest.raises(RuntimeError) as exc:
            ASRModel.verify_checkpoint_loaded(info, "/models/wrong", hint=self.HINT)
        message = str(exc.value)
        assert "randomly initialised" in message
        assert "/models/wrong" in message
        assert self.HINT in message

    def test_unexpected_keys_are_shown_as_evidence(self):
        info = {
            "missing_keys": [f"model.layers.{i}.mlp.up_proj.weight" for i in range(20)],
            "unexpected_keys": ["thinker.model.layers.0.mlp.up_proj.weight"],
        }
        with pytest.raises(RuntimeError, match="thinker"):
            ASRModel.verify_checkpoint_loaded(info, "/models/wrong")

    def test_a_few_missing_buffers_are_tolerated(self):
        info = {"missing_keys": ["model.embed_positions.weight"]}
        ASRModel.verify_checkpoint_loaded(info, "/models/ok")

    def test_non_weight_keys_do_not_count(self):
        info = {"missing_keys": [f"model.layers.{i}.attn.bias_k" for i in range(40)]}
        ASRModel.verify_checkpoint_loaded(info, "/models/ok")

    def test_absent_loading_info_is_not_an_error(self):
        ASRModel.verify_checkpoint_loaded(None, "/models/ok")
        ASRModel.verify_checkpoint_loaded({}, "/models/ok")

    def test_qwen_runner_carries_the_hf_variant_hint(self):
        from mce.models.qwen3_asr import NON_HF_CHECKPOINT_HINT

        assert "-hf" in NON_HF_CHECKPOINT_HINT
        assert "thinker." in NON_HF_CHECKPOINT_HINT


class TestDiagnosisOfEmptyOutput:
    def test_all_empty_is_called_a_broken_run_not_language_omission(self):
        pairs = [(f"u{i}", "我今日好busy要開meeting", "") for i in range(20)]
        metrics, _ = score_corpus(pairs)
        assert metrics.mer == 1.0
        report = diagnose(metrics)
        assert "broken run" in report
        assert "language omission" not in report

    def test_the_full_report_still_renders(self):
        pairs = [(f"u{i}", "我今日好busy", "") for i in range(5)]
        metrics, _ = score_corpus(pairs)
        text = format_metrics(metrics, title="broken")
        assert "broken run" in text

    def test_mostly_empty_output_is_flagged_before_deletion_analysis(self):
        pairs = [(f"u{i}", "我今日好busy要開meeting啊", "" if i else "我") for i in range(20)]
        metrics, _ = score_corpus(pairs)
        assert 0 < (metrics.length_ratio or 0) < 0.2
        assert "came back empty from a failed decode" in diagnose(metrics)

    def test_genuine_language_omission_is_still_diagnosed(self):
        # non-empty hypotheses that simply drop the English
        pairs = [(f"u{i}", "我今日好busy要開meeting", "我今日好要開") for i in range(20)]
        metrics, _ = score_corpus(pairs)
        report = diagnose(metrics)
        assert "language omission" in report
        assert "broken run" not in report


class TestScorerFlagsFailedTranscriptions:
    def test_score_warns_when_hypotheses_carry_errors(self, tmp_path, capsys):
        import json

        from mce.cli import main

        manifest = tmp_path / "m.jsonl"
        hyp = tmp_path / "h.jsonl"
        with open(manifest, "w", encoding="utf-8") as fh:
            for i in range(4):
                fh.write(json.dumps(
                    {"id": f"u{i}", "audio": "x.wav", "text": "我好busy"},
                    ensure_ascii=False) + "\n")
        with open(hyp, "w", encoding="utf-8") as fh:
            for i in range(4):
                rec = {"id": f"u{i}", "hyp": ""}
                if i < 3:
                    rec["error"] = "ValueError: chat template mismatch"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        rc = main([
            "score", "--manifest", str(manifest), "--hyp", str(hyp),
            "--script", "none", "--worst", "0",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "3/4 hypotheses came from FAILED transcriptions" in out
        assert "every rate below is inflated" in out
        assert "chat template mismatch" in out
