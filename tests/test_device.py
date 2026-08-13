"""Device placement, and the error hints that shortcut debugging.

`device_map="auto"` shards a model across GPUs. For a 0.6B-1.7B ASR model that
buys nothing and costs correctness: accelerate relocates module inputs and
outputs, but Qwen3-ASR's audio encoder reads its positional embedding buffer
directly inside forward, so the buffer stays on its assigned device and the
forward pass dies with "found at least two devices". Single-device placement is
therefore the default and sharding is opt-in.
"""

from dataclasses import dataclass
from typing import List, Sequence

import pytest

from mce.models.base import ERROR_HINTS, ASRModel, hint_for_error


@dataclass
class Dummy(ASRModel):
    model_id: str = "dummy"

    def _load(self) -> None:
        pass

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        return [""] * len(audio_paths)


class TestDeviceMap:
    def test_default_pins_to_a_single_device(self):
        m = Dummy(device="auto")
        mapping = m.resolve_device_map()
        assert isinstance(mapping, dict)
        assert set(mapping) == {""}
        assert mapping[""] in ("cuda:0", "cpu")

    def test_explicit_device_is_honoured(self):
        assert Dummy(device="cuda:2").resolve_device_map() == {"": "cuda:2"}
        assert Dummy(device="cpu").resolve_device_map() == {"": "cpu"}

    def test_bare_cuda_becomes_cuda_zero(self):
        assert Dummy(device="cuda").resolve_device() == "cuda:0"

    def test_sharding_is_opt_in_only(self):
        assert Dummy(device="shard").resolve_device_map() == "auto"

    def test_auto_never_returns_the_sharding_sentinel(self):
        # The regression: "auto" used to mean device_map="auto".
        assert Dummy(device="auto").resolve_device_map() != "auto"

    def test_resolve_device_is_a_plain_string(self):
        d = Dummy(device="cuda:3").resolve_device()
        assert isinstance(d, str) and d == "cuda:3"


class TestErrorHints:
    def test_cross_device_error_suggests_a_single_gpu(self):
        hint = hint_for_error(
            "Expected all tensors to be on the same device, but found at least "
            "two devices, cuda:2 and cuda:1!"
        )
        assert "--device cuda:0" in hint
        assert "does not need sharding" in hint

    def test_oom_suggests_batch_size(self):
        assert "batch-size" in hint_for_error("CUDA out of memory. Tried to allocate...")

    def test_chat_template_error_points_at_the_hf_variant(self):
        hint = hint_for_error(
            "continue_final_message is set but the final message does not appear"
        )
        assert "'-hf'" in hint

    def test_missing_audio_suggests_the_path_prefix(self):
        hint = hint_for_error("[Errno 2] No such file or directory: '/data/1_1.wav'")
        assert "--path-prefix" in hint

    def test_matching_is_case_insensitive(self):
        assert hint_for_error("FOUND AT LEAST TWO DEVICES")

    def test_unrecognised_errors_get_no_hint(self):
        assert hint_for_error("something entirely new went wrong") == ""

    def test_every_hint_is_actionable(self):
        # Each hint should name a flag or a concrete artefact, not just describe.
        for needle, hint in ERROR_HINTS:
            assert any(tok in hint for tok in ("--", "'-hf'", "manifest", "report")), needle


class TestAbortMessageCarriesTheHint:
    @dataclass
    class Broken(Dummy):
        def transcribe_batch(self, audio_paths):
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at "
                "least two devices, cuda:2 and cuda:1!"
            )

    def test_hint_is_appended_to_the_abort(self):
        model = self.Broken(batch_size=1)
        with pytest.raises(RuntimeError) as exc:
            model.run([{"id": f"u{i}", "audio": "x.wav"} for i in range(10)], progress=False)
        message = str(exc.value)
        assert "Likely fix:" in message
        assert "--device cuda:0" in message

    def test_unrecognised_error_aborts_without_a_hint_section(self):
        @dataclass
        class Weird(Dummy):
            def transcribe_batch(self, audio_paths):
                raise RuntimeError("cosmic ray")

        with pytest.raises(RuntimeError) as exc:
            Weird(batch_size=1).run(
                [{"id": f"u{i}", "audio": "x.wav"} for i in range(10)], progress=False
            )
        assert "Likely fix:" not in str(exc.value)
