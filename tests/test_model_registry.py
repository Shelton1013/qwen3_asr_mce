"""Model resolution: aliases, local checkpoints, and the errors in between.

`--model /models/qwen3-asr-1.7b` is what people type when the weights are already
on the server, so it has to work. Family detection prefers the checkpoint's own
config.json over its directory name, because a directory can be called anything.
"""

import json

import pytest

from mce.models import (
    REGISTRY,
    Qwen3ASRModel,
    SenseVoiceModel,
    WhisperModel,
    build_model,
    infer_family,
    looks_like_checkpoint,
)


def make_ckpt(tmp_path, name, config=None):
    d = tmp_path / name
    d.mkdir(parents=True)
    if config is not None:
        (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return d


class TestLooksLikeCheckpoint:
    def test_paths_and_repo_ids(self):
        assert looks_like_checkpoint("/models/qwen3-asr-1.7b")
        assert looks_like_checkpoint("Qwen/Qwen3-ASR-1.7B-hf")
        assert looks_like_checkpoint(r"C:\models\whisper")

    def test_plain_aliases_are_not_checkpoints(self):
        assert not looks_like_checkpoint("qwen3-asr-1.7b")
        assert not looks_like_checkpoint("whisper")


class TestInferFamily:
    def test_config_json_wins_over_the_directory_name(self, tmp_path):
        # deliberately misleading directory name
        d = make_ckpt(tmp_path, "whisper-ish-folder", {"model_type": "qwen3_asr"})
        family, evidence = infer_family(str(d))
        assert family == "qwen3-asr"
        assert "config.json" in evidence

    def test_architectures_are_read_too(self, tmp_path):
        d = make_ckpt(tmp_path, "ckpt", {"architectures": ["WhisperForConditionalGeneration"]})
        assert infer_family(str(d))[0] == "whisper"

    def test_falls_back_to_the_name_without_a_config(self, tmp_path):
        d = make_ckpt(tmp_path, "qwen3-asr-1.7b")
        family, evidence = infer_family(str(d))
        assert family == "qwen3-asr"
        assert evidence == "checkpoint name"

    def test_hub_repo_ids_resolve_by_name(self):
        assert infer_family("Qwen/Qwen3-ASR-0.6B-hf")[0] == "qwen3-asr"
        assert infer_family("openai/whisper-large-v3")[0] == "whisper"
        assert infer_family("iic/SenseVoiceSmall")[0] == "sensevoice"

    def test_underscores_and_spacing_variants(self):
        assert infer_family("/m/qwen3_asr_1.7b")[0] == "qwen3-asr"
        assert infer_family("/m/sense_voice_small_yue")[0] == "sensevoice"

    def test_unrecognised_checkpoint_yields_nothing(self, tmp_path):
        d = make_ckpt(tmp_path, "my-secret-model")
        assert infer_family(str(d))[0] is None

    def test_malformed_config_does_not_crash(self, tmp_path):
        d = make_ckpt(tmp_path, "qwen3-asr-x")
        (d / "config.json").write_text("{not json", encoding="utf-8")
        assert infer_family(str(d))[0] == "qwen3-asr"   # falls through to the name


class TestBuildModel:
    def test_aliases_still_work(self):
        m = build_model("qwen3-asr-1.7b")
        assert isinstance(m, Qwen3ASRModel)
        assert m.model_id == "Qwen/Qwen3-ASR-1.7B-hf"

    def test_local_path_is_accepted_directly(self, tmp_path, capsys):
        d = make_ckpt(tmp_path, "qwen3-asr-1.7b", {"model_type": "qwen3_asr"})
        m = build_model(str(d))
        assert isinstance(m, Qwen3ASRModel)
        assert m.model_id == str(d)
        assert "resolved" in capsys.readouterr().out

    def test_whisper_path_gets_the_whisper_runner_and_its_defaults(self, tmp_path):
        d = make_ckpt(tmp_path, "whisper-large-v3-local")
        m = build_model(str(d))
        assert isinstance(m, WhisperModel)
        # 'yue', which measurement chose over the folklore default of 'zh':
        # on MCE it halves MER and cuts English omission from 22% to 4%.
        assert m.language == "yue"

    def test_sensevoice_path(self, tmp_path):
        d = make_ckpt(tmp_path, "SenseVoice-Small-Yue")
        assert isinstance(build_model(str(d)), SenseVoiceModel)

    def test_explicit_model_id_overrides_the_path(self, tmp_path):
        d = make_ckpt(tmp_path, "qwen3-asr-1.7b")
        m = build_model(str(d), model_id="Qwen/Qwen3-ASR-0.6B-hf")
        assert m.model_id == "Qwen/Qwen3-ASR-0.6B-hf"

    def test_family_alias_with_model_id_is_still_supported(self, tmp_path):
        m = build_model("qwen3-asr", model_id="/models/local-ckpt")
        assert m.model_id == "/models/local-ckpt"

    def test_none_kwargs_do_not_clobber_runner_defaults(self):
        m = build_model("whisper-large-v3", language=None)
        assert m.language == "yue"

    def test_empty_language_is_distinct_from_unset(self):
        # "" asks for the model's own detection; None means "use the default".
        assert build_model("whisper-large-v3", language="").language == ""
        assert build_model("whisper-large-v3", language=None).language == "yue"

    def test_unknown_alias_error_points_at_model_id(self):
        with pytest.raises(KeyError) as exc:
            build_model("gpt-4o-transcribe")
        message = str(exc.value)
        assert "--model-id" in message
        assert "qwen3-asr" in message

    def test_unresolvable_checkpoint_error_shows_the_explicit_form(self, tmp_path):
        d = make_ckpt(tmp_path, "mystery-ckpt")
        with pytest.raises(KeyError) as exc:
            build_model(str(d))
        message = str(exc.value)
        assert "--model-id" in message
        assert str(d) in message

    def test_every_registry_alias_constructs(self):
        for alias in REGISTRY:
            build_model(alias)
