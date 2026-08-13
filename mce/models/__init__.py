"""Model registry.

Aliases exist only for checkpoints whose Hugging Face ids are known to be
correct. The Cantonese-adapted WenetSpeech-Yue checkpoints (SenseVoice-Small-Yue,
Whisper-Medium-Yue, U2pp-Conformer-Yue) are reachable through the generic family
names plus ``--model-id``, because their repo ids were not verifiable at the time
of writing and a wrong default here would silently benchmark the wrong model.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import ASRModel
from .qwen3_asr import Qwen3ASRModel
from .sensevoice import SenseVoiceModel
from .whisper import WhisperModel

#: alias -> factory taking keyword overrides
REGISTRY: Dict[str, Callable[..., ASRModel]] = {
    # Generic families: pair with --model-id to point at any checkpoint,
    # including the Yue-adapted ones.
    "qwen3-asr": lambda **kw: Qwen3ASRModel(**kw),
    "whisper": lambda **kw: WhisperModel(**kw),
    "sensevoice": lambda **kw: SenseVoiceModel(**kw),
    # Verified checkpoints.
    "qwen3-asr-1.7b": lambda **kw: Qwen3ASRModel(
        **{"model_id": "Qwen/Qwen3-ASR-1.7B-hf", **kw}
    ),
    "qwen3-asr-0.6b": lambda **kw: Qwen3ASRModel(
        **{"model_id": "Qwen/Qwen3-ASR-0.6B-hf", **kw}
    ),
    "whisper-large-v3": lambda **kw: WhisperModel(
        **{"model_id": "openai/whisper-large-v3", **kw}
    ),
    "whisper-large-v3-turbo": lambda **kw: WhisperModel(
        **{"model_id": "openai/whisper-large-v3-turbo", **kw}
    ),
    "sensevoice-small": lambda **kw: SenseVoiceModel(
        **{"model_id": "iic/SenseVoiceSmall", **kw}
    ),
}


def build_model(name: str, **kwargs) -> ASRModel:
    """Instantiate a runner by alias.

    ``kwargs`` with a ``None`` value are dropped so that unset CLI flags fall
    back to each runner's own default rather than overwriting it with ``None``
    (which matters for ``language``: ``None`` and "unset" mean different things
    to Whisper and to Qwen3-ASR).
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        )
    clean = {k: v for k, v in kwargs.items() if v is not None}
    return REGISTRY[name](**clean)


__all__ = [
    "ASRModel",
    "Qwen3ASRModel",
    "SenseVoiceModel",
    "WhisperModel",
    "REGISTRY",
    "build_model",
]
