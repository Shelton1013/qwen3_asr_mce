"""Model registry.

Aliases exist only for checkpoints whose Hugging Face ids are known to be
correct. The Cantonese-adapted WenetSpeech-Yue checkpoints (SenseVoice-Small-Yue,
Whisper-Medium-Yue, U2pp-Conformer-Yue) are reachable through the generic family
names plus ``--model-id``, because their repo ids were not verifiable at the time
of writing and a wrong default here would silently benchmark the wrong model.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

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


# Matched against the local config.json first, because that is evidence rather
# than a guess: a directory can be named anything.
_CONFIG_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("qwen3-asr", ("qwen3_asr", "qwen3asr")),
    ("whisper", ("whisper",)),
    ("sensevoice", ("sensevoice",)),
)

# Fallback for checkpoints with no readable config (FunASR ships config.yaml,
# and a bare HF repo id is not on disk at all).
_NAME_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("qwen3-asr", re.compile(r"qwen3[-_ ]?asr", re.I)),
    ("whisper", re.compile(r"whisper", re.I)),
    ("sensevoice", re.compile(r"sense[-_ ]?voice", re.I)),
)


class UnknownModelError(KeyError):
    """A KeyError whose ``str()`` is the message rather than its repr.

    Plain ``KeyError`` reprs its argument, which turns a multi-line hint
    containing Windows paths into unreadable escape soup at the CLI. Subclassing
    keeps ``except KeyError`` working for anyone catching it.
    """

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0] if self.args else ""


def looks_like_checkpoint(name: str) -> bool:
    """Is this a path or a Hub repo id rather than a registry alias?"""
    return "/" in name or "\\" in name or os.path.isdir(name)


def infer_family(checkpoint: str) -> Tuple[Optional[str], str]:
    """Guess which runner loads ``checkpoint``. Returns ``(family, evidence)``."""
    config = Path(checkpoint) / "config.json"
    if config.is_file():
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        blob = " ".join(
            [str(data.get("model_type", ""))] + list(data.get("architectures") or [])
        ).lower()
        for family, hints in _CONFIG_HINTS:
            if any(h in blob for h in hints):
                return family, f"config.json (model_type/architectures: {blob.strip()})"

    for family, pattern in _NAME_PATTERNS:
        if pattern.search(checkpoint):
            return family, "checkpoint name"
    return None, ""


#: The generic entries -- the ones that take an arbitrary --model-id. Named
#: separately so error messages suggest these three rather than every alias.
FAMILIES = ("qwen3-asr", "whisper", "sensevoice")


def build_model(name: str, **kwargs) -> ASRModel:
    """Instantiate a runner by alias, or straight from a checkpoint path.

    ``--model /models/qwen3-asr-1.7b`` is what people actually type, so a path
    or Hub repo id is accepted directly: the family is read from the
    checkpoint's ``config.json`` when there is one, and inferred from the name
    otherwise. Pass ``--model <family> --model-id <path>`` to be explicit.

    ``kwargs`` with a ``None`` value are dropped so that unset CLI flags fall
    back to each runner's own default rather than overwriting it with ``None``
    (which matters for ``language``: ``None`` and "unset" mean different things
    to Whisper and to Qwen3-ASR).
    """
    clean = {k: v for k, v in kwargs.items() if v is not None}

    if name in REGISTRY:
        return REGISTRY[name](**clean)

    if looks_like_checkpoint(name):
        family, evidence = infer_family(name)
        if family:
            print(f"[info] resolved {name} -> {family} runner, via {evidence}")
            clean.setdefault("model_id", name)
            return REGISTRY[family](**clean)
        raise UnknownModelError(
            f"{name!r} looks like a checkpoint but its family could not be "
            f"determined from its config.json or its name. Name it explicitly:\n"
            f"    --model <{'|'.join(FAMILIES)}> --model-id {name}"
        )

    raise UnknownModelError(
        f"Unknown model {name!r}. Available aliases: {', '.join(sorted(REGISTRY))}.\n"
        f"For a local checkpoint pass the path directly (--model /path/to/ckpt) "
        f"or name the family (--model qwen3-asr --model-id /path/to/ckpt)."
    )


__all__ = [
    "ASRModel",
    "Qwen3ASRModel",
    "SenseVoiceModel",
    "WhisperModel",
    "REGISTRY",
    "build_model",
    "infer_family",
    "looks_like_checkpoint",
]
