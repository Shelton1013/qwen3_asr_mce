"""SenseVoice runner, including the Cantonese-adapted SenseVoice-Small-Yue.

SenseVoice is a non-autoregressive encoder: very fast, and a fairer Cantonese
baseline than stock Whisper because SenseVoice-Small-Yue was actually adapted on
Cantonese data (WenetSpeech-Yue).

It is a baseline, not a candidate backbone: an encoder-only non-autoregressive
model has no autoregressive decoder to run preference optimisation against, so
the DPO stage of the training plan cannot target it. It also cannot take a
logits-processor style script constraint for the same reason.

Its raw output carries inline metadata tags -- ``<|yue|><|NEUTRAL|><|Speech|>``
and friends. ``rich_transcription_postprocess`` removes them properly; the
normaliser also strips ``<|...|>`` as a second line of defence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .base import ASRModel


@dataclass
class SenseVoiceModel(ASRModel):
    model_id: str = "iic/SenseVoiceSmall"
    #: 'auto' lets SenseVoice run its own LID -- which is the condition under
    #: which language collapse can happen at all, so it is the interesting
    #: default for this study. Pin 'yue' to measure recognition with LID removed.
    language: Optional[str] = "auto"
    use_itn: bool = False
    vad_model: Optional[str] = None
    #: SenseVoice emits emotion/event tags; banning the unknown ones keeps the
    #: transcript clean when postprocessing is unavailable.
    ban_emo_unk: bool = True

    def _load(self) -> None:
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SenseVoice needs FunASR: `pip install funasr`") from exc

        kwargs = {
            "model": self.model_id,
            # SenseVoiceSmall ships its own model.py; loading it with
            # trust_remote_code=False either fails or silently builds the wrong
            # architecture.
            "trust_remote_code": True,
            "remote_code": "./model.py",
            # FunASR defaults to CPU when device is unset. On a 4k-utterance
            # corpus that is the difference between minutes and hours.
            "device": self.resolve_device(),
            "disable_update": True,
        }
        if self.vad_model:
            kwargs["vad_model"] = self.vad_model
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

        try:
            self.model = AutoModel(**kwargs)
        except TypeError:
            # Older FunASR releases lack remote_code / disable_update.
            for key in ("remote_code", "disable_update"):
                kwargs.pop(key, None)
            self.model = AutoModel(**kwargs)

        try:
            from funasr.utils.postprocess_utils import (  # type: ignore
                rich_transcription_postprocess,
            )

            self._post = rich_transcription_postprocess
        except ImportError:  # pragma: no cover - older FunASR
            self._post = lambda s: s

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        self.ensure_loaded()
        call = {
            "input": list(audio_paths),
            "language": self.language or "auto",
            "use_itn": self.use_itn,
            "batch_size": max(1, self.batch_size),
            "ban_emo_unk": self.ban_emo_unk,
        }
        if self.vad_model:
            # With VAD the batch is measured in seconds of audio, not items.
            call["batch_size_s"] = 300
            call["merge_vad"] = True
        try:
            results = self.model.generate(**call)
        except TypeError:
            for key in ("ban_emo_unk", "merge_vad", "batch_size_s"):
                call.pop(key, None)
            results = self.model.generate(**call)

        texts = []
        for item in results:
            raw = item.get("text", "") if isinstance(item, dict) else str(item)
            texts.append(self._post(raw).strip())
        while len(texts) < len(audio_paths):
            texts.append("")
        return texts[: len(audio_paths)]
