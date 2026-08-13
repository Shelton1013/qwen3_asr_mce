"""SenseVoice runner, including the Cantonese-adapted SenseVoice-Small-Yue.

SenseVoice is a non-autoregressive encoder: very fast, and a fairer Cantonese
baseline than stock Whisper because SenseVoice-Small-Yue was actually adapted on
Cantonese data (WenetSpeech-Yue).

It is a baseline, not a candidate backbone: an encoder-only non-autoregressive
model has no autoregressive decoder to run preference optimisation against, so
the DPO stage of the training plan cannot target it.

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
    #: 'auto' lets SenseVoice pick; 'yue' pins Cantonese.
    language: Optional[str] = "auto"
    use_itn: bool = False
    vad_model: Optional[str] = None

    def _load(self) -> None:
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SenseVoice needs FunASR: `pip install funasr`"
            ) from exc

        kwargs = {"model": self.model_id, "trust_remote_code": False}
        if self.device != "auto":
            kwargs["device"] = self.device
        if self.vad_model:
            kwargs["vad_model"] = self.vad_model
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
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
        results = self.model.generate(
            input=list(audio_paths),
            language=self.language or "auto",
            use_itn=self.use_itn,
            batch_size=max(1, self.batch_size),
        )
        texts = []
        for item in results:
            raw = item.get("text", "") if isinstance(item, dict) else str(item)
            texts.append(self._post(raw).strip())
        while len(texts) < len(audio_paths):
            texts.append("")
        return texts[: len(audio_paths)]
