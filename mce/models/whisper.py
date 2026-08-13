"""Whisper runner, with the Cantonese language-token trap made explicit.

Forcing ``language="yue"`` on Whisper is a documented failure: the model saw far
more ``zh`` text than ``yue`` during training, and when forced to ``yue`` it
tries to reconcile spoken Cantonese with a written form it barely learned. The
result is repetition loops and garbage tokens -- which shows up downstream as an
insertion rate that pushes the error rate past 1.0.

The runner therefore defaults to ``zh``, and ``--whisper-language yue`` is the
opt-in ablation. Run both and report both; the gap between them is a property of
Whisper, not of your test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .base import ASRModel


@dataclass
class WhisperModel(ASRModel):
    model_id: str = "openai/whisper-large-v3"
    #: 'zh' is the sane default for Cantonese audio; 'yue' is the ablation.
    language: Optional[str] = "zh"
    chunk_length_s: float = 30.0
    #: Guards against repetition loops. Off by default so the baseline is the
    #: model's true behaviour rather than a patched-up version of it.
    no_repeat_ngram_size: int = 0
    condition_on_prev_tokens: bool = False

    def _load(self) -> None:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        dtype = self.resolve_torch_dtype()
        model = self.load_pretrained(
            AutoModelForSpeechSeq2Seq,
            self.model_id,
            hint="Check that this directory is a transformers Whisper checkpoint "
            "and not an OpenAI .pt release or a faster-whisper/CTranslate2 export.",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        device = 0 if (self.device == "auto" and torch.cuda.is_available()) else self.device
        if device == "auto":
            device = -1
        model.to("cuda" if device == 0 else "cpu")
        processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            dtype=dtype,
            device=device,
        )

    def _generate_kwargs(self) -> dict:
        kwargs = {
            "task": "transcribe",
            "condition_on_prev_tokens": self.condition_on_prev_tokens,
            "do_sample": False,
        }
        if self.language:
            kwargs["language"] = self.language
        if self.no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return kwargs

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        self.ensure_loaded()
        outputs = self.pipe(
            list(audio_paths),
            batch_size=max(1, self.batch_size),
            generate_kwargs=self._generate_kwargs(),
            return_timestamps=False,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [str(o.get("text", "")).strip() for o in outputs]
