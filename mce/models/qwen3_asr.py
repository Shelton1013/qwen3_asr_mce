"""Qwen3-ASR runner (0.6B / 1.7B).

Follows the ``-hf`` model card: ``AutoProcessor.apply_transcription_request``
builds the multimodal prompt, ``generate`` decodes, and the processor strips the
chat scaffolding back out with ``return_format="transcription_only"``.

Two options matter for code-switching:

``language``
    Leave it as ``None``.  Qwen3-ASR handles intra-sentential switching in a
    single checkpoint, and pinning it to ``yue`` tells the model the utterance is
    monolingual Cantonese -- which is the opposite of what you want to measure.
    The flag exists so you can run that ablation on purpose, not by accident.

``prompt``
    Context biasing.  Useful later for domain terms; leave empty for a clean
    baseline, since a biased run is not comparable to an unbiased one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .base import ASRModel


@dataclass
class Qwen3ASRModel(ASRModel):
    model_id: str = "Qwen/Qwen3-ASR-1.7B-hf"
    prompt: Optional[str] = None

    def _load(self) -> None:
        import transformers

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        model_cls = self._resolve_model_class(transformers)
        self.model = model_cls.from_pretrained(
            self.model_id,
            device_map=self.resolve_device_map(),
            dtype=self.resolve_torch_dtype(),
        )
        self.model.eval()

    @staticmethod
    def _resolve_model_class(transformers):
        """Pick whichever class this transformers version exposes.

        The card uses ``AutoModelForMultimodalLM``; older/newer releases have
        shipped the concrete ``Qwen3ASRForConditionalGeneration`` instead.
        """
        for name in (
            "AutoModelForMultimodalLM",
            "Qwen3ASRForConditionalGeneration",
            "AutoModelForSpeechSeq2Seq",
        ):
            cls = getattr(transformers, name, None)
            if cls is not None:
                return cls
        raise ImportError(
            "No suitable Qwen3-ASR model class in this transformers version. "
            "Install a release that provides AutoModelForMultimodalLM "
            "(`pip install -U transformers`)."
        )

    def _build_request(self, audio_paths: Sequence[str]):
        """Call ``apply_transcription_request`` with only the kwargs it accepts.

        Signatures have moved between releases; probing once and falling back is
        cheaper than pinning an exact transformers version in a research repo.
        """
        payload = list(audio_paths)
        audio_arg = payload if len(payload) > 1 else payload[0]
        attempts = []
        kwargs = {}
        if self.language:
            kwargs["language"] = self.language
        if self.prompt:
            kwargs["prompt"] = self.prompt
        attempts.append(dict(audio=audio_arg, **kwargs))
        if kwargs:
            attempts.append(dict(audio=audio_arg))
        last_exc: Optional[Exception] = None
        for attempt in attempts:
            try:
                return self.processor.apply_transcription_request(**attempt)
            except TypeError as exc:
                last_exc = exc
        raise RuntimeError(
            f"apply_transcription_request rejected every argument set: {last_exc}"
        )

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        import torch

        self.ensure_loaded()
        inputs = self._build_request(audio_paths)
        inputs = inputs.to(self.model.device, self.model.dtype)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        texts = self.processor.decode(generated, return_format="transcription_only")
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.strip() for t in texts]
        # A single-item request may still decode to one string; pad defensively
        # so the caller's zip against the batch never silently truncates.
        while len(texts) < len(audio_paths):
            texts.append("")
        return texts[: len(audio_paths)]
