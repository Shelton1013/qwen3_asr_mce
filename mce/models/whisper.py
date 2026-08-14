"""Whisper runner, with the Cantonese language-token trap made explicit.

Forcing ``language="yue"`` on Whisper is a documented failure: the model saw far
more ``zh`` text than ``yue`` during training, and when forced to ``yue`` it
tries to reconcile spoken Cantonese with a written form it barely learned. The
result is repetition loops and garbage tokens -- which shows up downstream as an
insertion rate that pushes the error rate past 1.0.

The runner therefore defaults to ``zh``, and ``--language yue`` is the opt-in
ablation. Run both and report both; the gap between them is a property of
Whisper, not of your test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .base import ASRModel, load_audio_16k


@dataclass
class WhisperModel(ASRModel):
    model_id: str = "openai/whisper-large-v3"
    #: ``'zh'`` is the pragmatic default for Cantonese audio and ``'yue'`` the
    #: ablation -- large-v3 does carry a ``yue`` token, but community reports
    #: have it collapsing into repetition loops, since Whisper saw far more
    #: ``zh`` text than ``yue``. Run both; the gap is a property of Whisper.
    #:
    #: ``None`` leaves language detection to the model. That is the only setting
    #: comparable to a model that picks its own language, and the only one under
    #: which language-identification collapse can occur at all: forcing a token
    #: hands the model the answer and suppresses the very failure being studied.
    language: Optional[str] = "zh"
    #: Seconds above which an utterance is chunked. ``None`` disables chunking.
    #:
    #: Disabled by default, and that matters: transformers itself warns that
    #: chunk_length_s is "very experimental with seq2seq models". Utterances here
    #: are seconds long and fit in Whisper's native 30s window, so chunking only
    #: routes them through the long-form path, which changes decoding and needs
    #: timestamps to stitch. Set it only for genuinely long audio.
    chunk_length_s: Optional[float] = None
    #: Guards against repetition loops. Off by default so the baseline is the
    #: model's true behaviour rather than a patched-up version of it.
    no_repeat_ngram_size: int = 0
    #: Long-form only. Passing it on the short path is at best ignored.
    condition_on_prev_tokens: Optional[bool] = None
    #: Restrict output to scripts that can legitimately appear (see mce.mitigation).
    script_constraint: Optional[str] = None
    _extra_generate: dict = field(default_factory=dict, repr=False)

    def _load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        dtype = self.resolve_torch_dtype()
        model = self.load_pretrained(
            AutoModelForSpeechSeq2Seq,
            self.model_id,
            hint="Check that this directory is a transformers Whisper checkpoint "
            "and not an OpenAI .pt release or a faster-whisper/CTranslate2 export.",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        # One device for both the model and the pipeline. Passing them separately
        # is how you end up with the weights on cuda:0 and the features on cpu.
        device = self.resolve_device()
        model.to(device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        kwargs = dict(
            model=model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=device,
        )
        if self.chunk_length_s:
            kwargs["chunk_length_s"] = self.chunk_length_s
        # transformers renamed torch_dtype -> dtype; accept either.
        try:
            self.pipe = pipeline("automatic-speech-recognition", dtype=dtype, **kwargs)
        except TypeError:
            self.pipe = pipeline(
                "automatic-speech-recognition", torch_dtype=dtype, **kwargs
            )

        self._logits_processor = None
        if self.script_constraint:
            from ..mitigation import build_script_logits_processor

            self._logits_processor = build_script_logits_processor(
                self.processor.tokenizer, self.script_constraint
            )

    def _generate_kwargs(self) -> dict:
        kwargs = {"task": "transcribe", "do_sample": False}
        if self.language:
            kwargs["language"] = self.language
        if self.no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        # Only meaningful on the long-form path, so only send it when chunking.
        if self.chunk_length_s and self.condition_on_prev_tokens is not None:
            kwargs["condition_on_prev_tokens"] = self.condition_on_prev_tokens
        if self._logits_processor is not None:
            from transformers import LogitsProcessorList

            kwargs["logits_processor"] = LogitsProcessorList([self._logits_processor])
        kwargs.update(self._extra_generate)
        return kwargs

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        """Decode a batch, feeding the pipeline arrays rather than paths.

        Handed a filename, the transformers ASR pipeline shells out to ffmpeg --
        a hard system dependency, and one subprocess per utterance. soundfile is
        already required for everything else here, so reading the audio
        ourselves removes the dependency and the fork.
        """
        self.ensure_loaded()
        inputs = [
            {"raw": load_audio_16k(path), "sampling_rate": 16000}
            for path in audio_paths
        ]
        call = dict(
            batch_size=max(1, self.batch_size),
            generate_kwargs=self._generate_kwargs(),
        )
        # Timestamps are required to stitch chunks and pointless without them.
        if self.chunk_length_s:
            call["return_timestamps"] = True
        outputs = self.pipe(inputs, **call)
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [str(o.get("text", "")).strip() for o in outputs]
