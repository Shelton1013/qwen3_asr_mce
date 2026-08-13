"""Common interface for ASR runners.

Runners are intentionally thin. They own model loading and decoding and nothing
else -- no normalisation, no metric logic -- so that swapping in a fourth model
never changes what a score means.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ASRModel(abc.ABC):
    """Base runner.

    Subclasses implement :meth:`_load` and :meth:`transcribe_batch`. Loading is
    lazy so that constructing a runner (e.g. to print its config) never pulls
    several gigabytes onto the GPU.
    """

    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    language: Optional[str] = None
    batch_size: int = 1
    max_new_tokens: int = 256
    #: Abort once this share of utterances has failed (and at least
    #: ``min_failures_to_abort`` of them). One corrupt wav in a corpus is worth
    #: skipping; a systematic failure must stop the run, because an evaluation
    #: built from empty hypotheses reads exactly like a very bad model.
    max_failure_rate: float = 0.2
    min_failures_to_abort: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)
    _loaded: bool = field(default=False, init=False, repr=False)

    # -- lifecycle --------------------------------------------------------

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    @abc.abstractmethod
    def _load(self) -> None:
        ...

    @abc.abstractmethod
    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        ...

    # -- driving ----------------------------------------------------------

    def transcribe(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path])[0]

    def run(self, records: Sequence[dict], progress: bool = True) -> List[dict]:
        """Transcribe a manifest, returning hypothesis records.

        An isolated failure is recorded as an empty hypothesis with the error
        attached and the run continues -- losing a whole evaluation to one
        corrupt wav is not useful, and an empty hypothesis scored as a full
        deletion is honest accounting.

        A *systematic* failure is different. If most utterances fail, the run
        aborts instead of producing a full set of empty hypotheses, because
        those score as 100% MER with 100% "language omission" and read exactly
        like a catastrophically bad model rather than a broken configuration.
        """
        self.ensure_loaded()
        out: List[dict] = []
        total = len(records)
        started = time.time()
        failures = 0
        last_exc: Optional[BaseException] = None

        for start in range(0, total, self.batch_size):
            chunk = list(records[start : start + self.batch_size])
            paths = [r["audio"] for r in chunk]
            try:
                hyps = self.transcribe_batch(paths)
            except Exception as exc:  # noqa: BLE001 - deliberate: keep going
                failures += len(chunk)
                last_exc = exc
                print(f"[error] batch at {start} failed: {type(exc).__name__}: {exc}")
                for rec in chunk:
                    out.append({"id": rec["id"], "hyp": "", "error": str(exc)})
                self._abort_if_systematic(failures, len(out), last_exc)
                continue
            for rec, hyp in zip(chunk, hyps):
                out.append({"id": rec["id"], "hyp": hyp})
            if progress:
                done = min(start + self.batch_size, total)
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                print(f"\r  {done}/{total} utts  ({rate:.2f} utt/s)", end="", flush=True)
        if progress:
            print()
        if failures:
            print(f"[warn] {failures}/{total} utterances failed and were written as empty")
        return out

    def _abort_if_systematic(
        self, failures: int, processed: int, last_exc: Optional[BaseException]
    ) -> None:
        if failures < self.min_failures_to_abort:
            return
        if processed and failures / processed <= self.max_failure_rate:
            return
        raise RuntimeError(
            f"aborting: {failures} of the first {processed} utterances failed "
            f"({failures / processed:.0%}). This is a configuration problem, not a "
            f"model result -- continuing would emit empty hypotheses that score as "
            f"100% MER and look like a real (terrible) system.\n"
            f"Last error: {type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    # -- loading helpers --------------------------------------------------

    def load_pretrained(self, model_cls, model_id: str, hint: str = "", **kwargs):
        """``from_pretrained`` plus a check that the weights actually loaded.

        ``transformers`` reports a key mismatch and carries on with randomly
        initialised tensors. For an evaluation that is the worst possible
        behaviour: the model runs, produces garbage, and the garbage is scored.
        """
        try:
            model, info = model_cls.from_pretrained(
                model_id, output_loading_info=True, **kwargs
            )
        except TypeError:
            # Some classes do not accept output_loading_info; nothing to check.
            return model_cls.from_pretrained(model_id, **kwargs)
        self.verify_checkpoint_loaded(info, model_id, hint=hint)
        return model

    @staticmethod
    def verify_checkpoint_loaded(
        loading_info: Optional[dict], model_id: str, hint: str = "", threshold: int = 10
    ) -> None:
        """Raise if a meaningful number of weight tensors were newly initialised."""
        if not loading_info:
            return
        missing = [
            k for k in loading_info.get("missing_keys", []) or []
            if k.endswith((".weight", ".bias"))
        ]
        if len(missing) <= threshold:
            return
        unexpected = list(loading_info.get("unexpected_keys", []) or [])
        detail = [
            f"{model_id} did not load: {len(missing)} weight tensors were missing "
            f"from the checkpoint and were randomly initialised. This model would "
            f"produce noise, so the run is stopped rather than scored.",
            f"  missing (first 3):    {', '.join(missing[:3])}",
        ]
        if unexpected:
            detail.append(f"  unexpected (first 3): {', '.join(unexpected[:3])}")
        if hint:
            detail.append(f"  {hint}")
        raise RuntimeError("\n".join(detail))

    # -- helpers ----------------------------------------------------------

    def resolve_torch_dtype(self):
        """Map the ``dtype`` string onto a torch dtype.

        ``auto`` means bfloat16 on CUDA and float32 on CPU: bfloat16 on CPU is
        technically supported but slow enough to look like a hang.
        """
        import torch  # imported here so the package imports without torch

        if self.dtype == "auto":
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return getattr(torch, self.dtype)

    def resolve_device_map(self):
        if self.device == "auto":
            return "auto"
        return self.device


def load_audio_16k(path: str):
    """Load a file as mono 16 kHz float32.

    Only used by runners whose processor will not take a path directly.
    """
    try:
        import soundfile as sf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Audio loading needs `pip install soundfile`") from exc
    import numpy as np

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        try:
            import librosa  # type: ignore

            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"{path} is {sr} Hz and needs resampling; `pip install librosa`"
            ) from exc
    return np.asarray(wav, dtype="float32")
