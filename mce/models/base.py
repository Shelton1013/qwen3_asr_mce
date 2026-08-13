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

        A failure on one utterance is recorded as an empty hypothesis with the
        error attached, and the run continues. Losing a whole evaluation to one
        corrupt wav is not a useful failure mode; an empty hypothesis is scored
        as a full deletion, which is the honest accounting.
        """
        self.ensure_loaded()
        out: List[dict] = []
        total = len(records)
        started = time.time()
        for start in range(0, total, self.batch_size):
            chunk = list(records[start : start + self.batch_size])
            paths = [r["audio"] for r in chunk]
            try:
                hyps = self.transcribe_batch(paths)
            except Exception as exc:  # noqa: BLE001 - deliberate: keep going
                print(f"[error] batch at {start} failed: {type(exc).__name__}: {exc}")
                hyps = [""] * len(chunk)
                for rec, hyp in zip(chunk, hyps):
                    out.append({"id": rec["id"], "hyp": hyp, "error": str(exc)})
                continue
            for rec, hyp in zip(chunk, hyps):
                out.append({"id": rec["id"], "hyp": hyp})
            if progress:
                done = min(start + self.batch_size, total)
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"\r  {done}/{total} utts  ({rate:.2f} utt/s)",
                    end="",
                    flush=True,
                )
        if progress:
            print()
        return out

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
