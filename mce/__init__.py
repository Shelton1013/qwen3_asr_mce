"""mce -- Mixed / Code-switching Evaluation toolkit for Cantonese-English ASR.

The package is deliberately split so that scoring never depends on inference:

    mce.normalize   text normalisation pipeline (script, case, punctuation, tags)
    mce.tokenizer   code-switching aware tokenisation (zh per character, en per word)
    mce.align       Levenshtein alignment producing per-token edit operations
    mce.metrics     MER / CER_zh / WER_en / PIER / omission / I-D-S breakdown
    mce.data        manifest + hypothesis IO
    mce.models      thin runners for Qwen3-ASR, Whisper, SenseVoice
    mce.report      console + markdown rendering

You can score hypotheses produced by any other tool as long as they are written
into the hypothesis JSONL format described in the README.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
