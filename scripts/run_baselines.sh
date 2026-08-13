#!/usr/bin/env bash
# Stage-0 baseline sweep on a Cantonese-English code-switching test set.
#
# Usage:
#   scripts/run_baselines.sh /path/to/manifest.jsonl exp/switchlingua_yue_en
#
# The two Whisper rows are the important pair: identical audio, identical
# scoring, the only difference being the forced language token. Forcing 'yue'
# is known to send Whisper into repetition loops, which is the most likely
# reason a previously reported WER exceeded 1.0. Report both and let the
# decomposition show which one is a model result and which one is an artefact.

set -euo pipefail

MANIFEST="${1:?usage: run_baselines.sh <manifest.jsonl> <outdir>}"
OUTDIR="${2:?usage: run_baselines.sh <manifest.jsonl> <outdir>}"

mkdir -p "$OUTDIR"

# Applied identically to every row below. Change it here, not per model.
NORM_ARGS=(--script t2s --poi-window 1)

run () {
  local name="$1"; shift
  echo
  echo "################ $name ################"
  python -m mce.cli run \
    --manifest "$MANIFEST" \
    --name "$name" \
    --hyp "$OUTDIR/$name.hyp.jsonl" \
    --out-json "$OUTDIR/$name.metrics.json" \
    --out-utts "$OUTDIR/$name.utts.jsonl" \
    "${NORM_ARGS[@]}" \
    "$@"
}

# --- Qwen3-ASR: no language pin, so the model is free to switch mid-utterance.
run qwen3-asr-1.7b --model qwen3-asr-1.7b --batch-size 8
run qwen3-asr-0.6b --model qwen3-asr-0.6b --batch-size 8

# --- Whisper: the zh/yue ablation.
run whisper-large-v3-zh  --model whisper-large-v3 --language zh  --batch-size 8
run whisper-large-v3-yue --model whisper-large-v3 --language yue --batch-size 8
run whisper-large-v3-turbo-zh --model whisper-large-v3-turbo --language zh --batch-size 8

# --- SenseVoice: a Cantonese-adapted baseline, fairer than stock Whisper.
# Point --model-id at SenseVoice-Small-Yue (WenetSpeech-Yue) once you have the
# checkpoint; the stock SenseVoiceSmall below is the fallback.
run sensevoice-small --model sensevoice-small --language yue --batch-size 8

echo
echo "################ comparison ################"
python -m mce.cli compare "$OUTDIR"/*.metrics.json | tee "$OUTDIR/comparison.md"
