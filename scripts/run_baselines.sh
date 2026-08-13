#!/usr/bin/env bash
# Stage-0 baseline sweep on a Cantonese-English code-switching test set.
#
# Usage:
#   scripts/run_baselines.sh <manifest.jsonl> <outdir>
#
# Checkpoints default to Hub ids. On a server with the weights already on disk
# (or no outbound network), point the variables at local directories:
#
#   MODEL_ROOT=/home/share/models scripts/run_baselines.sh test.jsonl exp/stage0
#
# or override them individually:
#
#   QWEN17B=/models/qwen3-asr-1.7b WHISPER_V3=/models/whisper-large-v3 \
#       scripts/run_baselines.sh test.jsonl exp/stage0
#
# The two Whisper rows are the important pair: identical audio, identical
# scoring, the only difference being the forced language token. Forcing 'yue'
# is known to send Whisper into repetition loops, which is the most likely
# reason a previously reported WER exceeded 1.0. Report both and let the
# decomposition show which one is a model result and which one is an artefact.

set -euo pipefail

MANIFEST="${1:?usage: run_baselines.sh <manifest.jsonl> <outdir>}"
OUTDIR="${2:?usage: run_baselines.sh <manifest.jsonl> <outdir>}"

# If MODEL_ROOT is set, look for each checkpoint under it by directory name,
# falling back to the Hub id when that directory does not exist. This keeps the
# script usable both on a fresh box and on one with a populated model cache.
local_or() {  # local_or <dirname> <hub-id>
  if [[ -n "${MODEL_ROOT:-}" && -d "${MODEL_ROOT}/$1" ]]; then
    echo "${MODEL_ROOT}/$1"
  else
    echo "$2"
  fi
}

QWEN17B="${QWEN17B:-$(local_or qwen3-asr-1.7b   Qwen/Qwen3-ASR-1.7B-hf)}"
QWEN06B="${QWEN06B:-$(local_or qwen3-asr-0.6b   Qwen/Qwen3-ASR-0.6B-hf)}"
WHISPER_V3="${WHISPER_V3:-$(local_or whisper-large-v3       openai/whisper-large-v3)}"
WHISPER_TURBO="${WHISPER_TURBO:-$(local_or whisper-large-v3-turbo openai/whisper-large-v3-turbo)}"
SENSEVOICE="${SENSEVOICE:-$(local_or SenseVoiceSmall        iic/SenseVoiceSmall)}"

BATCH_SIZE="${BATCH_SIZE:-8}"
# One GPU. These models are a few GB in bfloat16; sharding them across devices
# buys nothing and breaks Qwen3-ASR's audio encoder.
DEVICE="${DEVICE:-auto}"

mkdir -p "$OUTDIR"

echo "manifest:   $MANIFEST"
echo "output:     $OUTDIR"
echo "checkpoints:"
printf '  %-24s %s\n' qwen3-asr-1.7b "$QWEN17B" qwen3-asr-0.6b "$QWEN06B" \
  whisper-large-v3 "$WHISPER_V3" whisper-large-v3-turbo "$WHISPER_TURBO" \
  sensevoice "$SENSEVOICE"

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
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    "${NORM_ARGS[@]}" \
    "$@"
}

# --- Qwen3-ASR: no language pin, so the model is free to switch mid-utterance.
run qwen3-asr-1.7b --model qwen3-asr --model-id "$QWEN17B"
run qwen3-asr-0.6b --model qwen3-asr --model-id "$QWEN06B"

# --- Whisper: the zh/yue ablation.
run whisper-large-v3-zh        --model whisper --model-id "$WHISPER_V3"    --language zh
run whisper-large-v3-yue       --model whisper --model-id "$WHISPER_V3"    --language yue
run whisper-large-v3-turbo-zh  --model whisper --model-id "$WHISPER_TURBO" --language zh

# --- SenseVoice: a Cantonese-adapted baseline is fairer than stock Whisper.
# Point SENSEVOICE at SenseVoice-Small-Yue (WenetSpeech-Yue) if you have it.
run sensevoice --model sensevoice --model-id "$SENSEVOICE" --language yue

echo
echo "################ comparison ################"
python -m mce.cli compare "$OUTDIR"/*.metrics.json | tee "$OUTDIR/comparison.md"
