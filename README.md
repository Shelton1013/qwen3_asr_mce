# qwen3_asr_mce

Evaluation toolkit for **Cantonese–English code-switching ASR**.

Reports MER, per-language rates, switch-point error rate (PIER), and the
language-omission / translation-instead-of-transcription failure modes — plus an
insertion/deletion/substitution decomposition that tells you whether a bad number
is a model result or a scoring bug.

Built to score Qwen3-ASR, Whisper and SenseVoice on the same footing, but the
scorer is model-agnostic: any system that can write a JSONL of hypotheses can be
evaluated with it.

---

## Why not just report WER and CER

Both are misleading on code-switched Cantonese, in opposite directions.

**WER breaks.** Chinese has no spaces, so whitespace tokenisation collapses an
entire Cantonese span into one or two giant "words". The reference token count
implodes, and the error rate routinely exceeds 1.0 on clean read speech — a
number that says more about the tokenizer than the model:

```
ref = "我今日好busy要開好多meeting"

len(ref.split())     ->  1     # the whole utterance is one "word"
len(tokenize(ref))   -> 10     # 8 Chinese characters + busy + meeting
```

**CER hides the thing you care about.** Chinese is typically 70–85% of the
characters in a code-switched utterance. A model that transcribes Cantonese
perfectly and silently drops every English word still scores an excellent CER:

| | MER | CER_zh | WER_en | EN omission |
|---|---|---|---|---|
| ref `我今日好busy要開meeting` → hyp `我今日好要開` | 25.00 | **0.00** | **100.00** | **100%** |

CER says flawless. The model lost all the English. That is the "language
omission" failure mode, and it is the single most common way an audio LLM fails
at code-switching — so it gets its own metric here rather than being averaged
away.

---

## Install

```bash
git clone https://github.com/Shelton1013/qwen3_asr_mce.git
cd qwen3_asr_mce
pip install -e ".[score]"          # scoring only, CPU, no model weights
pip install -e ".[score,hf]"       # + Qwen3-ASR / Whisper inference
pip install -e ".[score,funasr]"   # + SenseVoice
```

Scoring itself has no hard dependencies beyond OpenCC (for Traditional /
Simplified unification). Inference dependencies are opt-in so you can score on a
laptop and transcribe on a server.

---

## Quickstart

**1. Build a manifest** (JSONL, one utterance per line):

```json
{"id": "sl_yue_en_0001", "audio": "/data/switchlingua/yue_en/0001.wav", "text": "我今日好busy要開好多meeting"}
```

TSV/CSV/JSON also work; use `--audio-key` / `--text-key` / `--id-key` if your
column names differ. Or pull straight from the Hub with `--hf-dataset`.

**2. Transcribe and score:**

```bash
python -m mce.cli run \
  --manifest data/yue_en.jsonl \
  --model qwen3-asr-1.7b \
  --hyp exp/qwen17b.hyp.jsonl \
  --out-json exp/qwen17b.metrics.json \
  --out-utts exp/qwen17b.utts.jsonl \
  --script t2s --name qwen3-asr-1.7b
```

**3. Compare models:**

```bash
python -m mce.cli compare exp/*.metrics.json
```

Or run the whole Stage-0 sweep, including the Whisper `zh` vs `yue` ablation:

```bash
scripts/run_baselines.sh data/yue_en.jsonl exp/stage0
```

Transcription and scoring are separate subcommands on purpose. Normalisation
choices change the numbers, and you will want to rescore several ways;
re-running inference each time would make that expensive enough that you'd stop.

---

## What it reports

```
-- primary --
  MER            18.42 %
  CER_zh         11.07 %   (over 4820 zh tokens)
  WER_en         41.63 %   (over 683 en tokens)
  PIER           33.90 %   (over 1204 switch-region tokens)

-- error decomposition (share of reference tokens) --
  substitutions  12.10 %   (665)
  deletions       4.02 %   (221)
  insertions      2.30 %   (126)

-- code-switching behaviour --
  en omission rate     6.31 %   (24/380 utts lost all English)
  zh omission rate     0.00 %   (0/512 utts lost all Chinese)
  en->zh substitution 18.45 %   (126 en tokens replaced by Chinese)
  en lost (del+sub)   29.72 %   (203 tokens)
  switch ratio         0.812    (508 produced / 626 expected)

-- sanity --
  length ratio         0.981    (hyp tokens / ref tokens)
  runaway rate         0.00 %   (0 utts far longer than reference)
  mean CMI (reference) 0.243    (0 = monolingual, 0.5 = balanced mix)
```

followed by a diagnosis section that names the likely cause of anything
anomalous, and the N worst utterances with reference/hypothesis side by side.

### Metric definitions

| Metric | Definition |
|---|---|
| `MER` | `(S+D+I)/N` over the mixed tokenisation: one token per CJK character, one per English word. |
| `CER_zh` | Same formula restricted to Chinese reference tokens (plus insertions of Chinese material). |
| `WER_en` | Same, restricted to English. **The number CER hides.** |
| `PIER` | Point-of-Interest Error Rate: errors inside a ±`--poi-window` region around each Cantonese↔English boundary, over the size of that region. This is what actually measures code-switching ability. Expect it well above MER. |
| `en_omission_rate` | Share of English-bearing utterances whose hypothesis contains no English at all. |
| `en_sub_by_zh_rate` | Share of English reference tokens replaced by Chinese material — translating instead of transcribing. |
| `switch_ratio` | Language switches produced ÷ switches expected. Far below 1.0 means the model is flattening utterances into one language. |
| `runaway_rate` | Share of utterances more than `--runaway-ratio`× the reference length. Direct detector for decoder collapse / repetition loops. |
| `mean_cmi_ref` | Code Mixing Index of the *reference*: 0 = monolingual, 0.5 = balanced. Tells you whether a low error rate came from an easy, barely-mixed test set. |

All corpus rates are ratios of summed counts, never means of per-utterance
rates — otherwise a two-token utterance outweighs a fifty-token one.

Rates with an empty denominator are reported as `n/a`, not `0.00`. "English error
rate on a corpus with no English" is undefined; printing zero would read as a
perfect score.

### Reading the decomposition

| Symptom | Likely cause |
|---|---|
| High insertions + `runaway_rate` > 0 | Decoder collapse / repetition. For Whisper this is the known consequence of forcing `language="yue"`. |
| High substitutions, `CER_zh` > 45% | Script mismatch — Traditional reference vs Simplified hypothesis. Set `--script`. |
| High deletions on English, `en_omission_rate` up | Language omission. |
| `en_sub_by_zh_rate` up | Translation instead of transcription. |
| `PIER` ≫ `MER` | Good at both languages, bad at switching between them — the genuine code-switching gap. |

---

## Known traps this tool is built around

**Whisper and `language="yue"`.** Whisper saw far more `zh` text than `yue` in
training. Forced to `yue` it tries to reconcile spoken Cantonese with a written
form it barely learned, and falls into repetition loops. The runner therefore
defaults to `zh`; `--language yue` is the opt-in ablation. Run both — the gap
between them is a property of Whisper, not of your test set.

**Qwen3-ASR and language pinning.** Leave `--language` unset. Qwen3-ASR handles
intra-sentential switching in a single checkpoint; pinning it to `yue` asserts
the utterance is monolingual Cantonese, which is the opposite of what you are
measuring.

**Traditional vs Simplified.** Qwen3-ASR is trained mostly on Simplified; Hong
Kong references are usually Traditional. Without `--script`, every Chinese
character is a substitution and `CER_zh` lands above 0.5. Which direction you
pick does not matter — only that both sides get the same one.

**Decoder tags.** SenseVoice emits `<|yue|><|NEUTRAL|><|Speech|><|woitn|>`
inline. Left in, they are pure insertions. Stripped by default.

**書面語 vs 口語 is deliberately *not* normalised.** Whisper tends to write 「是」
for an audible 「係」, 「不是」 for 「唔係」, 「的」 for 「嘅」 — semantically right,
character-by-character wrong. This is real model behaviour, not a formatting
artefact, and folding it away would hide the largest single component of
Whisper's apparent Cantonese error. Measure it instead: take 50 utterances from
`--out-utts` and hand-label how many of the substitutions are pure register
conversion. That number is what makes a Whisper-vs-Qwen comparison fair.

---

## Models

| Alias | Checkpoint |
|---|---|
| `qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B-hf` |
| `qwen3-asr-0.6b` | `Qwen/Qwen3-ASR-0.6B-hf` |
| `whisper-large-v3` | `openai/whisper-large-v3` |
| `whisper-large-v3-turbo` | `openai/whisper-large-v3-turbo` |
| `sensevoice-small` | `iic/SenseVoiceSmall` |
| `qwen3-asr` / `whisper` / `sensevoice` | generic families — pair with `--model-id` |

The Cantonese-adapted WenetSpeech-Yue checkpoints (SenseVoice-Small-Yue,
Whisper-Medium-Yue, U2pp-Conformer-Yue) are **not** aliased, because their Hub
repo ids were not verifiable when this was written and a wrong default would
silently benchmark the wrong model. Use the generic family plus an explicit id:

```bash
python -m mce.cli run --model sensevoice --model-id <the-real-repo-id> ...
```

They are the fairer Cantonese baseline — stock Whisper was never adapted on
Cantonese, so beating it proves less than it looks like.

Note that SenseVoice is a baseline, not a backbone candidate: it is an
encoder-only non-autoregressive model, so there is no autoregressive decoder for
preference optimisation (DPO) to target.

---

## Scoring hypotheses from another system

Nothing here requires you to run inference through this repo. Write a JSONL:

```json
{"id": "sl_yue_en_0001", "hyp": "我今日好busy要開好多meeting"}
```

then:

```bash
python -m mce.cli score --manifest data/yue_en.jsonl --hyp yours.jsonl --script t2s
```

Utterances present in the manifest but missing from the hypothesis file are
scored as empty output — a full deletion — rather than dropped. A model that
crashed on an utterance should be charged for it, not quietly excluded from the
denominator.

---

## Preparing the MCE dataset

`scripts/prepare_mce.py` turns the MCE Cantonese–English corpus into train/test
manifests. Expected layout:

```
MCE_Dataset/
  Audio/{N}_MCE/{N}_{i}.wav     one utterance per file
  Text/data_{N}.csv             header "Topic,Instance", one row per utterance
```

```bash
python scripts/prepare_mce.py /data/MCE_Dataset --out data/mce --train-folders 112
```

Writes `train.jsonl`, `test.jsonl` and `split.json` (which folders went where,
plus every problem found).

Three things it does on purpose:

**Splits by folder, not by utterance.** Folders are speakers. An utterance-level
split puts the same voice on both sides, so a fine-tuned model gets scored partly
on speakers it trained on — and that optimism is unrecoverable after the fact.

**Refuses to guess when counts disagree.** Row *i* of the CSV is the transcript
of wav *i*; the pairing is positional. If a folder has 86 wavs and 85 rows, the
folder is skipped with a warning rather than emitting a manifest that looks fine
and scores nonsense.

**Sorts folders numerically.** Lexicographic ordering would put `10_MCE` before
`2_MCE` and silently change which speakers land in the training split.

Encoding is probed automatically. The corpus stores *Traditional* Chinese in
**GBK**, so the probe tries UTF-8 first (strict, fails fast) and GB18030 before
any Big5 variant. Transcripts arrive wrapped in doubled quotes; those are
stripped, since they are not speech and would otherwise be scored as tokens.

Building manifests on Windows for a Linux GPU box:

```bash
python scripts/prepare_mce.py F:/csasr/MCE_Dataset --out data/mce \
    --train-folders 112 --path-prefix /data/MCE_Dataset
```

Then run the Stage-0 baseline sweep on the held-out speakers:

```bash
scripts/run_baselines.sh data/mce/test.jsonl exp/mce_stage0
```

### Skipping the manifest

For a one-off evaluation, read the corpus directly:

```bash
python -m mce.cli run --dataset mce --dataset-root /data/MCE_Dataset \
    --dataset-split test --train-folders 112 \
    --model qwen3-asr-1.7b --hyp exp/qwen17b.hyp.jsonl --script t2s
```

`--dataset-split` takes `train`, `test` (default) or `all`. Both paths call the
same code in `mce.datasets`, so they produce identical ids, order and split
boundary — a score never depends on which flag you used.

Manifests still earn their keep when the split has to be **frozen and shared**:
fine-tuning reads `train.jsonl` hours before evaluation reads `test.jsonl`, and
`split.json` is the auditable record of which speakers went where. Use
`--dataset` when you just want a number now, and `prepare_mce.py` when the split
is a decision you will have to defend later.

---

## Development

```bash
pip install -e ".[score,dev]"
pytest
```

The test suite encodes each failure mode as the smallest example that exhibits
it, including the whitespace-tokenisation regression that motivates the whole
package.

---

## License

Apache-2.0.
