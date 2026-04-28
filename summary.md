# ASR Experiment on MINDS-14: Notebook Summary

## Project Overview

**Objective:** Fine-tune and compare Whisper models on the MINDS-14 dataset (en-US) for automatic speech recognition (ASR) transcription.

**Key Challenge:** Mitigate overfitting on a small dataset (~450 training samples) using:
- Frozen encoder (only train the decoder)
- Weight decay + early stopping (patience=3)
- Both WER and CER metrics for checkpoint selection

---

## Workflow

### 1. Dataset Inspection
- **Source:** PolyAI/minds14 (en-US split)
- **Size:** 563 samples in train split
- **Features:** path, audio, transcription, english_transcription, intent_class, lang_id
- **Audio:** 8 kHz sampling rate, WAV format

### 2. Exploratory Data Analysis (EDA)
- **Audio Duration:** Mean ~8.5s, range 1.7s–58.5s
- **Transcript Length:** 1–50+ words, 4–200+ characters
- **Visualizations:** Duration histogram, word/character count distributions, spectrograms, MFCCs
- **Top Tokens:** "i", "to", "my", "a", "account", "i'm", "and", "the"

### 3. Preprocessing
- **Audio:** Resampled from 8 kHz → 16 kHz using librosa
- **Text:** Lowercased, whitespace-normalized
- **Split:** 80/10/10 → Train: 450 / Val: 56 / Test: 57 samples
- **Verification:** No overlap between splits (assertions passed)

### 4. Model Configuration
| Model | Parameters | Frozen Encoder | Trainable Params |
|-------|------------|----------------|------------------|
| whisper-tiny | ~39M | 8,208,384 | ~29.5M |
| whisper-base | ~74M | 20,590,592 | ~52M |

**Training Settings:**
- Learning rate: 1e-5
- Weight decay: 0.01
- Warmup steps: 50
- Batch size: 8 (train), 4 (eval)
- Max epochs: 10
- Early stopping: patience=3
- FP16: enabled (CUDA)

---

## Results

### Whisper-tiny Training Progress
| Epoch | Train Loss | Val Loss | WER | CER |
|-------|-----------|----------|-----|-----|
| 1 | 0.836 | 0.546 | 0.267 | 0.204 |
| 2 | 0.434 | 0.479 | 0.269 | 0.208 |
| 3 | 0.325 | 0.474 | 0.263 | 0.211 |
| 4 | 0.261 | 0.475 | **0.254** | 0.202 |
| 5 | 0.209 | 0.483 | 0.266 | 0.208 |
| 6 | 0.187 | 0.493 | 0.267 | 0.213 |
| 7 | 0.135 | 0.505 | 0.283 | 0.232 |

**Best checkpoint:** Epoch 4

### Whisper-base Training Progress
| Epoch | Train Loss | Val Loss | WER | CER |
|-------|-----------|----------|-----|-----|
| 1 | 0.738 | 0.463 | 0.276 | 0.225 |
| 2 | 0.374 | 0.425 | 0.240 | 0.185 |
| 3 | 0.269 | 0.433 | **0.194** | 0.147 |
| 4 | 0.197 | 0.444 | 0.221 | 0.172 |
| 5 | 0.140 | 0.475 | 0.269 | 0.221 |
| 6 | 0.099 | 0.499 | 0.227 | 0.172 |

**Best checkpoint:** Epoch 3

### Final Test Set Evaluation

| Model | Parameters | Training | WER | CER |
|-------|-----------|----------|-----|-----|
| whisper-tiny | ~39M | zero-shot | 0.5202 | 0.3724 |
| whisper-tiny | ~39M | fine-tuned | 0.3489 | 0.2953 |
| whisper-base | ~74M | zero-shot | 0.4887 | 0.3483 |
| whisper-base | ~74M | fine-tuned | **0.2544** | **0.2111** |

**Fine-tuning gain:** whisper-tiny improves by 32.9% WER / 20.7% CER over zero-shot; whisper-base improves by 47.9% WER / 39.4% CER over zero-shot.

**Fine-tuned base vs fine-tuned tiny:** WER +27.1% relative improvement.

### Detailed Metrics

| Model | Exact Match Rate | Avg Sample WER | Overall WER | Overall CER |
|-------|-----------------|----------------|-------------|-------------|
| whisper-tiny | 26.32% | 0.2805 | 0.3489 | 0.2953 |
| whisper-base | **33.33%** | **0.2154** | **0.2544** | **0.2111** |

whisper-base also leads in exact match rate (+7.01pp) and avg sample WER (-0.065).

### Error Analysis

**Whisper-base Best Predictions (WER = 0.00):**
- "what's my current balance"
- "hi hello i just need to deposit money into my account please"
- "show me my account balance"
- "card not working"
- "transfer money to the account"

**Whisper-base Worst Predictions:**
- Short utterances with misrecognized keywords (e.g., "change my dress" → "can i change my address")
- Long transcripts with semantic errors (hallucinations)
- Technical terms not well-represented in training

---

## Key Findings

1. **Anti-overfitting strategy helped:** Frozen encoder + weight decay + early stopping mitigated overfitting on the small dataset. Training loss continued to decrease while validation loss began rising after the best epoch, suggesting early stopping was useful for checkpoint selection.
2. **Fine-tuning clearly improves over zero-shot Whisper:** whisper-tiny improves by 32.9% WER, while whisper-base improves by 47.9% WER on the same held-out test split.
3. **whisper-base is the strongest model:** fine-tuned whisper-base reaches 0.2544 WER / 0.2111 CER and outperforms fine-tuned whisper-tiny by 27.1% relative WER.
4. **Dataset limitations:**
   - MINDS-14 audio is 8 kHz → half Mel bins empty after upsampling to 16 kHz
   - Only 450 training samples limits learning capacity
   - Lowercased labels and evaluation predictions improve metric consistency, but may not match Whisper's cased pretraining distribution

---

## Recommendations for Improvement

1. **Audio augmentation:** Speed perturbation, SpecAugment
2. **Text casing:** Train on cased text, lowercase only at eval time
3. **Hyperparameter tuning:** Sweep learning rate and batch size
4. **Data filtering:** Remove very long/short utterances before training
5. **Larger models:** Try whisper-small or whisper-medium for better capacity
