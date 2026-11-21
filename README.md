---
title: Multimodal CX Platform
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time multimodal audio, speech emotion & sentiment CX monitor
---

# Multimodal CX Platform

An intentionally compact, local-first customer-experience monitor:

`audio → ASR + acoustic emotion + text sentiment → multimodal late fusion → attention → Redis Stream → FastAPI/WebSocket → dashboard`

It uses Redis **Streams**, not transient Pub/Sub: a browser reconnects with its last event ID and receives missed events. The dashboard's sample data is clearly labelled `SIMULATED`; it is never presented as model output.

## What is included

- `emotion_model/`: compact CNN, consistent log-Mel feature extraction, RAVDESS parser, and speaker-independent split.
- `scripts/prepare_ravdess.py`: makes a gender-balanced, speaker-independent 16/4/4 train/validation/test manifest from extracted RAVDESS speech data.
- `scripts/train_emotion.py`: trains on train speakers, selects by validation weighted F1, and writes held-out test metrics.
- `app/`: lazy-loading Whisper/sentiment/emotion adapters, durable event stream, FastAPI endpoints, and WebSocket feed.
- `static/`: dependency-free live dashboard.
- `docker-compose.yml`: API plus Redis for portable deployment.

## Local run

```bash
cd outputs/multimodal-cx-platform
source .venv/bin/activate
pip install -r requirements.txt
redis-server
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The **Load simulated demo events** button verifies the complete Redis → API → WebSocket → dashboard path without downloading models or using any customer audio.

To use the full audio endpoint, upload audio through http://127.0.0.1:8000/docs. The first request downloads the configured Whisper and sentiment checkpoints into ignored `models/cache/` storage. It reports `train_model_first` for emotion until the local CNN has been trained; it does not fabricate an emotion prediction.

## Train the actual emotion model

1. Download and extract the RAVDESS **audio speech actors** dataset from its [official Zenodo record](https://zenodo.org/records/1188976). Store the extracted actor folders under `data/raw/`, for example `data/raw/Actor_01/` through `data/raw/Actor_24/`.
2. Build a reproducible, speaker-independent manifest. The optional limit is useful for a first laptop smoke test.

```bash
python -m scripts.prepare_ravdess data/raw --limit-per-emotion 24
python -m scripts.train_emotion --manifest data/processed/manifest.csv --epochs 20
```

3. The resulting `models/emotion_cnn.pt` is automatically used by `/api/analyze`; `models/emotion_cnn.metrics.json` contains validation and unseen-speaker test weighted F1 plus the class-wise report.

For a defendable report, repeat runs with several seeds, retain every metrics JSON, and state that RAVDESS is acted English speech, not real customer-support calls.

## API

- `GET /api/health` — Redis and model readiness.
- `GET /api/events?last_event_id=0-0` — retained stream events after an ID.
- `POST /api/analyze` — multipart `audio` field; executes the real pipeline.
- `POST /api/demo/seed` — three **simulated** UI-verification events.
- `WS /ws/events?last_event_id=<id>` — reconnectable event feed.

## Docker

Docker is optional while iterating locally. Once the local path works:

```bash
docker compose up --build
```

The API will have no trained emotion model unless `models/emotion_cnn.pt` is present in the build context. Keep real recordings out of the repository and obtain consent before processing customer audio.


## Multimodal late fusion

The system explicitly combines the acoustic emotion output and linguistic sentiment output at
decision level. The current fusion rule is deterministic because the project does not yet have
a supervised customer-service dataset with labels for training a learned fusion classifier.

`F = alpha*S + (1-alpha)*E`

where `S` is signed sentiment confidence, `E` is emotion valence multiplied by emotion confidence,
and `alpha=0.5` initially. The resulting state is `positive`, `mixed`, or `negative`.
A negative fused state generates the `Customer needs attention` alert.

The 20 team recordings are demonstration data, not a formal benchmark. Actual model outputs must
be retained without replacing them with the intended scenario labels.

## Phase 2 — model comparison

The project now includes reproducible comparison scripts. **Do not copy published model-card scores into the final project table**; the scripts measure the candidates under the project's own protocol.

### 1. ASR: Whisper Base vs Whisper Small

RAVDESS uses two fixed speech statements, so the test split has known reference transcripts. The script measures word error rate (WER), mean inference latency, and total runtime.

```bash
python -m scripts.compare_asr --manifest data/processed/manifest.csv --limit 8
python -m scripts.compare_asr --manifest data/processed/manifest.csv
```

Output: `data/processed/asr_comparison.json`

### 2. Speech emotion: CNN vs Wav2Vec2 vs WavLM

The existing compact CNN is the project baseline. The comparison script trains Wav2Vec2 and WavLM classifiers on the **same speaker-independent RAVDESS train/validation split**, evaluates the same unseen-speaker test set, and reports accuracy, macro F1, weighted F1, per-class F1, training time, and evaluation time.

The default configuration freezes the low-level feature extractor to keep the experiment feasible on a 16 GB laptop. Use `--no-freeze-feature-extractor` only when deliberately running the more expensive full-backbone experiment.

```bash
python -m scripts.compare_emotion --manifest data/processed/manifest.csv --epochs 1 --limit-train 32 --limit-validation 16 --limit-test 16
python -m scripts.compare_emotion --manifest data/processed/manifest.csv --epochs 5
```

Output: `data/processed/emotion_comparison.json`; candidate checkpoints are stored under `models/comparison/`.

### 3. Sentiment: DistilBERT vs RoBERTa

Both candidates expose the same three sentiment classes: negative, neutral, and positive. The supplied `sentiment_demo.csv` contains the 20 scripted team scenarios and is intended only as a small exploratory/domain smoke test. It is **not** a statistically sufficient benchmark for the final scientific claim.

```bash
python -m scripts.compare_sentiment --input data/processed/sentiment_demo.csv
```

Output: `data/processed/sentiment_comparison.json`.

### Model-selection rule

Do not select a model from model-card numbers or from a single convenient example. The final production candidates are selected after measuring the same evaluation protocol. For emotion, the primary selection metric is test macro F1 because the eight classes are not perfectly balanced; weighted F1 and latency remain secondary evidence. For ASR, lower WER is better, with latency/resource cost reported alongside it. For sentiment, macro F1 is the primary comparison metric on a 3-class evaluation set.

The final architecture becomes:

`selected ASR + selected emotion model + selected sentiment model → multimodal late fusion → attention → Redis Stream → FastAPI/WebSocket → dashboard`
# multimodal-cx-platform
