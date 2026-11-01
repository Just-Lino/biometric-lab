# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

"Biometric Virtual Lab": an educational FastAPI app (Python 3.11/3.12) where students step through fingerprint, face and iris recognition pipelines, compare samples, compute FAR/FRR/EER, write graded Python code, and watch a NumPy CNN train. All user-facing text (API error messages, task JSON, UI, README) is in **Russian**. Keep it that way when adding strings.

## Commands

```
pip install -r requirements.txt
uvicorn app.main:app --reload          # UI at http://127.0.0.1:8000, OpenAPI at /docs
pytest                                 # all tests
pytest tests/test_fingerprint.py::test_name   # single test
python -m scripts.download_models      # YuNet/SFace ONNX into models/
python -m scripts.prepare_lfw          # populate datasets/faces/ from LFW
python -m scripts.prepare_real --casia <dir> --soco <dir>   # copy downloaded CASIA-Iris-Interval / SOCOFing into datasets/real/ (no network)
python -m scripts.generate_dataset     # regenerate datasets/synthetic/ (12 fingers x 3 impressions)
```

Config comes from `.env` (see `.env.example`), loaded by `app/core/config.py`: `TUTOR_PROVIDER=rule|openai`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `DB_PATH`, `AUTO_DOWNLOAD_MODELS`. The `openai` package is not in `requirements.txt` (commented out) and must be installed separately for the OpenAI tutor. No linter/formatter is configured. Tests set `DB_PATH` to a temp file *before* importing `app.main`; follow that pattern in new API tests.

## Architecture

**Biometric modules (plugin contract).** `app/modules/base.py` defines `BiometricModule` (`list_samples(source, q, limit)`, `get_sample_preview`, `process`, `compare`, optional `evaluate`/`upload`/`relief`/`info`/`stress`). Implementations live in `app/modules/fingerprint/`, `app/modules/face/`, `app/modules/iris/` and `app/modules/voice/` and are registered in `app/modules/registry.py`. `app/api/v1/biometrics.py` exposes generic routes `/api/v1/biometrics/{modality}/...` that dispatch through the registry, so a new modality needs only a module + `register(...)` + a tab in `app/web/index.html`. Each module validates its `params: dict` with a Pydantic model in its `params.py` (bounds there drive the UI sliders); the API's `_guard` maps `ValidationError`→422, `KeyError`→404, `ValueError`→400.

- Fingerprint: `pipeline.py` (8 visual steps: gray → segmentation → CLAHE → orientation → Gabor → binarize → skeleton → minutiae), `minutiae.py`, `matching.py` (Hough voting over rotation/shift). Samples come from `datasets/synthetic/` via `synthetic.py` or from real sets via `real.py` (FVC2002/2004 set B and SOCOFing in `datasets/real/`, not in Git); the data source is chosen with `params.source` and `GET .../samples?source=&q=&limit=`. SOCOFing has one image per finger, so `evaluate` refuses it and `stress()` (original vs altered copies) is used instead.
- Face: two backends — classic Haar + LBP histograms (no downloads; subtracts the gallery "mean face", needs 3+ people) and YuNet + SFace ONNX (`models/`, downloaded in background at startup by `face/models.py`). Gallery is `datasets/faces/<source>/<person>/<img>.jpg` (e.g. LFW) or the flat `datasets/faces/<person>/<img>.jpg`; the scan is cached for 60 s. Results are kept in an LRU cache in `face/module.py`.
- Iris: `synthetic.py` (known circles) or `real.py` (CASIA-Iris-Interval in `datasets/real/casia_iris_interval/`, no ground-truth circles).
- Voice: `app/modules/voice/` (speaker recognition: `audio.py` VAD/MFCC/f0 in NumPy, `synthetic.py` 12 formant-synthesized speakers saying one phrase, `features.py` templates + z-score similarity, `render.py` step images, `module.py`). Uploads are WAV (the browser converts any audio to 16 kHz WAV); `BiometricModule.upload_kind` ("image"/"audio") selects the upload check and `GET .../sample/{id}.wav` plays a sample.
- Uploads (`modules/uploads.py`) live only in process memory and are private per anonymous session (`core/session.py`, cookie `lab_sid`); upload limits are in `api/v1/biometrics.py`.

**Code practicum (sandbox + grading).** Task definitions are JSON in `app/tasks/NN_name.json` (text, starter code, 4 hint levels, solution, quiz question, rubric weights). The trusted reference implementation and test cases for each task live in `app/sandbox/reference.py` (`SPECS` dict, keyed by the task's `spec` field) — adding a task needs both. `sandbox/tasks.py` re-reads the JSON files on every call (keyed by each file's `id` field), so task edits need no restart. Flow: `api/v1/code.py` → `sandbox/executor.py` (AST safety check in `safety.py`, then spawns `python -I sandbox/runner.py` with JSON on stdin, result read from a `@@RESULT@@` stdout line; max 2 concurrent runs) → `services/grader.py` applies the rubric → `services/attempts.py` stores attempts. Full solutions are only revealed server-side after 3 graded attempts; `public_task()` strips solution/hints/answer before sending to the client. On Windows the sandbox enforces only a timeout (rlimits are POSIX-only).

**Training.** `app/training/` is a from-scratch NumPy CNN (`cnn.py`), data loading from synthetic or real (FVC, CASIA) fingerprints/iris and the face gallery (`datasets.py`, chosen with `dataset`; `/api/v1/training/datasets`), the trained-net registry (`registry.py`, one net per modality, remembers its dataset and refuses other datasets), and a singleton background-thread `Trainer` (`trainer.py`) that stores a per-epoch snapshot (loss, EER, filters, feature maps, embeddings, weight deltas) served by `/api/v1/training/{start,stop,status,snapshot/<epoch>}`.

**Services / persistence.** `app/core/db.py` is a raw-sqlite3 layer (tables `experiments`, `attempts`, `events`, created in `init_db()` at startup); all DB access goes through `app/services/*` so it can later be swapped for PostgreSQL. `services/events.record_event` logs every student action (learning-analytics basis). `services/tutor.py` has `TutorProvider` classes (rule-based default, OpenAI optional with automatic fallback to rules); `services/code_tutor.py` handles error analysis/hints for the code practicum.

**Assistant.** `services/assistant.py` parses a spoken/typed phrase into actions (`/api/v1/assistant/parse`, rule-based, no LLM); `index.html` (block "Помощник лаборатории") executes them by clicking the same buttons as the user and runs the guided tour; STT is the browser's; TTS is either the browser's or the optional server voice (`services/tts.py`, Piper ONNX voices in `models/tts/`, `pip install -r requirements-tts.txt` + `python -m scripts.download_tts`; falls back to the browser voice when absent). `labplus/` (third-party add-on mounted in `app/main.py`) provides the voice/metrics/robustness/standards pages. Keep the action names in `assistant.py` and `asDo()` in sync.

**Frontend.** Single file `app/web/index.html` (served at `/`), vanilla JS with Plotly, Three.js, and Monaco loaded from CDN. The "neural network playground" tab runs entirely in the browser.

## Known limitations (intentional for the MVP)

Matcher is educational (minutia orientation is mod 180°, rotation ≤ ±40°). Metrics on synthetic data give EER ≈ 0 by design. The 3D relief is a pseudo-3D heightmap from the image, not a reconstruction.

## Environment (developer machine)

Windows + PowerShell, Python 3.13 (the app also works on 3.11/3.12). Give shell commands in PowerShell syntax
(no `&&`; run tools as `python -m ...`). `opencv-python` must stay `>=4.9,<5`: OpenCV 5.x has no `cv2.CascadeClassifier`,
which the Haar face detector needs.

## Do not touch

`datasets/` (synthetic fingerprints, face gallery) and `models/*.onnx` are data, not code: never regenerate or delete them
unless asked. `data/lab.db` is the student's local database. Task solutions in `app/tasks/*.json` must never be exposed to the client.

## Working rules

Before changing behaviour, run `python -m pytest -q` (195 tests, about 8 min) and report what fails. Keep new user-facing strings in Russian.
All tabs live in `app/web/index.html` (see `setMod` and its `SECTIONS` map).