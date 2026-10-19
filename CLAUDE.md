# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

"Biometric Virtual Lab": an educational FastAPI app (Python 3.11/3.12) where students step through fingerprint and face recognition pipelines, compare samples, compute FAR/FRR/EER, write graded Python code, and watch a NumPy CNN train. All user-facing text (API error messages, task JSON, UI, README) is in **Russian**. Keep it that way when adding strings.

## Commands

```
pip install -r requirements.txt
uvicorn app.main:app --reload          # UI at http://127.0.0.1:8000, OpenAPI at /docs
pytest                                 # all tests
pytest tests/test_fingerprint.py::test_name   # single test
python -m scripts.download_models      # YuNet/SFace ONNX into models/
python -m scripts.prepare_lfw          # populate datasets/faces/ from LFW
python -m scripts.generate_dataset     # regenerate datasets/synthetic/ (12 fingers x 3 impressions)
```

Config comes from `.env` (see `.env.example`), loaded by `app/core/config.py`: `TUTOR_PROVIDER=rule|openai`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `DB_PATH`, `AUTO_DOWNLOAD_MODELS`. The `openai` package is not in `requirements.txt` (commented out) and must be installed separately for the OpenAI tutor. No linter/formatter is configured. Tests set `DB_PATH` to a temp file *before* importing `app.main`; follow that pattern in new API tests.

## Architecture

**Biometric modules (plugin contract).** `app/modules/base.py` defines `BiometricModule` (`list_samples`, `get_sample_preview`, `process`, `compare`, optional `evaluate`/`upload`/`relief`/`info`). Implementations live in `app/modules/fingerprint/` and `app/modules/face/` and are registered in `app/modules/registry.py`. `app/api/v1/biometrics.py` exposes generic routes `/api/v1/biometrics/{modality}/...` that dispatch through the registry, so a new modality needs only a module + `register(...)` + a tab in `app/web/index.html`. Each module validates its `params: dict` with a Pydantic model in its `params.py` (bounds there drive the UI sliders); the API's `_guard` maps `ValidationError`→422, `KeyError`→404, `ValueError`→400.

- Fingerprint: `pipeline.py` (8 visual steps: gray → segmentation → CLAHE → orientation → Gabor → binarize → skeleton → minutiae), `minutiae.py`, `matching.py` (Hough voting over rotation/shift). Samples come from `datasets/synthetic/` via `synthetic.py`; custom datasets plug in by replacing `_gray()` in `fingerprint/module.py`. Metrics are cached after the first (slow, ~8–10 s) computation.
- Face: two backends — classic Haar + LBP histograms (no downloads; subtracts the gallery "mean face", needs 3+ people) and YuNet + SFace ONNX (`models/`, downloaded in background at startup by `face/models.py`). Gallery is `datasets/faces/<person>/<img>.jpg`. Results are kept in an LRU cache in `face/module.py`.

**Code practicum (sandbox + grading).** Task definitions are JSON in `app/tasks/NN_name.json` (text, starter code, 4 hint levels, solution, quiz question, rubric weights). The trusted reference implementation and test cases for each task live in `app/sandbox/reference.py` (`SPECS` dict, keyed by the task's `spec` field) — adding a task needs both. `sandbox/tasks.py` re-reads the JSON files on every call (keyed by each file's `id` field), so task edits need no restart. Flow: `api/v1/code.py` → `sandbox/executor.py` (AST safety check in `safety.py`, then spawns `python -I sandbox/runner.py` with JSON on stdin, result read from a `@@RESULT@@` stdout line; max 2 concurrent runs) → `services/grader.py` applies the rubric → `services/attempts.py` stores attempts. Full solutions are only revealed server-side after 3 graded attempts; `public_task()` strips solution/hints/answer before sending to the client. On Windows the sandbox enforces only a timeout (rlimits are POSIX-only).

**Training.** `app/training/` is a from-scratch NumPy CNN (`cnn.py`), data loading from synthetic fingerprints or the face gallery (`datasets.py`), and a singleton background-thread `Trainer` (`trainer.py`) that stores a per-epoch snapshot (loss, EER, filters, feature maps, embeddings, weight deltas) served by `/api/v1/training/{start,stop,status,snapshot/<epoch>}`.

**Services / persistence.** `app/core/db.py` is a raw-sqlite3 layer (tables `experiments`, `attempts`, `events`, created in `init_db()` at startup); all DB access goes through `app/services/*` so it can later be swapped for PostgreSQL. `services/events.record_event` logs every student action (learning-analytics basis). `services/tutor.py` has `TutorProvider` classes (rule-based default, OpenAI optional with automatic fallback to rules); `services/code_tutor.py` handles error analysis/hints for the code practicum.

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

Before changing behaviour, run `python -m pytest -q` (about 40 tests, ~15 s) and report what fails. Keep new user-facing strings in Russian.
All tabs live in `app/web/index.html` (see `setMod` and its `SECTIONS` map).