# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

"Biometric Virtual Lab": an educational FastAPI app (Python 3.11/3.12) where students step through fingerprint, face and iris recognition pipelines, compare samples, compute FAR/FRR/EER, write graded Python code, and watch a NumPy CNN train. All user-facing source text (API error messages, task JSON, UI, README) is in **Russian**; English and Kazakh come from the catalogs in `app/i18n/` (see "Languages"). Keep source strings Russian when adding strings and add the translations.

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
python -m scripts.download_pyodide     # Pyodide 314.0.7 + numpy/scipy/scikit-image into app/web/vendor/pyodide/ (gitignored) for browser-side example runs
```

Config comes from `.env` (see `.env.example`), loaded by `app/core/config.py`: `TUTOR_PROVIDER=rule|openai`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `DB_PATH`, `AUTO_DOWNLOAD_MODELS`. The `openai` package is not in `requirements.txt` (commented out) and must be installed separately for the OpenAI tutor. No linter/formatter is configured. Tests set `DB_PATH` to a temp file *before* importing `app.main`; follow that pattern in new API tests.

## Architecture

**Biometric modules (plugin contract).** `app/modules/base.py` defines `BiometricModule` (`list_samples(source, q, limit)`, `get_sample_preview`, `process`, `compare`, optional `evaluate`/`upload`/`relief`/`info`/`stress`). Implementations live in `app/modules/fingerprint/`, `app/modules/face/`, `app/modules/iris/` and `app/modules/voice/` and are registered in `app/modules/registry.py`. `app/api/v1/biometrics.py` exposes generic routes `/api/v1/biometrics/{modality}/...` that dispatch through the registry, so a new modality needs only a module + `register(...)` + a tab in `app/web/index.html`. Each module validates its `params: dict` with a Pydantic model in its `params.py` (bounds there drive the UI sliders); the API's `_guard` maps `ValidationError`→422, `KeyError`→404, `ValueError`→400.

- Fingerprint: `pipeline.py` (8 visual steps: gray → segmentation → CLAHE → orientation → Gabor → binarize → skeleton → minutiae), `minutiae.py`, `matching.py` (Hough voting over rotation/shift). Samples come from `datasets/synthetic/` via `synthetic.py` or from real sets via `real.py` (FVC2002/2004 set B and SOCOFing in `datasets/real/`, not in Git); the data source is chosen with `params.source` and `GET .../samples?source=&q=&limit=`. SOCOFing has one image per finger, so `evaluate` refuses it and `stress()` (original vs altered copies) is used instead.
- Face: two backends — classic Haar + LBP histograms (no downloads; subtracts the gallery "mean face", needs 3+ people) and YuNet + SFace ONNX (`models/`, downloaded in background at startup by `face/models.py`). Gallery is `datasets/faces/<source>/<person>/<img>.jpg` (e.g. LFW) or the flat `datasets/faces/<person>/<img>.jpg`; the scan is cached for 60 s. Results are kept in an LRU cache in `face/module.py`.
- Iris: `synthetic.py` (known circles) or `real.py` (CASIA-Iris-Interval in `datasets/real/casia_iris_interval/`, no ground-truth circles).
- Voice: `app/modules/voice/` (speaker recognition: `audio.py` VAD/MFCC/f0 in NumPy, `synthetic.py` 12 formant-synthesized speakers saying one phrase, `features.py` templates + z-score similarity, `render.py` step images, `module.py`). Uploads are WAV (the browser converts any audio to 16 kHz WAV); `BiometricModule.upload_kind` ("image"/"audio") selects the upload check and `GET .../sample/{id}.wav` plays a sample.
- Uploads (`modules/uploads.py`) live only in process memory and are private per anonymous session (`core/session.py`, cookie `lab_sid`); upload limits are in `api/v1/biometrics.py`.

**Code practicum (sandbox + grading).** Task definitions are JSON in `app/tasks/NN_name.json` (text, starter code, 4 hint levels, solution, quiz question, rubric weights). The trusted reference implementation and test cases for each task live in `app/sandbox/reference.py` (`SPECS` dict, keyed by the task's `spec` field) — adding a task needs both. `sandbox/tasks.py` re-reads the JSON files on every call (keyed by each file's `id` field), so task edits need no restart. Flow: `api/v1/code.py` → `sandbox/executor.py` (AST safety check in `safety.py`, then spawns `python -I sandbox/runner.py` with JSON on stdin, result read from a `@@RESULT@@` stdout line; max 2 concurrent runs) → `services/grader.py` applies the rubric → `services/attempts.py` stores attempts. Full solutions are only revealed server-side after 3 graded attempts; `public_task()` strips solution/hints/answer before sending to the client. On Windows the sandbox enforces only a timeout (rlimits are POSIX-only). There are 12 tasks (`app/tasks/01..12_*.json`, specs in `reference.py`; each spec needs >=4 basic and >=3 edge cases, and `tests/test_code_tasks.py` needs a typical wrong solution per task).

**Browser-side example run.** The "Запустить на примере" button (`/run` mode) runs in the browser: `app/web/codelab/worker.js` (module Web Worker, Pyodide from `/vendor/pyodide/`) executes `app/web/codelab/lab_runner.py` on the example from `GET /api/v1/code/tasks/{id}/example` (`app/sandbox/serialize.py` encodes args/expected; `reference.CMP_DESC` describes how to compare; no solutions, hints or edge tests leave the server). The page enforces a 3 s limit by `worker.terminate()` + respawn, then POSTs the result to `/api/v1/code/feedback` (server builds `code_tutor.feedback`). If Pyodide is unavailable the page falls back to the server `/run`. `/submit` (grading, attempts, solution unlock) is unchanged and never trusts the browser. The browser loads the server's own `safety.py` via `/codelab/safety.py`, so block messages match. Comparators in `lab_runner.py` mirror `reference.py`; `tests/test_code_browser.py` checks parity (browser runner vs `execute`), `tests/test_code_browser_ui.py` (Playwright + Edge/Chrome, skipped without them) checks the real Pyodide path. Known harmless difference: Pyodide (wasm32) default int is int32, so a dtype message may say `int32` where the server says `int64`.

**Training.** `app/training/` is a from-scratch NumPy CNN (`cnn.py`), data loading from synthetic or real (FVC, CASIA) fingerprints/iris and the face gallery (`datasets.py`, chosen with `dataset`; `/api/v1/training/datasets`), the trained-net registry (`registry.py`, one net per modality, remembers its dataset and refuses other datasets), and a singleton background-thread `Trainer` (`trainer.py`) that stores a per-epoch snapshot (loss, EER, filters, feature maps, embeddings, weight deltas) served by `/api/v1/training/{start,stop,status,snapshot/<epoch>}`.

**Services / persistence.** `app/core/db.py` is a raw-sqlite3 layer (tables `experiments`, `attempts`, `events`, created in `init_db()` at startup); all DB access goes through `app/services/*` so it can later be swapped for PostgreSQL. `services/events.record_event` logs every student action (learning-analytics basis). `services/tutor.py` has `TutorProvider` classes (rule-based default, OpenAI optional with automatic fallback to rules); `services/code_tutor.py` handles error analysis/hints for the code practicum.

**Assistant.** `services/assistant.py` parses a spoken/typed phrase into actions (`/api/v1/assistant/parse`, rule-based, no LLM); `index.html` (block "Помощник лаборатории") executes them by clicking the same buttons as the user and runs the guided tour; STT is the browser's; TTS is either the browser's or the optional server voice (`services/tts.py`, Piper ONNX voices in `models/tts/`, `pip install -r requirements-tts.txt` + `python -m scripts.download_tts`; falls back to the browser voice when absent). `labplus/` (third-party add-on mounted in `app/main.py`) provides the voice/metrics/robustness/standards pages. Keep the action names in `assistant.py` and `asDo()` in sync.

**Languages (ru / kk / en).** Source strings stay Russian and are the catalog keys (gettext-style): `app/i18n/*.i18n` blocks `ru>/en>/kk>` with `{0}` placeholders (`ui_*` and `client.i18n` are also sent to the browser, `server.i18n`, `tasks.i18n`, `assistant.i18n`, `intent.i18n`, `server_extra.i18n` stay server-side). `app/core/i18n.py`: `current_lang()` contextvar set by `I18nMiddleware` (query `?lang=`, cookie `lab_lang`, `Accept-Language`), which translates JSON responses on output (exact -> `{n}` templates -> sentence split; unknown text stays Russian); `inject_page` puts `LAB_LANG`/`LAB_CAT` into the HTML. Browser: `app/web/i18n/i18n.js` (`LT(msgid, [args])` for literals, `LTX(text)` for dynamic Russian text; do not name globals `T`/`TX`, LAB+ uses `T`). Task JSON in `app/tasks/` is translated through `tasks.i18n` keyed by the Russian text (starter-code comments per line). Assistant commands in en/kk are mapped to Russian anchor words in `services/assistant_i18n.py` (rules stay in `assistant.py`); replies are built from translated parts. New user-facing strings: write them in Russian, wrap JS ones in `LT()`, add the `en`/`kk` block; `tests/test_i18n.py` checks completeness and placeholder parity. The Kazakh text is machine-assisted and needs a native speaker's review (README "Три языка"). The `.i18n` files are the source of truth (edit them directly).

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

Before changing behaviour, run `python -m pytest -q` (565 passed, 7 skipped, about 9 min; the 4 Playwright UI tests need Edge/Chrome and the downloaded Pyodide, otherwise they are skipped) and report what fails. Keep new user-facing strings in Russian.
All tabs live in `app/web/index.html` (see `setMod` and its `SECTIONS` map).
## Deployment

New env vars (see `.env.example`, README "Переменные окружения"): `DATABASE_URL` (PostgreSQL; otherwise SQLite from `DB_PATH`), `DB_POOL_MAX_SIZE`, `TUTOR_RATE_LIMIT_PER_HOUR`, `SANDBOX_MAX_CONCURRENT`, `PORT`, `COOKIE_SECURE`, `SUBJECT_HEADER` (ID пользователя сайта-хозяина из заголовка; tables events/attempts/experiments/training_runs have nullable `session_id` (hash of the lab_sid cookie) and `subject`, filled via `app.core.session.owner()`; never expose them to the client). SQL in `app/core/db.py` is written once with `?` placeholders and runs on both SQLite and PostgreSQL; use `c.insert(...)` and `json_field(...)` instead of dialect-specific SQL.
Load test: `scripts/loadtest/locustfile.py` (local server only). In-memory state (uploads, training, rate limits, caches) is per process, so deploy with a single instance; see README "Ограничения при деплое".
