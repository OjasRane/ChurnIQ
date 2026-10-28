# Verification record — 28 October 2026

## Executed successfully

- Python import/bytecode check: `python -m compileall -q backend/app backend/train_model.py`.
- Installed dependencies and executed training with Python 3.12.13, then repeated clean-model integration testing with **Python 3.11.15** (the deployment runtime).
- Fresh training under Python 3.11: `MODEL_DIR=/tmp/churniq-model311 PYTHONPATH=backend .venv311/bin/python -m pytest backend/tests -q` — **9 passed**. This constructed CatBoost, the logistic-regression baseline, the State_Code ablation, and the final full-data CatBoost refit from the bundled training CSV.
- Compared the complete evaluation metadata between Python 3.11 and 3.12 — identical metric/curve results.
- Final API suite: `PYTHONPATH=backend .venv311/bin/python -m pytest backend/tests -q` — **9 passed** after response-schema and financial probability-source changes.
- `npm install` — completed; npm reported **0 vulnerabilities** at installation time.
- `npm run build` — TypeScript check and production Vite build passed. Vendor chunks separate Recharts and Markdown; no oversized-chunk warning in the final build.
- Browser suite against the running FastAPI and Vite servers: `PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/google-chrome npm run test:e2e` — **3 passed**, final run in 8.5 seconds.
- Viewed desktop screenshot after disabling chart animation; charts render actual target counts and model importance. Mobile test confirmed no horizontal overflow at 390px.
- Verified all four `backend/data/` CSV copies are byte-identical to root source files.

## Backend assertions

CSV discovery by columns; exact 2,280 training rows / 357 positives / 1,923 negatives; feature order excludes IDs; real validation confusion matrices total 570; baseline and ablation metrics; model serialization/reload; one cached model service; probability range; 12 native SHAP contributions; SHAP log-odds additivity; missing/extra/invalid feature errors; mixed-validity batch prediction; identifier preservation; invalid CSV errors; exactly 570 model-produced competition classes with exact original sample IDs/order; discounted-contribution formula; positive-benefit budget allocation; zero-cost ROI; invalid probability rejection; health; overview/features/importance/quality endpoints; 3,624 noisy missing cells; allowed/rejected CORS origins; missing-Gemini-key errors without breaking health.

## Browser assertions

Actual API-backed overview counts; all six pages navigable; 12-field dynamic prediction form; modified feature submission; prediction and SHAP display; 570-row CSV upload and pagination; downloaded competition CSV header and row count; changed financial slider updates calculated contribution; Gemini missing-key error; model metrics and noisy data diagnostics; mobile menu and responsive layout; malformed CSV error; simulated API failure with successful retry. No uncaught page errors in the main flow.

## Actual CatBoost validation result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9631578947 |
| Precision | 0.8617021277 |
| Recall | 0.9101123596 |
| F1 | 0.8852459016 |
| ROC-AUC | 0.9922212619 |
| PR-AUC (average precision) | 0.9637485165 |
| Brier score | 0.0258011811 |

Confusion matrix: `[[468, 13], [8, 81]]`. These are 570-row holdout results, not externally validated deployment performance. See README and dashboard limitations.

## Remaining external blockers

- **No public deployment completed.** No GitHub, Render, or Vercel credentials are configured and their integrations are not confirmed connected. The earlier read-only `.git` restriction was resolved during the deployment audit: a real Git repository is initialized on `main`. Dataset provenance is now recorded in README.md based on the project owner's confirmation of the UCI CC BY 4.0 source, assuming no third-party data was incorporated. Source files, Render blueprint, Vercel configuration, and account-by-account instructions are ready.
- **No live Gemini response verified.** GEMINI_API_KEY is absent. Missing-key behavior was tested; successful provider access/model availability requires the user's backend credentials. Default requested model is configurable.
- FastAPI/Starlette emits a TestClient deprecation warning about `httpx` in the installed version. All tests pass; it does not affect application requests.
- Browser/API tests required execution outside the filesystem/network sandbox because it blocks event-loop sockets, loopback connections, and Chrome startup. The approved executions passed.

## Final deployment audit

See `DEPLOYMENT.md`. The existing Render and Vercel configuration was inspected unchanged; production URL injection was verified by a separate temporary frontend build. Additional credential/cache ignore rules were added, and a heuristic candidate-file credential scan passed. No application redesign, code refactoring, feature additions, or local model training occurred. No commit/push or public deployment is claimed.
