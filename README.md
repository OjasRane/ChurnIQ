# ChurnIQ — Customer Retention Intelligence

A working React + FastAPI application for a college Engineering Economics project: real CatBoost inference, native SHAP explanations, portfolio scoring, competition submissions, transparent retention scenarios, and optional Gemini interpretation.

## Start locally

Use Python **3.11** and Node **22+**. Run these from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open **http://localhost:5173**. Vite proxies `/api` to the backend. Do **not** copy the frontend production example URL into local development unless using a remote backend. FastAPI interactive API docs: **http://localhost:8000/docs**.

The trained model is already generated in `backend/artifacts/` in the working directory. These generated artifacts are ignored by Git: on a clean checkout the API trains once during startup, saves artifacts, then reuses the in-memory model. Render pre-trains during its build. No API request retrains the model.

Optional Gemini: put `GEMINI_API_KEY` in `backend/.env` and restart the backend. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`; change it if your account lacks access. There is no frontend API key. Missing credentials return a clear 503; provider, quota, or model errors return a generic 502 without exposing secrets. Other features continue working.

## What is implemented

- **Overview:** actual dataset counts, target distribution, validation performance, final-model feature importance, validation-cohort high-risk counts, latest prediction when one exists in the current session.
- **Predict Churn:** backend-defined schema for all 12 original numeric features, observed ranges and median defaults, churn probability, binary class, risk tier, and signed native CatBoost SHAP contributions in log-odds.
- **Bulk Analysis:** drag-and-drop CSV, up to 5 MB / 10,000 rows, individual invalid-row errors, optional customer IDs, segmentation, filtering, pagination, CSV export, exact competition submission export.
- **Profit Simulator:** configurable revenue, margin, duration, monthly discount, intervention cost, assumed effectiveness, budget, and monetary unit. Uses a hypothetical probability, the latest prediction, or the valid uploaded portfolio. Python computes discounted contribution, value at risk, gross/net benefits, ROI, sensitivity, exact budget allocation under identical customer costs, and row-level CSV export. Changing sliders never retrains the model.
- **AI Analyst:** prediction interpretation, retention hypotheses, economic interpretation, executive report, project conclusion, Markdown rendering, clipboard copy. The backend recomputes predictions/economics from structured inputs and supplies measured metrics. No CSV IDs, names, or arbitrary uploaded columns enter prompts.
- **Model Insights:** CatBoost/baseline/ablation comparison, ROC and precision-recall curves, confusion matrix, native SHAP importance, methodology, data quality, and limitations.

Client analyses live in React memory for the current session; no database or authentication is used. Reloading clears them. Uploaded files are not persisted.

## Dataset Attribution

- **Original dataset:** Iranian Churn, UCI Machine Learning Repository (2020).
- **DOI:** [https://doi.org/10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **License URL:** [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

The bundled competition CSVs are derived from the original dataset. The competition version contains modified/anonymized feature names, added noise, and training/test splits. This provenance statement is based on the project owner's confirmation and assumes that no third-party data was incorporated. The original data and these modifications should be distinguished when citing or redistributing the files.

UCI Machine Learning Repository and the original dataset contributors do not endorse this application. This attribution concerns the dataset and does not assign a license to the application code.

## Dataset integrity

The CSVs are discovered by columns, not hardcoded filenames. Their original feature names are preserved. Only the 2,280-row competition training file with binary `Outcome` is used to fit models. There are **357 positive** and **1,923 negative** records, with no missing training cells. `id` is never a feature. The 570 test records are unlabeled. Submission placeholder predictions are never interpreted as labels.

The separate 2,850-row Iranian dataset has **3,624 missing cells** and is used only for diagnostics. There is no row matching, concatenation, or automatic mapping of noisy features to competition features. `Mission_Priority_Index` is not treated as lifetime value, revenue, or any other verified monetary field.

Root `data/` remains the source data. **Identical copies are tracked inside `backend/data/`**, because Render excludes files outside `rootDir: backend`. After modifying source data run:

```bash
python scripts/sync_data.py
```

Commit both locations. Backend data resolution: explicit `DATA_DIR`, otherwise `backend/data/`, otherwise repository `data/`. Paths are resolved from the Python module, independently of the working directory.

Test IDs are unpadded (`1`); sample IDs are padded (`001`). Export matches canonical numeric IDs, checks uniqueness and equal ID sets, predicts with the final model, and emits the **original sample ID strings and order**. Output contains exactly `id,prediction`, 570 rows, binary model-produced classes.

## Measured model results

Fixed 75/25 stratified split, `random_state=42`: **1,710 fitting rows**, **570 validation rows** (89 positive, 481 negative). CatBoost uses 450 iterations, depth 6, learning rate 0.05, L2 regularization 5, balanced class weights, two CPU threads, fixed seed, and threshold 0.5. Numeric-only features retain the same order during fitting and inference. The baseline fits median imputation, scaling, and balanced logistic regression **only on fitting rows**. No tuning, early stopping, or test data fitting is performed.

| Held-out measure | CatBoost | Logistic Regression | CatBoost without State_Code |
| --- | ---: | ---: | ---: |
| Accuracy | 0.9632 | 0.8439 | 0.9754 |
| Precision | 0.8617 | 0.5000 | 0.8788 |
| Recall | 0.9101 | 0.9213 | 0.9775 |
| F1 | 0.8852 | 0.6482 | 0.9255 |
| ROC-AUC | 0.9922 | 0.9394 | 0.9947 |
| PR-AUC (average precision) | 0.9637 | 0.7358 | 0.9714 |
| Brier score | 0.0258 | 0.1093 | 0.0246 |

CatBoost confusion matrix (actual rows, predicted columns): `[[468, 13], [8, 81]]`.

The primary model was chosen in advance. Its final inference instance is fitted on all labeled training rows; the metrics above describe the **held-out evaluation instance**, not an independent evaluation of the final refit. A fingerprint covers training CSV contents, feature order, hyperparameters, and pipeline revision; changed training data triggers rebuilding. Metadata persists feature order, model settings, split sizes, metrics, curves, and limitations. Native `.cbm` serialization avoids Python pickle loading.

**Limitations:** one random split, no external or temporal evaluation, unknown feature semantics and timestamps, possible relationships among records, no causal intervention outcomes, no separately calibrated probability estimate. `State_Code` could be leakage-prone; the better ablation result on this split is reported openly, but does not resolve timing or leakage concerns. SHAP describes associations in model log-odds, not causal effects. Global SHAP uses the first 128 validation rows; per-customer SHAP runs only on demand. Risk tiers are presentation conventions: low <0.3, medium 0.3–0.7, high ≥0.7; binary class uses 0.5.

The survey's **external research ROC-AUC 0.9823 is not our model result**. The application cites it only in the methodology with that distinction.

## Financial methodology

For monthly revenue `R`, contribution margin `m`, monthly discount rate `d`, and duration `T`:

```text
Discounted contribution = sum(R * m / (1 + d)^t, t=1..T)
Value at risk = churn probability * discounted contribution
Gross retention benefit = value at risk * assumed effectiveness
Net benefit = gross retention benefit - intervention cost
ROI (%) = net benefit / intervention cost * 100
```

Zero intervention cost produces `null` ROI (displayed as N/A), not infinity. Budget allocation sorts strictly positive net-benefit customers descending and takes the largest affordable prefix. This is **exact under the stated identical-cost assumptions**, not a claim of solving heterogeneous-cost campaigns or uplift optimization. The portfolio value-at-risk KPI covers all supplied valid records; campaign gross benefit, net benefit, spend, and ROI cover selected records. Sensitivity shows the entire cohort before selection, so its net benefit may be negative even when no loss-making campaign is selected.

All monetary results are **scenario estimates**, not experimentally established incremental profits. Effectiveness is an assumption, not causal uplift. No intervention is guaranteed to work.

## Verification

```bash
# From the repository root, with the virtual environment active:
python -m compileall -q backend/app backend/train_model.py
PYTHONPATH=backend python -m pytest backend/tests -q

# From frontend/:
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests expect the backend at port 8000 and Vite at port 5173, using the development proxy. They cover actual API-backed navigation, form submission, SHAP rendering, 570-row uploads, a downloaded submission, sliders, missing-AI-key behavior, responsive mobile navigation, CSV errors, and API retry. Run them without a Gemini key for the missing-key assertion. To use an installed Chrome instead of Playwright's browser:

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/google-chrome npm run test:e2e
```

Backend tests verify detection, class distribution, actual model comparisons, saved-model reload, native SHAP additivity, single/bulk inference and validation, exact 570-row submission IDs and direct model output, discounted financial formulas and budget selection, zero cost, API metadata, CORS, and missing Gemini credentials.

To force a clean separate training check without altering saved local artifacts:

```bash
MODEL_DIR=/tmp/churniq-clean-model PYTHONPATH=backend python -m pytest backend/tests -q
```

No successful live Gemini response or public deployment is claimed without account access. Test results and public deployment status are documented in `VERIFICATION.md`.

## Deploy to GitHub, Render, and Vercel

For the final audit, private GitHub publication commands, dataset attribution, and exact dashboard sequence, follow [DEPLOYMENT.md](DEPLOYMENT.md). A real Git repository is now initialized on `main`; dataset provenance is recorded above, and publication is pending account access.

### 1. GitHub

Retain the Dataset Attribution section when sharing the bundled dataset derivatives. Create a **private** repository named `churniq` in your GitHub account. From this directory in your own terminal:

```bash
git add .
git commit -m "Build ChurnIQ retention intelligence application"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/churniq.git
git push -u origin main
```

Review staged files before committing. `.env`, virtual environments, generated artifacts, node_modules, build output, and browser artifacts are ignored. Both data directories are prepared for inclusion with the attribution above, under the stated assumption that no third-party data was incorporated. The earlier read-only placeholder restriction has been resolved: this directory now contains a real Git repository on `main`. GitHub authentication is still required before publication.

### 2. Render backend

Connect the GitHub repo to Render. You can use **New → Blueprint** with the root `render.yaml`, or create a **Python Web Service** manually:

| Setting | Value |
| --- | --- |
| Root Directory | `backend` |
| Python | `3.11.15` (set `PYTHON_VERSION`) |
| Build Command | `pip install -r requirements.txt && python train_model.py` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check | `/api/health` |

Set `CORS_ORIGINS` to your exact frontend origin, e.g. `https://churniq-yourteam.vercel.app`. Multiple origins are comma separated. Do not include `/` paths. Set `GEMINI_MODEL=gemini-3.5-flash-lite` and optionally set `GEMINI_API_KEY` as a secret. Omitting the key disables only Gemini. Training during build creates the saved `.cbm` and metadata within the deployed service; startup loads those artifacts and has a one-time fallback if absent. Use one Uvicorn worker for this small demonstration app. Runtime changes to model files are ephemeral on Render; bundled data remains available after restart.

Wait for the service to become healthy. Record the backend origin, e.g. `https://churniq-api.onrender.com`. Low-cost hosting may sleep or start slowly; use the dashboard's Retry after the API wakes.

### 3. Vercel frontend

Import the same GitHub repository into Vercel:

| Setting | Value |
| --- | --- |
| Framework | Vite |
| Root Directory | `frontend` |
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Environment Variable | `VITE_API_BASE_URL=https://YOUR-API.onrender.com` |

Set the backend **origin only**, without `/api`. Vite reads this at build time; changing it requires a new frontend deployment. No Vercel serverless backend is used. After Vercel assigns its production URL, update Render's `CORS_ORIGINS` to that exact origin and restart/redeploy the backend. Add explicitly approved preview origins separately if needed.

### 4. Public smoke test

1. Visit `https://YOUR-API.onrender.com/api/health`: expect `status: ok`, `model_ready: true`, and the correct `ai_available` state.
2. Visit the Vercel URL: confirm 2,280 training customers and actual model metrics load without CORS errors.
3. Submit a single prediction; confirm probability and SHAP contributions.
4. Upload `data/test (1).csv`; expect 570 valid records. Download the competition CSV and check header `id,prediction`, 570 data rows, padded IDs in sample order, and binary classes.
5. Open Profit Simulator; change a slider and confirm Python-calculated outputs update. Select the uploaded portfolio for campaign allocation.
6. With a configured Gemini key, generate an executive report. If the model is unavailable to your account, choose an accessible model through `GEMINI_MODEL` and restart Render.
7. Repeat navigation on mobile and verify invalid CSV and missing-feature errors remain understandable.

Deployment requires access to your GitHub, Render, and Vercel accounts. The account integrations can be installed/connected to continue publishing from this session; never paste secrets into source code or commit them.

## API endpoints

`GET /api/health`, `/api/overview`, `/api/model/metrics`, `/api/model/importance`, `/api/features`, `/api/data/quality`, `/api/competition/submission`; `POST /api/predict`, `/api/predict-batch`, `/api/finance/analyze`, `/api/ai/analyze`, `/api/ai/report`. Explore generated schemas at `/docs`.

## Primary implementation references

- [CatBoost native SHAP API](https://catboost.ai/docs/en/concepts/python-reference_catboost_get_feature_importance)
- [Official Gemini generate_content API](https://ai.google.dev/api/generate-content)
- [Render monorepo root-directory behavior](https://render.com/docs/monorepo-support)
