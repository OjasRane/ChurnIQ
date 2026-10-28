# Final deployment handoff

Deployment audit: 28 October 2026. Existing application code, model artifacts, Render blueprint, and Vercel configuration are unchanged. No local model training was run during this phase.

## Verified

- A real Git repository is initialized on `main` in the project directory. The previous sandbox read-only `.git` restriction was resolved by authorized execution outside the sandbox. No project files were deleted.
- `render.yaml` uses `rootDir: backend`, Python 3.11.15, the existing build command `pip install -r requirements.txt && python train_model.py`, start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and `/api/health`.
- Imported the actual FastAPI entry point without starting model training. Confirmed `/api/health` exists.
- All four CSVs are present under `backend/data/`; discovery selects that directory independently of the working directory. Each copy is byte-identical to its root `data/` original.
- Production build passed with `VITE_API_BASE_URL=https://churniq-deployment-check.invalid`, written to a separate temporary output directory. Confirmed that URL is embedded in the compiled JavaScript. This reserved test address is not a real backend and was not saved to application configuration.
- Vercel configuration remains Vite, `npm run build`, output `dist`, with the existing SPA rewrite.
- Gemini credentials are read only in Python. Frontend references to `GEMINI_API_KEY` are explanatory UI text, not credential access. The backend reads `GEMINI_MODEL` and defaults to `gemini-3.5-flash-lite`.
- Ignore rules cover actual `.env` files, virtual environments, dependencies, artifacts, caches, tool state, deployment credentials, private-key files, and local secret directories. Blank `.env.example` templates remain eligible for commit.
- A heuristic scan of non-ignored candidate files found no recognizable Google/GitHub/AWS credentials or private keys. This does not establish dataset licensing.

## Dataset provenance and attribution

The CSV fields are numeric; inspection found no name, email, phone, address, or other obvious direct-identifier columns. The sequential competition IDs are not verified real customer identities. Numeric/anonymized appearance alone does not establish privacy clearance or redistribution rights.

The project owner has confirmed the original source as **Iranian Churn, UCI Machine Learning Repository (2020), DOI 10.24432/C5JW3Z, CC BY 4.0**. The four competition CSVs are modified derivatives, under the stated assumption that no third-party data was incorporated. README.md records the original source, DOI, license link, modified/anonymized names, added noise, training/test splits, and non-endorsement. Retain that attribution when distributing the data. Do not add private customer data. Keep the GitHub repository private as requested.

GitHub CLI is not installed in the current execution environment, no GitHub token is configured, and GitHub/Render/Vercel integrations are not confirmed connected. No commit, remote repository, push, or live deployment is claimed.

## 1. GitHub: authenticate and create the private repository

With the README dataset attribution retained, use your own terminal in this project directory. On Ubuntu/Debian, if `gh` is missing:

```bash
sudo apt update
sudo apt install gh
```

Then:

```bash
cd /home/ojas/EEFM-Project/ChurnIQ
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status

git status
git add .gitignore README.md VERIFICATION.md DEPLOYMENT.md render.yaml backend frontend scripts data
git diff --cached --stat
git diff --cached --name-only
# Confirm only approved CSVs and source/configuration files are staged.
git commit -m "Publish tested ChurnIQ application"
gh repo create churniq --private --source=. --remote=origin --push

gh repo view --json nameWithOwner,url,visibility
git status --short
```

Repository initialization is already complete on `main`; do not delete or recreate `.git`. Your existing Git author configuration is available. If executing somewhere else, use that checkout's actual path.

If `churniq` already exists in the authenticated account, stop and inspect it; do not overwrite another repository. A successful `gh repo create` prints the actual private repository URL. Check its main branch contains `render.yaml`, `frontend/vercel.json`, and all four approved CSVs in `backend/data/`.

## 2. Render: use the existing Blueprint

1. Sign in at https://dashboard.render.com and select your workspace.
2. Choose **New → Blueprint**. Connect GitHub and grant Render access to the private `churniq` repository.
3. Select `churniq`, branch `main`, and the root **`render.yaml`**. Use the existing blueprint; do not replace its commands or set the blueprint path to `backend/render.yaml`.
4. Review the `churniq-api` Python Web Service. Keep all settings from the file, including `rootDir: backend` and Python 3.11.15.
5. At the secret-variable prompts, set `GEMINI_API_KEY` to your real Gemini key and initially set `CORS_ORIGINS=http://localhost:5173`. After Vercel assigns its URL, replace this origin as described below. To run without AI, omit/leave the Gemini key empty if the dashboard permits; all non-AI features work without it.
6. Apply/create the Blueprint and wait for a healthy deployment. The existing blueprint runs its training command during the remote build; no local retraining or alternative pipeline is introduced.
7. Copy the service's **actual HTTPS URL** from Render. The service name does not guarantee an exact hostname.
8. Open `<RENDER_URL>/api/health`; require HTTP 200, `status: "ok"`, and `model_ready: true`. `ai_available: true` means a key is present; it does not verify provider access.

Render-only variables:

| Name | Value |
| --- | --- |
| `PYTHON_VERSION` | `3.11.15` (already in blueprint) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` (already in blueprint; configurable) |
| `GEMINI_API_KEY` | Your secret Gemini API key |
| `CORS_ORIGINS` | Exact permitted frontend origin(s), comma-separated |

Set secrets in Render, never in `render.yaml`, `.env.example`, GitHub files, or Vercel frontend variables. The model name remains configurable; access to the default must be confirmed by an actual AI request. On model-access errors, choose a model your key can access through Render's `GEMINI_MODEL` environment setting.

## 3. Vercel: deploy the existing frontend

1. Sign in at https://vercel.com, choose **Add New → Project**, and import the private GitHub `churniq` repository. Grant access when requested.
2. Set **Root Directory = `frontend`** and **Framework Preset = Vite**.
3. Keep **Build Command = `npm run build`**, **Output Directory = `dist`**, and set **Install Command = `npm ci`**. Select Node.js 22.x in project settings if the current default differs from your intended runtime.
4. Add **`VITE_API_BASE_URL=<RENDER_URL>`** for **Production**. Use the actual HTTPS backend origin without `/api` or a trailing path. Configure Preview separately only if you intend to test preview deployments.
5. Deploy and copy the actual production **`<VERCEL_URL>`**.
6. For public access, check **Settings → Deployment Protection** and ensure the production URL is accessible to visitors without a Vercel login. Confirm in an incognito window.

Do not add a Gemini key to Vercel. Vite bakes `VITE_API_BASE_URL` into the bundle: changing it requires a new deployment. An absent value falls back to same-origin `/api`, which does not provide the FastAPI backend on this static deployment.

## 4. Connect production origins

1. In Render, open `churniq-api` → **Environment**.
2. Set **`CORS_ORIGINS=<VERCEL_URL>`**, e.g. `https://churniq-yourteam.vercel.app`, using the exact assigned origin. No path and no wildcard are needed. To retain local access, use `<VERCEL_URL>,http://localhost:5173,http://127.0.0.1:5173`.
3. Save and apply the environment change; wait for Render's service deployment/restart to become healthy.
4. Verify Vercel's production build uses the Render origin, not localhost or the reserved audit URL. If you changed the frontend variable, redeploy Vercel.
5. Open the production frontend and inspect browser Developer Tools → Network: `/api/overview`, `/api/model/metrics`, and `/api/features` must target the Render HTTPS origin and return 200 without CORS errors.

## 5. Final smoke test

Replace placeholders with actual URLs:

```bash
curl --fail --show-error '<RENDER_URL>/api/health'
curl --fail --show-error '<RENDER_URL>/api/overview'
curl --fail --show-error --request OPTIONS '<RENDER_URL>/api/predict' \
  --header 'Origin: <VERCEL_URL>' \
  --header 'Access-Control-Request-Method: POST' \
  --header 'Access-Control-Request-Headers: content-type' --include
curl --fail --show-error '<RENDER_URL>/api/competition/submission' \
  --output /tmp/churniq-production-submission.csv
```

Require `Access-Control-Allow-Origin` to equal the actual Vercel origin. Then, in an incognito browser:

1. Open `<VERCEL_URL>` without login. Overview shows 2,280 training records and the measured model metrics.
2. Predict a customer and confirm a probability, risk tier, and signed SHAP contributions.
3. Upload the approved `test (1).csv`; confirm 570 valid rows. Download the competition submission: exactly `id,prediction`, 570 data rows, original sample ID order, binary values.
4. Move a Profit Simulator slider; confirm financial outputs update. Choose the uploaded cohort for campaign allocation.
5. Generate an AI executive summary. A successful report verifies Gemini key/model access. If it fails, inspect Render environment settings and model access; other pages must remain functional.
6. Test the mobile menu, an invalid CSV, and an API retry.

Only declare deployment complete after both public URLs and the browser flow work. Until then the status is **prepared, not publicly deployed**.

References: [Render Blueprint environment variables](https://render.com/docs/blueprint-spec), [Render monorepo root directory](https://render.com/docs/monorepo-support), [Render Python versions](https://render.com/docs/python-version), [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite), [Vercel environment variable redeployment](https://vercel.com/docs/environment-variables/managing-environment-variables).
