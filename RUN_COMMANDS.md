# Run, test, publish, and prepare for Render

## 1. Local setup

Use Python 3.11. The current environment is already created on the pendrive.

```bash
cd "/Volumes/Projects/1. Projects/Fake News Detection Using Machine Learning/Project_final"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a fresh clone only, create the local environment file without overwriting an existing one:

```bash
cp -n .env.example .env
```

Edit `.env` locally and set `GNEWS_API_KEY`, `TAVILY_API_KEY`, and `GROQ_API_KEY`.
`GROQ_MODEL=openai/gpt-oss-20b` is the model tested during this review. Never paste keys into source code or commit `.env`.

Start from `Project_final` because the application uses relative template/static paths:

```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Stop with Ctrl+C. Restart after `.env` edits; browser refresh alone does not reload environment variables. Do not start a second process on the same port.

## 2. Verification

From `Project_final`, with `.venv` activated:

```bash
python -B -m unittest discover -s tests -v
python -m pip check
```

Tests use controlled responses and do not spend provider credits. For a live check, use the browser or:

```bash
curl --fail-with-body http://127.0.0.1:8000/verify_realtime \
  -H 'Content-Type: application/json' \
  -d '{"text":"CM of tamilnadu is Vijay"}'
```

A live check can use one Tavily basic-search credit plus GNews and Groq quota. A successful HTTP response is not proof that the claim is true: inspect provider notices, evidence assessments, dates, and original links.

## 3. Push to the existing GitHub repository

Repository: <https://github.com/Tejaramidi0118/Fake-News-Detection>

The pendrive project's top-level directory is not currently a Git repository. `Project_final/Liar_Dataset/.git` belongs to a different historical repo (`rough`). Do not initialize an unrelated history and force-push it, or stage that nested repo as a submodule.

Instead, clone the existing GitHub history into an ignored local publishing checkout, then copy the updated project files into it. These commands do not delete existing GitHub research files or change the nested historical repository.

Run the clone command once; if `.publish-checkout` already exists, inspect it rather than deleting it or rerunning clone over it.

```bash
cd "/Volumes/Projects/1. Projects/Fake News Detection Using Machine Learning"
git clone https://github.com/Tejaramidi0118/Fake-News-Detection.git .publish-checkout

rsync -av \
  --exclude='.git' --exclude='.publish-checkout' --exclude='.local-backups' \
  --include='.env.example' --exclude='.env' --exclude='.env.*' \
  --exclude='.venv' --exclude='venv' --exclude='env' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' \
  --exclude='*.pkl' --exclude='*.pickle' --exclude='*.joblib' \
  --exclude='._*' --exclude='.DS_Store' --exclude='*.log' \
  --exclude='*.pem' --exclude='*.key' --exclude='credentials.json' \
  --exclude='service-account*.json' \
  ./ .publish-checkout/

cd .publish-checkout
git status --short
git add .
git diff --cached --stat
git diff --cached --name-only
git diff --cached --check -- README.md RUN_COMMANDS.md RUNCOMMANDS.md .gitignore requirements.txt Project_final/requirements.txt Project_final/.env.example
```

Review the staged files. Actual `.env` files, `.venv`, model binaries, and nested `.git` metadata must not be included; `.env.example` should be included. Existing root-level research files from GitHub remain in place, alongside the organized `Project_final` copy. Consolidating those duplicates is deferred rather than deleting historical files during this update.

When satisfied:

```bash
git commit -m "Add evidence-first verification with GNews, Tavily and Groq"
git push origin main
```

No force push is required. If GitHub reports a newer remote commit, stop and integrate it before pushing. Git authentication may be required; GitHub passwords are not accepted as Git HTTPS credentials.

For later updates, work in the publishing checkout or repeat the same rsync command after reviewing local changes. Keep `.env` on the local development machine; configure production secrets in Render.

## 4. Proposed Render configuration — one service

Deployment still requires approval and has not been performed.

Create **one Python Web Service** from the existing GitHub repository:

| Setting | Value |
| --- | --- |
| Branch | `main` |
| Root Directory | `Project_final` |
| Build Command | `python -m pip install -r requirements.txt` |
| Start Command | `python -m uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/` |
| Python | `3.11`, selected by `.python-version` |

Set these variables privately in Render's Environment settings:

- `GNEWS_API_KEY`
- `TAVILY_API_KEY`
- `GROQ_API_KEY`
- `GROQ_MODEL=openai/gpt-oss-20b`

Do not upload `.env`, use `--reload` in production, or create a separate Static Site for the frontend. FastAPI serves HTML, CSS, JavaScript, and API routes on the same service/domain. Provider calls remain server-side so keys are not exposed. Browser-only hosting cannot safely retain these shared secrets.

One service removes the need for separate frontend/backend hosting; it cannot eliminate search, scraping, model-request latency, or hosting cold starts. No database is required for the current implementation.

After deployment, check the homepage, styles, `/docs`, and a live `/verify_realtime` request on the public URL. Confirm source links, the answer citations, and provider status messages. Public deployment has not yet been verified.

Official references:
- <https://render.com/docs/deploy-fastapi>
- <https://render.com/docs/python-version>
- <https://docs.tavily.com/documentation/api-credits>

## 5. Optional historical ML work

The production web requirements intentionally omit unused spaCy, sentence-transformers, DuckDuckGo, mnemonic, and ML training libraries. The current live verifier does not use a trained local classifier.

`requirements-research.txt` lists the notebook/training dependencies separately. That optional environment has not been installed or validated during this review. `train_model.py` and the retained `/predict_local` route have known serialization/input compatibility issues; simply running training is not yet a verified way to enable local classification. Training is deferred.

The separate historical Flask demo has its own `Liar_Dataset/Docker_Practise/requirements.txt`. Its model files remain local and are ignored by Git. Its development Dockerfile is not the deployment configuration for the main application.

## 6. Progress record — 2026-11-21

- Inspected the main application, datasets, notebooks, and historical Flask demo.
- Created a Python 3.11 environment; fixed the homepage template compatibility error.
- Replaced unavailable Groq model configuration with an environment-selectable tested model.
- Implemented evidence-first search, provider errors, contextual article assessments, and source links.
- Removed model-controlled clarification that blocked searchable statements.
- Added a responsive direct-answer card with citations and conservative fallbacks.
- Replaced Serper with Tavily basic search; GNews and Groq retained.
- Twenty-one offline regression tests passed; live search and cited answers verified on the user's server.
- Updated runtime dependencies, optional research dependencies, documentation, environment example, and ignore rules for GitHub/single-service hosting.
- User controls GitHub push. No repository history was moved, no server started for this documentation update, and nothing deployed to Render.
