# TruSight AI — Fake News Detection

An evidence-first claim-checking web application, alongside the project's original LIAR and PolitiFact machine-learning experiments.

## Current application

1. Prepare a concise search query from the user's statement.
2. Search GNews and Tavily for relevant sources.
3. Retrieve up to five article excerpts and ask Groq to assess their relationship to the claim.
4. Show supporting, contradicting, mixed, or insufficient evidence with original links, dates when available, and excerpts.
5. Present a direct Groq-written answer with citations when usable evidence exists.

The application summarizes retrieved evidence; it does not guarantee factual truth. Current-event dates, incomplete article extraction, search coverage, and model errors can affect results. A source quote is checked against retrieved text, but its presence alone does not prove the assessment is correct. No invented confidence percentage is displayed.

## Architecture

**One FastAPI application serves both the HTML/CSS/JavaScript frontend and API routes.** Browser requests use the same origin. GNews, Tavily, and Groq keys stay server-side. A separate frontend service, database, JavaScript build, or CORS configuration is not required.

Render should host this as **one Python Web Service**, not a Static Site. A browser-only version would expose the provider API keys. External search, article retrieval, and model calls still take time even in a single service.

## Files

- `Project_final/app.py`: FastAPI routes and HTML/static delivery.
- `Project_final/verifier.py`: query preparation, provider search, article assessment, and cited answers.
- `Project_final/templates/index.html`, `Project_final/static/style.css`: browser UI.
- `Project_final/tests/`: offline regression tests.
- `Project_final/.env.example`: blank environment-variable template.
- `Project_final/requirements.txt`: tested runtime dependency versions.
- `Project_final/requirements-research.txt`: optional, unvalidated historical notebook dependencies.
- `Project_final/Liar_Dataset/`, `Project_final/Politifact_Dataset/`: existing datasets, notebooks, and the separate historical Flask demo.

## Run and publish

See [RUN_COMMANDS.md](RUN_COMMANDS.md) for local setup, tests, history-preserving GitHub commands, and proposed single-service Render settings. [RUNCOMMANDS.md](RUNCOMMANDS.md) points to the same guide.

## API

- `GET /`: web interface.
- `POST /verify_realtime`: JSON `{"text": "Your claim, including date or event context where relevant"}`.
- `GET /docs`: interactive FastAPI API documentation.
- `POST /predict_local`: retained historical local-classifier route; returns 503 without a model. Local training/inference compatibility issues identified during review remain unresolved and this route is not used by the browser.

## Verification status

The live GNews + Tavily + Groq integration and cited answer have been exercised locally. Twenty-one offline regression tests cover evidence handling, provider errors, clarification, answer citations, and Tavily configuration. The browser answer box was checked at desktop and mobile widths. No Render deployment has been performed or verified.

Original research notebooks and datasets are preserved. Their saved evaluation metrics are historical and are not accuracy claims for the live verifier. Training is optional and deferred.
