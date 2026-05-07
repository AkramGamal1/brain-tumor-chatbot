# Brain Tumor Chatbot

Layperson-facing FastAPI service that wraps the brain tumor MRI classifier
(`E:\projects\brain-tumor-detection`) with safety, literacy, and refusal logic.

This service does not import the ML model — it talks to the model's HTTP API
(`POST /predict`) and translates predictions into plain-language explanations.

## What it does

- **`POST /explain`** — image in, layperson explanation out. Calls the parent
  ML model and translates the prediction into gentle, structured prose with
  the appropriate confidence-band language (and the documented
  glioma↔meningioma override when applicable).
- **`POST /chat`** — text question in, corpus-grounded answer or polite
  refusal out. In-scope topics now cover the four tumor classes, MRI basics,
  the patient journey (post-MRI flow, biopsies, second opinions, the care
  team, follow-up imaging, recurrence), educational treatment overviews
  (surgery, radiation, chemo, watchful waiting, clinical trials),
  informational mental health (common emotional responses, finding a
  therapist, support groups), and practical life (work, school, driving,
  fatigue, cognitive changes, finances, telling family, caregivers,
  nutrition).
- **`GET /`** — single-file demo UI (vanilla JS, no build step) that
  exercises both endpoints with a friendly interface.
- **`GET /health`** — readiness probe; returns `{status, retriever,
  corpus_pages, model}`.

The chatbot **never** diagnoses, recommends specific treatment, or
predicts outcomes — that contract is enforced both in the system prompt
and in a deterministic post-LLM regex backstop. Crisis-language input is
intercepted before any LLM call.

## Setup

```powershell
# 1. Create and activate a virtualenv (Python 3.11 or 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install (this pulls sentence-transformers + numpy for semantic
#    retrieval, ~600 MB on disk including PyTorch CPU)
pip install -e ".[dev]"

# 3. Configure — copy the template and fill in your Gemini API key
#    (free tier; create at https://aistudio.google.com/app/apikey)
copy .env.example .env
# then edit .env to set GOOGLE_API_KEY=...
```

## Run

```powershell
# Optional: start the parent ML API in the other repo (only needed for
# /explain to work end-to-end)
#   cd ..\brain-tumor-detection
#   python scripts/run_api.py            # binds 0.0.0.0:8000

# Start this chatbot:
uvicorn chatbot.api:app --reload --port 8001
```

Then open http://localhost:8001/ in a browser for the demo UI, or hit
the API directly:

```powershell
# Health check
curl.exe http://localhost:8001/health

# Chat
curl.exe -X POST http://localhost:8001/chat `
  -H "Content-Type: application/json" `
  -d '{\"message\": \"What is a meningioma, in plain language?\"}'

# Explain (needs parent ML on :8000)
curl.exe -F "image=@some_mri.png" http://localhost:8001/explain
```

Swagger UI: http://localhost:8001/docs

## Configuration knobs (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Gemini API key. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model name. `gemini-2.5-flash-lite` is the working free-tier option. |
| `RETRIEVER` | `embedding` | `embedding` (semantic, top-k=5) or `whole_corpus` (full corpus every call). |
| `RETRIEVER_TOP_K` | `5` | How many chunks the embedding retriever returns. |
| `ML_API_BASE_URL` | `http://localhost:8000` | Parent ML service. |
| `BAND_HIGH` / `BAND_LOW` | `0.85` / `0.65` | Confidence-band thresholds. |
| `GLIOMA_MENINGIOMA_GAP` | `0.20` | Threshold for the glioma↔meningioma override. |

## Eval

Phase 1 eval (synthetic prediction fixtures, no real ML model needed,
runs against real Gemini):

```powershell
python eval/run_eval.py --phase 1
```

Phase 2 manual eval — 25 prompts across in-scope, OOS, crisis, and
prompt-injection categories. Run the chatbot, then:

```powershell
python eval/_run_chat_responses.py
# → produces eval/_chat_responses.json
# → score by hand against eval/prompts.yaml
# → write eval/scorecard.md
```

The current scorecard is in `eval/scorecard.md`. Known issues are
tracked in `docs/LIMITATIONS.md`.

## Project structure

```
brain-tumor-chatbot/
├── src/chatbot/                 # FastAPI service (back end)
│   ├── api.py                   # Endpoints: /explain, /chat, /health, GET /
│   ├── llm.py                   # Gemini SDK wrapper + provider-agnostic errors
│   ├── ml_client.py             # HTTP client to the parent ML on :8000
│   ├── retriever.py             # WholeCorpusRetriever + EmbeddingRetriever
│   ├── prompts.py               # System-prompt builders for /chat and /explain
│   ├── confidence.py            # Probability → verbal band, override logic
│   ├── safety_check.py          # Post-LLM regex backstop + crisis detection
│   ├── corpus.py                # Loads markdown pages into memory at startup
│   ├── disclaimer.py            # Canonical disclaimer + idempotent appender
│   ├── image_check.py           # Heuristic "looks like an MRI" gate
│   └── static/                  # Front end
│       └── index.html           # Single-file demo UI (vanilla JS, no build)
├── corpus/                      # 38 markdown pages — the chatbot's knowledge
├── eval/                        # Phase 1 + Phase 2 evaluation
│   ├── run_eval.py              # Phase 1 gate enforcer (synthetic fixtures)
│   ├── prompts.yaml             # Phase 2 manual eval inputs (25 prompts)
│   ├── _run_chat_responses.py   # Helper that POSTs each prompt to /chat
│   └── scorecard.md             # Latest manual eval result
├── tests/                       # pytest smoke tests (non-network paths)
├── docs/
│   ├── chatbot-plan.md          # Original implementation plan
│   ├── INTEGRATION.md           # Front-end integration contract
│   └── LIMITATIONS.md           # Known shortfalls + planned fixes
├── reference/                   # READ-ONLY snapshots of the parent ML's docs
├── CLAUDE.md                    # Cross-cutting invariants for back-end work
├── HANDOFF.md                   # Current state-of-the-project notes
└── pyproject.toml
```

## Architecture

```
                       ┌──────────────────────────────┐
                       │   Browser / API client       │
                       │   (or src/chatbot/static/)   │
                       └──────────────┬───────────────┘
                                      │  HTTP
                                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │   FastAPI service  (src/chatbot/api.py)                       │
   │                                                               │
   │   POST /chat                  POST /explain                   │
   │     │                          │                              │
   │     ▼                          ▼                              │
   │   crisis pre-check           image_check (or force=true)      │
   │     │ (matches → canned)      │                               │
   │     ▼                          ▼                              │
   │     │                        ml_client.predict ─── HTTP ───▶ Parent ML on :8000
   │     │                          │                              │
   │     ▼                          ▼                              │
   │   retriever.retrieve(query)  retriever.retrieve(class, force-include)
   │     │                          │                              │
   │     ▼                          ▼                              │
   │   prompts.build_chat_…       prompts.build_explain_…          │
   │     │                          │                              │
   │     ▼                          ▼                              │
   │   llm.complete  ───── HTTPS ──▶ Google Gemini API             │
   │     │                          │                              │
   │     ▼                          ▼                              │
   │   safety_check.scan          safety_check.scan                │
   │     │ (forbidden→sub)          │                              │
   │     ▼                          ▼                              │
   │   disclaimer append          disclaimer append                │
   │     │                          │                              │
   │     ▼                          ▼                              │
   │   JSON response              JSON response                    │
   └──────────────────────────────────────────────────────────────┘
```

Key invariants are in [`CLAUDE.md`](CLAUDE.md). Errors at any step (Gemini
quota, ML unreachable, image rejected) translate into clean JSON 4xx/5xx
responses with `error`, `message`, and `retry_suggested` fields.

## How the front and back team work together

This is a monolith for simplicity, but the two halves are deliberately
decoupled by an HTTP contract and have separate ownership.

**Front team owns** `src/chatbot/static/`. Edits the single `index.html`
(HTML + CSS + vanilla JS, all inline). Consumes the API contract documented
in [`docs/INTEGRATION.md`](docs/INTEGRATION.md). Should never touch Python.
Workflow:
1. Pull latest, start the back end (`uvicorn chatbot.api:app --reload --port 8001`).
2. Edit `index.html`. Refresh the browser — no server restart needed.
3. Test against the live `/chat`, `/explain`, and `/health` endpoints.
4. If a new endpoint or response field is needed, **open an issue** asking
   the back team for it. Don't reach into Python yourself.

**Back team owns** `src/chatbot/` (everything except `static/`), `corpus/`,
`tests/`, `eval/`. Workflow:
1. Pull latest, activate `.venv`.
2. Run `pytest tests/` before any commit (suite must stay green).
3. For prompt or safety-rule changes, run `python eval/run_eval.py --phase 1`
   against real Gemini (Phase 1 gates are LLM-driven). Don't ship prompt
   changes that drop a gate.
4. For corpus changes, just commit — `EmbeddingRetriever` rebuilds its index
   at next server start.
5. Respect the invariants in [`CLAUDE.md`](CLAUDE.md). The most load-bearing
   ones: class-name canonical order, single inference path through
   `MLClient`, crisis pre-check is always step 1, no numeric confidence in
   user-facing text, `notumor != healthy`, stateless service.

**Both teams** read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before
filing a bug — known limitations should not become regressions.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Daily limit reached` panel in the UI | Gemini free-tier daily quota exhausted (~20/day on `gemini-2.5-flash-lite` for trial accounts) | Wait for midnight US Pacific reset, switch to a different account/key, or add a paid credit card to the project. Quotas at https://ai.dev/rate-limit. |
| `Service temporarily unavailable` panel | Gemini upstream 503 | Retry in a moment; the chatbot caught it and surfaced a clean message. |
| `Analysis service unreachable` panel on `/explain` | Parent ML on `:8000` is not running | Start the brain-tumor-detection service in its own repo. `/chat` does not need this. |
| First server start hangs ~30 s | `EmbeddingRetriever` is downloading and loading `sentence-transformers/all-MiniLM-L6-v2` (~80 MB). Subsequent starts are fast. | Wait. To skip embeddings entirely, set `RETRIEVER=whole_corpus` in `.env`. |
| `pytest` runs slow first time, fast after | Same — the test suite triggers the embedding model load once. Cached afterward. | Wait. |
| `/health` reports a `model:` you didn't expect | Shell environment variable is overriding `.env` (load_dotenv defaults to non-override). | `unset GEMINI_MODEL` in your shell, or set the right value there. |

## Safety contract

This service:

- Never gives diagnostic, treatment, or prognosis advice for a specific
  person.
- Never claims to "rule out" any condition — the model only knows four
  classes.
- Surfaces the model's documented weaknesses (overconfidence,
  glioma↔meningioma confusion) instead of hiding them.
- Detects crisis-language input and short-circuits to crisis resources
  before any ML or LLM call.
- Catches forbidden patterns in the LLM's output via a deterministic
  regex backstop and substitutes a safe canned response.

The full set of cross-cutting invariants is in [`CLAUDE.md`](CLAUDE.md).
The implementation plan is in
[`docs/chatbot-plan.md`](docs/chatbot-plan.md). Front-end integration
notes are in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

This is not medical advice. Consult a qualified clinician for any
medical decisions.
