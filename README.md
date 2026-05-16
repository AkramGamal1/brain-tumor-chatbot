# Brain Tumor Chatbot

Layperson-facing FastAPI service that wraps the brain tumor MRI classifier
(`E:\projects\brain-tumor-detection`) with safety, literacy, and refusal logic.

This service does not import the ML model — it talks to the model's HTTP API
(`POST /predict`) and translates predictions into plain-language explanations.

> **Architecture change**: the built-in demo UI (`GET /`) has been removed.
> This service is now a pure REST API. The front-end team owns a separate
> application that consumes the API over HTTP. Swagger UI is available at
> `/docs` for interactive testing and contract reference.

---

## What it does

- **`POST /chat`** — text question in, corpus-grounded answer or polite
  refusal out. In-scope topics cover the four tumor classes, MRI basics,
  the patient journey (post-MRI flow, biopsies, second opinions, the care
  team, follow-up imaging, recurrence), educational treatment overviews
  (surgery, radiation, chemo, watchful waiting, clinical trials),
  informational mental health (common emotional responses, finding a
  therapist, support groups), and practical life (work, school, driving,
  fatigue, cognitive changes, finances, telling family, caregivers,
  nutrition).
- **`POST /explain`** — image in, layperson explanation out. Calls the parent
  ML model and translates the prediction into gentle, structured prose with
  the appropriate confidence-band language (and the documented
  glioma↔meningioma override when applicable).
- **`GET /health`** — readiness probe; returns `{status, retriever,
  corpus_pages, model}`.
- **`GET /docs`** — Swagger UI; interactive API explorer and contract
  reference for both teams.
- **`GET /redoc`** — ReDoc; clean read-only API documentation.
- **`GET /openapi.json`** — raw OpenAPI 3.1 schema; use this to generate
  typed clients in any language.

The chatbot **never** diagnoses, recommends specific treatment, or
predicts outcomes — that contract is enforced both in the system prompt
and in a deterministic post-LLM regex backstop. Crisis-language input is
intercepted before any LLM call.

---

## Setup

```powershell
# 1. Create and activate a virtualenv (Python 3.11 or 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install (this pulls sentence-transformers + numpy for semantic
#    retrieval, ~600 MB on disk including PyTorch CPU)
pip install -e ".[dev]"

# 3. Configure — copy the template and fill in your keys
copy .env.example .env
# then edit .env:
#   GOOGLE_API_KEY=...   (required — get free key at https://aistudio.google.com/app/apikey)
#   CORS_ORIGINS=http://localhost:3000   (set to your front-end dev URL)
```

---

## Run

Two services must run in separate terminals for full functionality.
`/chat` only needs the chatbot. `/explain` needs both.

**Terminal 1 — Parent ML classifier** (only needed for `/explain`):
```powershell
cd E:\projects\brain-tumor-detection
python scripts/run_api.py        # binds 0.0.0.0:8000
```

**Terminal 2 — This chatbot API**:
```powershell
cd E:\projects\brain-tumor-chatbot
uvicorn chatbot.api:app --reload --port 8001
```

| URL | Purpose |
|---|---|
| `http://localhost:8001/docs` | Swagger UI — interactive, try-it-out |
| `http://localhost:8001/redoc` | ReDoc — clean read-only reference |
| `http://localhost:8001/openapi.json` | Raw OpenAPI schema for SDK generation |
| `http://localhost:8001/health` | Readiness probe |

### Quick smoke tests (PowerShell)

```powershell
# Health check
curl.exe http://localhost:8001/health

# Chat
curl.exe -X POST http://localhost:8001/chat `
  -H "Content-Type: application/json" `
  -d '{\"message\": \"What is a meningioma, in plain language?\"}'

# Explain (needs parent ML running on :8000)
curl.exe -X POST http://localhost:8001/explain `
  -F "image=@some_mri.png"
```

---

## API contract

### `POST /chat`

**Request** (`application/json`):
```json
{
  "message": "What is a glioma?"
}
```

**Response 200** (`application/json`):
```json
{
  "reply": "A glioma is a type of tumor that starts in the glial cells...",
  "crisis": false
}
```

**Error responses**:
| Status | `error` field | Meaning |
|---|---|---|
| 422 | `validation_error` | Empty or too-long message |
| 503 | `llm_unavailable` | Gemini quota or upstream error |
| 500 | `internal_error` | Unexpected server error |

All errors follow the same shape:
```json
{
  "error": "llm_unavailable",
  "message": "The chatbot has reached its daily request limit. Please try again later.",
  "retry_suggested": true
}
```

---

### `POST /explain`

**Request** (`multipart/form-data`):
| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | Yes | JPEG or PNG brain MRI scan, max 10 MB |
| `force` | bool | No (default: `false`) | Skip MRI heuristic gate — for integration testing only |

**Response 200** (`application/json`):
```json
{
  "predicted_class": "glioma",
  "confidence_band": "fairly certain",
  "explanation": "The model looked at this scan and its best guess is...",
  "override_applied": false
}
```

`confidence_band` values: `"fairly certain"` | `"moderately confident"` | `"uncertain"` | `"suppressed"` (when the glioma↔meningioma override fires).

**Error responses**:
| Status | `error` field | Meaning |
|---|---|---|
| 400 | `not_an_mri` | Image rejected by heuristic gate |
| 422 | `validation_error` | No file uploaded |
| 502 | `ml_unreachable` | Parent ML service on `:8000` is not running |
| 503 | `llm_unavailable` | Gemini quota or upstream error |
| 500 | `internal_error` | Unexpected server error |

---

### `GET /health`

**Response 200**:
```json
{
  "status": "ok",
  "retriever": "EmbeddingRetriever",
  "corpus_pages": 38,
  "model": "gemini-2.5-flash"
}
```

---

## Configuration knobs (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Gemini API key. Get free key at https://aistudio.google.com/app/apikey |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model name. `gemini-2.5-flash-lite` is the working free-tier option. |
| `RETRIEVER` | `embedding` | `embedding` (semantic, top-k=5) or `whole_corpus` (full corpus every call). |
| `RETRIEVER_TOP_K` | `5` | How many chunks the embedding retriever returns. |
| `ML_API_BASE_URL` | `http://localhost:8000` | Parent ML service base URL. |
| `BAND_HIGH` / `BAND_LOW` | `0.85` / `0.65` | Confidence-band thresholds. |
| `GLIOMA_MENINGIOMA_GAP` | `0.20` | Threshold for the glioma↔meningioma override. |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed front-end origins. Set to your front-end URL in production (e.g. `https://yourapp.com`). |
| `SERVE_DEMO_UI` | `false` | Set to `true` to re-enable the legacy demo UI at `GET /`. For local back-end dev only — never enable in production. |

---

## How the front and back teams work together

The service is a pure REST API. Both teams integrate over HTTP — neither
team touches the other's codebase.

### Back-end team owns
`src/chatbot/` (all Python), `corpus/`, `tests/`, `eval/`.

Workflow:
1. Pull latest, activate `.venv`.
2. Run `pytest tests/` before any commit — suite must stay green.
3. For prompt or safety-rule changes, run `python eval/run_eval.py --phase 1`
   against real Gemini. Don't ship prompt changes that drop a gate.
4. For corpus changes, just commit — `EmbeddingRetriever` rebuilds its index
   at next server start.
5. Respect the invariants in [`CLAUDE.md`](CLAUDE.md): class-name canonical
   order, single inference path through `MLClient`, crisis pre-check is
   always step 1, no numeric confidence in user-facing text,
   `notumor != healthy`, stateless service.
6. **API contract changes** (new fields, new endpoints, changed response
   shapes) must be communicated to the front-end team before merging.
   Update this README's API contract section and notify the front team.

### Front-end team owns
Their own separate application (React, Next.js, mobile app, etc.).
They consume this API over HTTP and never touch any Python in this repo.

Workflow:
1. Start the back-end locally:
   ```powershell
   uvicorn chatbot.api:app --reload --port 8001
   ```
2. Open `http://localhost:8001/docs` — use Swagger UI to explore endpoints,
   read request/response schemas, and test calls interactively before
   writing any front-end code.
3. Point your front-end dev server at `http://localhost:8001`. Set
   `CORS_ORIGINS=http://localhost:3000` (or your dev port) in the back-end
   `.env` so CORS allows your origin.
4. Generate a typed API client from the OpenAPI schema if needed:
   ```bash
   # TypeScript / Axios
   npx @openapitools/openapi-generator-cli generate \
     -i http://localhost:8001/openapi.json \
     -g typescript-axios \
     -o ./src/api-client

   # Or use the Fetch-based generator
   npx openapi-typescript http://localhost:8001/openapi.json \
     --output ./src/api-client/schema.d.ts
   ```
5. If a new endpoint or response field is needed, open an issue and tag
   the back-end team. Don't modify Python yourself.

### Integration checklist for the front-end team

| Endpoint | Content-Type | Notes |
|---|---|---|
| `POST /chat` | `application/json` | Send `{"message": "..."}`. Always check `crisis` field — if `true`, display the reply prominently as a safety message. |
| `POST /explain` | `multipart/form-data` | Send `image` as a file field. Optionally send `force=true` for testing. |
| `GET /health` | — | Poll on app load to show a service status indicator. |

**Error handling**: every error response has `{error, message, retry_suggested}`.
Write a single interceptor that reads `retry_suggested` — if `true`, show a
retry button; if `false`, show the `message` as a permanent error.

**CORS**: the back-end sends `Access-Control-Allow-Origin` headers. In
production, the back-end team must set `CORS_ORIGINS` to your deployed
front-end domain. Coordinate this before any production deployment.

---

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

---

## Project structure

```
brain-tumor-chatbot/
├── src/chatbot/                 # FastAPI service (back end)
│   ├── api.py                   # Endpoints: /chat, /explain, /health
│   ├── exceptions.py            # Centralized exception handlers
│   ├── llm.py                   # Gemini SDK wrapper + provider-agnostic errors
│   ├── ml_client.py             # HTTP client to the parent ML on :8000
│   ├── retriever.py             # WholeCorpusRetriever + EmbeddingRetriever
│   ├── prompts.py               # System-prompt builders for /chat and /explain
│   ├── confidence.py            # Probability → verbal band, override logic
│   ├── safety_check.py          # Post-LLM regex backstop + crisis detection
│   ├── corpus.py                # Loads markdown pages into memory at startup
│   ├── disclaimer.py            # Canonical disclaimer + idempotent appender
│   ├── image_check.py           # Heuristic "looks like an MRI" gate
│   └── static/                  # Legacy demo UI (not served unless SERVE_DEMO_UI=true)
│       └── index.html
├── corpus/                      # 38 markdown pages — the chatbot's knowledge base
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

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │   Front-end application          │
                    │   (separate repo / team)         │
                    └──────────────┬──────────────────┘
                                   │  HTTP + JSON / multipart
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│   FastAPI service  (src/chatbot/api.py)  :8001               │
│                                                              │
│   POST /chat                  POST /explain                  │
│     │                          │                             │
│     ▼                          ▼                             │
│   crisis pre-check           image_check (or force=true)     │
│     │ (matches → canned)      │                              │
│     ▼                          ▼                             │
│     │                        ml_client.predict ── HTTP ───▶ Parent ML :8000
│     │                          │                             │
│     ▼                          ▼                             │
│   retriever.retrieve         retriever.retrieve              │
│     │                          │                             │
│     ▼                          ▼                             │
│   prompts.build_chat_…       prompts.build_explain_…         │
│     │                          │                             │
│     ▼                          ▼                             │
│   llm.complete  ──── HTTPS ──▶ Google Gemini API             │
│     │                          │                             │
│     ▼                          ▼                             │
│   safety_check.scan          safety_check.scan               │
│     │                          │                             │
│     ▼                          ▼                             │
│   disclaimer append          disclaimer append               │
│     │                          │                             │
│     ▼                          ▼                             │
│   JSON response              JSON response                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `POST /explain` returns 502 | Parent ML on `:8000` is not running | Start `python scripts/run_api.py` in the brain-tumor-detection repo. `/chat` does not need this. |
| CORS error in browser | `CORS_ORIGINS` doesn't include your front-end origin | Add your front-end URL to `CORS_ORIGINS` in `.env` (comma-separated). |
| `Daily limit reached` error | Gemini free-tier daily quota exhausted | Wait for midnight US Pacific reset, or add a paid credit card. Quotas at https://ai.dev/rate-limit. |
| `Service temporarily unavailable` | Gemini upstream 503 | Retry in a moment; the chatbot surfaces a clean `retry_suggested: true` response. |
| First server start hangs ~30 s | `EmbeddingRetriever` downloading `all-MiniLM-L6-v2` (~80 MB). Subsequent starts are fast. | Wait. To skip, set `RETRIEVER=whole_corpus` in `.env`. |
| `/health` reports unexpected `model` | Shell env var overriding `.env` | Run `Remove-Item Env:GEMINI_MODEL` in PowerShell, or set the correct value there. |
| `GET /` returns 404 | Demo UI is disabled (correct behavior in API mode) | Use `/docs` for interactive testing. To re-enable locally, set `SERVE_DEMO_UI=true` in `.env`. |

---

## Safety contract

This service:

- Never gives diagnostic, treatment, or prognosis advice for a specific person.
- Never claims to "rule out" any condition — the model only knows four classes.
- Surfaces the model's documented weaknesses (overconfidence,
  glioma↔meningioma confusion) instead of hiding them.
- Detects crisis-language input and short-circuits to crisis resources
  before any ML or LLM call.
- Catches forbidden patterns in the LLM's output via a deterministic
  regex backstop and substitutes a safe canned response.

The full set of cross-cutting invariants is in [`CLAUDE.md`](CLAUDE.md).
The implementation plan is in [`docs/chatbot-plan.md`](docs/chatbot-plan.md).
Front-end integration notes are in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

*This is not medical advice. Consult a qualified clinician for any medical decisions.*
