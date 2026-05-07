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
