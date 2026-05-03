# Brain Tumor Chatbot

Layperson-facing FastAPI service that wraps the brain tumor MRI classifier
(`E:\projects\brain-tumor-detection`) with safety, literacy, and refusal logic.

This service does not import the ML model — it talks to the model's HTTP API
(`POST /predict`) and translates predictions into plain-language explanations.

Two endpoints:
- `POST /explain` — image in, layperson explanation out.
- `POST /chat` — text question in, in-scope answer or polite refusal out (Phase 2+).

The full implementation plan is in [`docs/chatbot-plan.md`](docs/chatbot-plan.md).
Cross-cutting invariants are in [`CLAUDE.md`](CLAUDE.md).

## Setup

```powershell
# 1. Create and activate a virtualenv (Python 3.11 or 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install
pip install -e ".[dev]"

# 3. Configure
copy .env.example .env
# then edit .env to fill in GROQ_API_KEY
# (free key: https://console.groq.com/keys)
```

## Run

```powershell
# Start the parent ML API in the other repo:
#   cd ..\brain-tumor-detection
#   python scripts/run_api.py            # binds 0.0.0.0:8000

# Start this chatbot:
uvicorn chatbot.api:app --reload --port 8001
```

Swagger UI: http://localhost:8001/docs

## Quick smoke

```powershell
# Image explanation
curl.exe -F "image=@some_mri.png" http://localhost:8001/explain
```

## Eval

Phase 1 eval (synthetic prediction fixtures, no real ML model needed):

```powershell
python eval/run_eval.py --phase 1
```

Hard gates and A→B upgrade triggers are documented in `eval/README.md`.

## Safety

This service:
- Never gives diagnostic, treatment, or prognosis advice.
- Never claims to "rule out" any condition — it only knows about four classes.
- Surfaces the model's documented weaknesses (overconfidence, glioma↔meningioma
  confusion) instead of hiding them.
- Detects crisis-language input and short-circuits to crisis resources before any
  ML or LLM call.

This is not medical advice. Consult a qualified clinician for any medical decisions.
