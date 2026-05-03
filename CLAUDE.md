# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

FastAPI service that wraps the brain-tumor-detection ML model
(`E:\projects\brain-tumor-detection`) with a layperson-friendly chat layer. The
ML model is a separate process reached over HTTP at `${ML_API_BASE_URL}/predict`
(default `http://localhost:8000`). This repo never imports from the ML repo and
must not depend on it being installed locally.

The full implementation plan and rationale live in `docs/chatbot-plan.md`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env   # then fill GROQ_API_KEY (https://console.groq.com/keys)
```

## Common commands

```powershell
# Run the chatbot (ML repo must be running on :8000 for /explain)
uvicorn chatbot.api:app --reload --port 8001

# Phase 1 eval — synthetic fixtures, no real ML model
python eval/run_eval.py --phase 1

# Tests
pytest
```

## Cross-cutting invariants — read these before changing things

- **Class names are load-bearing.** `chatbot.CLASS_NAMES = ("glioma",
  "meningioma", "notumor", "pituitary")` is the single source of truth and must
  match the parent ML repo's canonical alphabetical order. `ml_client.py`
  asserts the response's `probabilities` keys match this set on every call. If
  the parent renames a class, update here too.
- **Single inference path.** All calls to the ML model go through
  `chatbot.ml_client.MLClient`. Don't reimplement HTTP-to-`/predict` anywhere
  else.
- **Endpoint flow is fixed.** Every endpoint runs in this order: crisis
  pre-check → (image check, ML call for `/explain`) → LLM call → safety check →
  disclaimer append → return. Crisis pre-check is **always step 1** — it must
  short-circuit before any image work, ML round trip, or LLM cost.
- **No numeric confidence in user-facing text.** The LLM never sees raw
  probabilities. `confidence.derive(...)` translates probabilities into a verbal
  band string (or suppresses the band when the glioma↔meningioma override
  fires). `safety_check` enforces this at the output side as defense in depth.
- **Glioma↔meningioma override.** When the top-2 classes are exactly
  `{glioma, meningioma}` and the gap between their probabilities is `< 0.20`,
  the band is suppressed and the response surfaces the documented confusion.
  Tests in `eval/run_eval.py` enforce this.
- **Notumor is not "healthy".** When `predicted_class == "notumor"`, the
  response must explicitly state that this model only checks for four specific
  conditions and that "no tumor detected" does NOT mean "healthy" — only a
  clinician can assess overall neurological health. Highest-priority safety
  rule.
- **No diagnosis, no treatment, no prognosis.** Forbidden in the system prompt
  and in `safety_check.scan(...)`. A safety-substituted response is returned
  *without* the standard disclaimer.
- **Crisis substitution wins.** If the user input contains crisis indicators
  and the LLM output lacks crisis resources, `safety_check` substitutes the
  canned crisis response from `corpus/crisis-resources.md`. The crisis text is
  cached at startup in `CorpusBundle.crisis_response_text` — substitution path
  must not touch disk.
- **Stateless service.** No persistence, no chat history, no PHI handling. Do
  not add session state, request logging that captures user text at info level,
  or any persistence layer.
- **Retriever indirection (Phase 2+).** All corpus access from prompt builders
  goes through `chatbot.retriever.Retriever`. Phase 1 calls `corpus.load_corpus`
  directly; Phase 2 introduces the ABC and `WholeCorpusRetriever`. Phase 3's
  `EmbeddingRetriever` is the swap point for Option B.
- **LLM provider.** The current model is **Llama 3.3 70B Versatile**
  (`llama-3.3-70b-versatile`), served by **Groq** via the `groq` SDK
  (OpenAI-compatible chat completions). Reason: the project has no
  Anthropic billing and the Gemini free tier was not provisioned on the
  available Google account; Groq's free tier covers the workload with no
  credit card.
- **Provider-agnostic by design.** Switching back to Anthropic Claude or
  Google Gemini (or to any other provider) is a one-file change in
  `src/chatbot/llm.py`. The `LLMClient.complete(system_blocks, user_message)
  -> str` interface, `Retriever` indirection, prompt-block structure,
  post-generation `safety_check`, disclaimer append, and the seven Phase 1
  hard gates are all provider-agnostic. Do not add provider-specific logic
  outside `llm.py`.
- **Prompt caching note.** The Anthropic-style `cache_control` field on
  system blocks (Block 2 carries it for the corpus) is silently ignored by
  the current Groq wrapper — Groq's free tier does not expose prompt
  caching. Leave the `cache_control` field on the block: it costs nothing
  and is the right setting to inherit on a future swap back to Claude.

## Layout

- `src/chatbot/` — service code. Each module has a single responsibility; see
  `docs/chatbot-plan.md` for the dependency map.
- `corpus/` — 12 markdown pages, edited like code. Layperson tone, ≤~400 words.
- `eval/` — synthetic fixtures + scoring harness. `run_eval.py --phase N` is
  the gate enforcer.
- `reference/` — read-only snapshots of the parent ML repo's docs. **Never
  modify.**
- `tests/` — `pytest` smoke tests for non-network paths.

## Future work (not in scope yet)

- Authentication / rate limiting.
- Tightening CORS for production.
- Streaming responses.
- Telemetry beyond stdout logging.
