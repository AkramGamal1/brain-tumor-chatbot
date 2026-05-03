# HANDOFF — Brain Tumor Chatbot, end of Phase 1 work session

## TL;DR

Phase 1 is **built but not committed**. All 22 smoke tests pass. Six of the
seven Phase 1 hard gates can be exercised right now without an LLM. Gates
1–3 (and part of gate 4) require real LLM output, which the project cannot
yet produce because no LLM provider has a working key in `.env`. Three
providers were attempted in this session (Anthropic → Gemini → Groq); the
service is currently wired to Groq and a free key is the missing piece.

**Do not commit Phase 1 until all seven gates pass on a clean
`python eval/run_eval.py --phase 1` run.**

---

## What's done

- **Repo bootstrap.** `pyproject.toml`, `.env.example`, `.gitignore`,
  `README.md`, `CLAUDE.md`, virtualenv at `.venv`, package installed
  editable.
- **Plan persisted.** `docs/chatbot-plan.md` is the full implementation plan
  with the Phase-1-mid-flight LLM-provider addendum at the bottom.
- **Corpus.** All 12 markdown pages in `corpus/` (4 class pages, MRI basics,
  model capabilities, what-this-model-cannot-tell-you, seek-care,
  confidence-meaning, emotional-impact, crisis-resources,
  questions-to-ask-your-doctor).
- **Source modules.** `src/chatbot/` has `__init__.py`, `disclaimer.py`,
  `confidence.py`, `image_check.py`, `corpus.py`, `ml_client.py`,
  `safety_check.py`, `llm.py`, `prompts.py`, `api.py`. `retriever.py` is
  intentionally deferred to Phase 2 per the plan.
- **Eval harness.** `eval/run_eval.py` (Phase 1 mode), 5 prediction JSON
  fixtures, 3 simulated ML-failure cases (mocked exceptions),
  `eval/fixtures/_make_images.py` plus the 4 generated PNG/JPG image
  fixtures. The lifespan-context bug in the harness was found and fixed
  (`ASGITransport` does not forward FastAPI lifespan; the eval now drives
  it explicitly via `app.router.lifespan_context`).
- **Smoke tests.** `tests/test_smoke.py` — 22 cases covering class-name
  invariant, disclaimer idempotence, confidence band thresholds,
  glioma↔meningioma override boundaries, image-check pass/reject,
  safety-pattern hits, crisis substitution, and corpus loading. Currently
  22/22 PASS.
- **End-to-end behaviors verified via real HTTP:**
  - `/health` returns `{"status":"ok"}`.
  - `/explain` with a color image → 200 `image_warning` + `force_available:true`.
  - `/explain` with extreme-aspect → 200 `image_warning`.
  - `/explain` with valid grayscale and parent ML offline → 502
    `ml_service_unavailable` with `retry_suggested: true`. No LLM call
    occurred (verified by the `CountingLLMClient` wrapper used in the eval).

## What's blocking

Phase 1 eval cannot complete end-to-end because no LLM provider is reachable.
Three providers attempted this session:

| Provider | Outcome |
|---|---|
| Anthropic Claude Haiku 4.5 | Auth OK; 400 `Your credit balance is too low to access the Anthropic API`. No billing path. |
| Google Gemini (`gemini-2.0-flash`, then `gemini-1.5-flash`) | Auth OK on a fresh AI Studio key; 429 `RESOURCE_EXHAUSTED` with `limit: 0` on free-tier per-minute and per-day quotas. Same result on both models → project- or account-level provisioning issue, not model-specific. |
| **Groq (`llama-3.3-70b-versatile`)** | **Wired in, awaiting `GROQ_API_KEY` in `.env`. Not yet exercised against the API.** |

The current code is wired for Groq. To unblock: add a free key from
https://console.groq.com/keys to the `GROQ_API_KEY=` line in `.env` and
re-run the eval.

## What's testable right now without an LLM key

Six of the seven gates can be exercised deterministically. The smoke test
suite covers the unit-level invariants for these:

| Gate | Source of truth | Status |
|---|---|---|
| 5 — image_check rejects color and extreme aspect, passes grayscale | `tests/test_smoke.py` (`test_image_check_*`) | PASS |
| 6 — `force=true` override skips image check | exercised by the eval's image-fixtures section + verified manually via curl | wired and verified manually |
| 7 — ML failures return structured errors with no LLM call | `eval/run_eval.py` mocks the three failure modes; manual curl confirms the unavailable path | wired |
| 4 (partial) — `append_disclaimer` is idempotent and adds the canonical text | `tests/test_smoke.py` (`test_append_disclaimer_*`) | PASS |
| Safety regex — diagnostic, treatment, ruling-out, percent, decimal-near-confidence | `tests/test_smoke.py` (`test_safety_blocks_*`) | PASS |
| Crisis substitution + crisis detection | `tests/test_smoke.py` (`test_crisis_*`, `test_contains_crisis_language`) | PASS |
| Corpus loads with `crisis_response_text` extracted | `tests/test_smoke.py` (`test_corpus_*`) | PASS |

Gates 1, 2, and 3 (notumor rule wording, glioma↔meningioma rule wording, no
numeric confidence in any rendered explanation) require real LLM output to
inspect — the eval harness is the only thing that exercises them.

## Provider history (this session)

Three swaps, each strictly scoped to `src/chatbot/llm.py`, the `LLMClient`
construction in `src/chatbot/api.py`, the env-var name, the eval-harness
precheck, `.env`/`.env.example`, and surfacing docs. The `Retriever`
indirection, system-prompt structure, post-generation `safety_check`,
disclaimer pipeline, endpoint flow, and the seven Phase 1 hard gates were
**unchanged across all three** — the safety architecture is provider-
agnostic by construction.

1. **Anthropic** (planned baseline, model `claude-haiku-4-5-20251001`,
   `anthropic` SDK, `cache_control` ephemeral block on the corpus).
2. **Google Gemini** (`gemini-2.0-flash`, briefly via `google-generativeai`
   then immediately migrated to `google-genai` because the older SDK is in
   end-of-support). Three system blocks were concatenated into a single
   `system_instruction` since Gemini's free tier has no caching equivalent.
3. **Groq** (current, `llama-3.3-70b-versatile`, `groq` SDK,
   OpenAI-compatible chat completions). Three system blocks concatenated
   into one system message in the messages list. `cache_control` field on
   the corpus block remains in the prompt builder (silently ignored) so a
   future swap back to Claude is still a one-file change.

The full sequence is captured in the addendum at the bottom of
`docs/chatbot-plan.md`.

## Files that ground future sessions

Read these three before doing anything else:

- `docs/chatbot-plan.md` — full implementation plan (Phases 1, 2, 3) plus
  the LLM-swap addendum.
- `reference/CLAUDE.md` and `reference/README.md` — snapshots of the
  parent ML repo's docs (the `/predict` contract, the four-class canonical
  order, documented overconfidence, glioma↔meningioma confusion). **Never
  modify these.**
- This `HANDOFF.md` — current state.

`CLAUDE.md` (chatbot-side) lists the cross-cutting invariants — class-name
order, single inference path through `MLClient`, fixed endpoint flow with
crisis pre-check as step 1, no numeric confidence in user-facing text,
`Retriever` indirection rule, stateless service, current LLM provider, and
the provider-agnostic-by-design clause.

## What to do next

1. **Get Groq running.** Generate a key at https://console.groq.com/keys,
   paste it into the `GROQ_API_KEY=` line of `.env`. No code change needed.
2. **Run the eval:** `.venv\Scripts\python.exe eval\run_eval.py --phase 1`.
   Expect: a scorecard table of seven gates, each PASS or FAIL with
   per-case reasons; exit 0 if all pass, 1 if any fail.
3. **If gates 1 or 2 fail on exact-phrase compliance** (most likely failure
   mode under a new model — Llama 3.3 may phrase the notumor or
   glioma↔meningioma language differently from how the regex scorer
   expects), tighten the system prompt in `src/chatbot/prompts.py` to
   require the specific anchor phrases the scorer looks for. The scorer
   patterns live in `eval/run_eval.py` (`NOTUMOR_*` and `OVERLAP_*`
   constants near the top of the scoring section). Iterate until green.
   **Do not weaken the scorer to make tests pass** — tighten the prompt.
4. **Field-name verification** (one-time): start the parent ML API
   (`E:\projects\brain-tumor-detection`, `python scripts/run_api.py`),
   open http://localhost:8000/docs, and confirm `/predict`'s multipart
   field name. The chatbot assumes `file` (FastAPI convention). If it's
   different, update `src/chatbot/ml_client.py` line ~55 (the `files = {...}`
   dict) and re-run the eval.
5. **Write `docs/INTEGRATION.md`** for the front-end team. Required
   sections per the deferred TODO from this session: the two endpoints
   (`/explain`, eventually `/chat`) with method, content type, request
   and response shapes; the four `/explain` response shapes (success,
   `image_warning`, `ml_service_*` errors); the `force=true` override
   flow with a curl example; the canonical disclaimer text; a CORS note
   that `*` is currently allowed and must be locked down for production.
   Keep under 200 lines.
6. **Commit Phase 1.** Single commit. Message body must explicitly note
   the two non-trivial fixes that landed during Phase 1:
   - the `eval/run_eval.py` lifespan-context fix (without it, `state`
     wasn't populated and `KeyError: 'llm_client'` crashed the eval),
   - the Anthropic → Gemini → Groq provider swap, with the rationale
     (no Anthropic billing, Gemini `limit: 0` provisioning issue).
   Record the seven-gate scorecard in the message body.
7. **Then proceed to Phase 2** per the plan: `Retriever` ABC +
   `WholeCorpusRetriever` + `EmbeddingRetriever` stub (commit 2.1),
   `/chat` endpoint (commit 2.2), full categorized eval harness with all
   seven prompt categories (commit 2.3).

## Sanity-check commands for the next session

Run these three after pulling the repo to confirm nothing rotted:

```powershell
# 1. Unit suite — should be 22/22 PASS in <1s
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v

# 2. App imports cleanly and has the expected routes
.\.venv\Scripts\python.exe -c "from chatbot.api import app; print([r.path for r in app.routes])"

# 3. Eval harness — exits 2 with a clear `GROQ_API_KEY not set` message if
#    the key isn't in .env yet; otherwise prints the seven-gate scorecard.
.\.venv\Scripts\python.exe eval\run_eval.py --phase 1
```

## Hygiene note (one-time, off the critical path)

Three LLM-provider keys passed through the assistant's context window
during this session when `.env` was read for swap-related edits:

- One Anthropic key (now zero-balance, harmless but inert),
- Two Google Gemini keys (created during the Gemini provisioning debug).

None are required by the project anymore. Recommend revoking them at
https://console.anthropic.com/settings/keys and
https://aistudio.google.com/app/apikey for cleanliness. The Groq key, once
you create it, will go through the same path the next time `.env` is
read — that is unavoidable for any provider that uses an env-var-loaded
key, and Groq's free-tier keys are easy to rotate.
