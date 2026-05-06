# HANDOFF — end of Day 1, ready for Day 2 manual eval

## TL;DR

**Day 1 is done.** Commits `3a4c0c0` (2.1) and `5d938a7` (2.2) landed
on `experiment/gemini`. The chatbot has both endpoints (`/explain`,
`/chat`), Phase 1 eval passes 7/7, and `eval/prompts.yaml` is in place
ready for the Day 2 manual scoring run.

**Day 2 is the manual eval.** Hand-paste each of the 15 prompts in
`eval/prompts.yaml` into `POST /chat`, score against the
`pass_criteria` / `fail_signals` for each, and commit a scorecard as
**Commit 2.3**.

**Day 3 is Phase 3 Branch A** — static HTML demo page +
end-to-end test against the real parent ML on `localhost:8000`.

## Commits on this branch (newest first)

| Commit | Title |
|---|---|
| `5d938a7` | **Commit 2.2:** `/chat` endpoint with chat-specific system prompt. |
| `3a4c0c0` | **Commit 2.1:** Retriever + Gemini 2.5 swap + override-text fix + `eval/prompts.yaml`. |
| `605459d` | **Phase 1 close-out:** `/explain` passes all seven hard gates. |

### What landed in `3a4c0c0` (Commit 2.1)

12 files, +571 / −144. Bundle:

- `src/chatbot/retriever.py` (new) — `Retriever` ABC,
  `WholeCorpusRetriever` (live, byte-stable formatting),
  `EmbeddingRetriever` stub (Phase 3 Branch B trigger).
- `src/chatbot/prompts.py` — `build_explain_request` consumes
  `Retriever`. Notumor rule re-framed to be confidence-symmetric
  (literal "does not mean healthy / no disease" sentence is mandatory
  at every band — fix for the high-conf regression observed on
  flash-lite).
- `src/chatbot/api.py` — retriever wired in lifespan; Gemini env
  vars (`GOOGLE_API_KEY`, `GEMINI_MODEL`).
- `src/chatbot/llm.py` — Gemini SDK rewrite via
  `google.genai.Client.aio.models.generate_content`. External
  `complete(system_blocks, user_message) -> str` signature unchanged.
- `src/chatbot/confidence.py` — gate-2 fix:
  `GLIOMA_MENINGIOMA_OVERRIDE_TEXT` rewritten as a directive that
  mandates a verbatim sentence colocating "glioma" and "meningioma"
  within ~17 chars on a single line.
- `tests/test_smoke.py` — 4 retriever tests (suite 22 → 26).
- `eval/run_eval.py` — `--categories` flag for chunked-eval; precheck
  requires `GOOGLE_API_KEY`.
- `eval/prompts.yaml` (new) — 15 manual-eval prompts:
  `in_scope` (3), `adjacent_medical_oos` (5), `crisis` (3),
  `prompt_injection` (4). Each prompt carries `id`, `category`,
  `gate`, `prompt`, `pass_criteria`, `fail_signals`. Crisis prompts
  also carry `triggers_phrase` matching `safety_check._CRISIS_PHRASES`.
- `pyproject.toml` — `groq` removed, `google-genai>=0.8.0,<2.0` added.
- `.env.example` — Gemini-shape placeholder.
- `CLAUDE.md`, `docs/chatbot-plan.md` — invariants and plan addendum
  updated for the Gemini 2.5 swap.

### What landed in `5d938a7` (Commit 2.2)

3 files, +202 / −2:

- `src/chatbot/prompts.py` — `_CHAT_RULES` system prompt with
  explicit in-scope topics, OOS refusal categories, refusal template,
  and a defense-in-depth forbidden list mirroring `/explain`.
  `build_chat_system_prompt(retriever, user_message)` returns two
  system blocks (rules + corpus with `cache_control`).
- `src/chatbot/api.py` — `POST /chat` with `ChatRequest(message)`
  validator. Flow:
  `crisis pre-check → retriever → LLM → safety_scan →
  disclaimer append → return`. Crisis pre-check on user input
  short-circuits before any LLM cost.
- `tests/test_smoke.py` — 3 new tests (suite 26 → 29):
  prompt-builder structure, `_CHAT_RULES` content markers, and an
  ASGI test that verifies `/chat` short-circuits on crisis input
  with **zero LLM calls** (returns `safety_substituted=true` +
  `reason="crisis"` + 988 in body).

Manual smoke at commit time: 3 real `/chat` calls via ASGI
(in-scope glioma, OOS medical, OOS off-topic) — all returned the
expected shapes with disclaimer appended.

## Verification at end of Day 1

- `pytest tests/test_smoke.py -v`: **29/29 PASS**.
- `python eval/run_eval.py --phase 1`: **7/7 gates PASS** on
  `gemini-2.5-flash-lite` (verified post-2.1; 2.2 only added
  `/chat`, did not touch the `/explain` path).

## Day 2 — manual Phase 2 eval (the scorecard run)

**Goal:** score all 15 prompts in `eval/prompts.yaml` against their
`pass_criteria` / `fail_signals` and produce a scorecard.

1. Make sure `.env` has `GEMINI_MODEL=gemini-2.5-flash-lite` and a
   live `GOOGLE_API_KEY`.
2. Start the chatbot:
   `uvicorn chatbot.api:app --reload --port 8001`. Parent ML does
   NOT need to be running — `/chat` does not call it.
3. For each prompt in `eval/prompts.yaml`, POST to
   `http://localhost:8001/chat` with body
   `{"message": "<the prompt text>"}`. Inspect the response.
4. Score against `pass_criteria` (all must hold) and `fail_signals`
   (any one observed = fail). A response that hits any fail signal
   fails even if pass criteria are met.
5. Build a scorecard (markdown or yaml) recording, per prompt:
   `id`, `pass`/`fail`, observed response excerpt, and the
   specific failed criterion or fail signal if applicable.
6. **Commit the scorecard as Commit 2.3.** Suggested message:
   ```
   Commit 2.3: Phase 2 manual eval scorecard.
   15 prompts scored against the four hard gates
   (in_scope_answered, oos_refused, crisis_handled, no_forbidden).
   Failed gates documented for Phase 3's docs/LIMITATIONS.md.
   ```

### Token-budget note for Day 2

Gemini Flash-Lite free tier is **~250 requests/day** for this
account. 15 prompts plus a comfortable buffer for re-runs and
sanity checks fits well inside that. **No automated runner** —
this is manual by design (revisited and locked: see "Compression
decisions" below).

**Important constraint:** do not make any non-eval LLM calls
during the Day 2 session. Save the budget for the 15 prompts and
re-runs. Tonight's session ends before any further LLM calls.

## Day 3 — Phase 3 Branch A

Branch A = single-file static UI mounted via FastAPI's
`StaticFiles`. It is the **graduation deliverable**. Branch B
(`EmbeddingRetriever` + semantic retrieval) is deferred
post-graduation.

Tasks:

1. Add a static HTML demo page (one HTML file, vanilla JS, no
   build step) that hits `/explain` and `/chat`. Mount via
   `app.mount("/", StaticFiles(directory=..., html=True))` (or
   under `/ui` if root collides with anything).
2. Bring up the parent ML on `http://localhost:8000` and run
   real end-to-end tests:
   - Upload a real grayscale MRI through the UI to `/explain`.
   - Try a non-MRI image to verify the `image_warning` flow.
   - Try `force=true` override to verify the bypass.
   - Hit `/chat` from the UI with an in-scope, OOS, and crisis
     message.
3. If any of the Phase 2 gates failed in Day 2's scorecard,
   create `docs/LIMITATIONS.md` and document them. Phase 3 ships
   regardless.
4. Commit Phase 3 work in the natural number of commits — no
   pre-determined message structure.

## LLM provider state

- **Active model:** `gemini-2.5-flash-lite` (set in `.env`).
  Retained for Day 2 evaluation consistency — switching mid-eval
  would invalidate scoring.
- `gemini-2.5-flash`: was 503-UNAVAILABLE through Day 1.
  Re-try periodically. Prompts are tuned to pass on either model.
- `gemini-2.5-pro`: free-tier `limit:0` for this account (not
  provisioned).
- Provider-agnostic interface in `llm.py` preserved — switching
  models is a `.env` change.

## Compression decisions (locked, do not revisit)

- Phase 2 eval is **manual**, not automated. The three-commit
  Phase 2 structure is preserved: 2.1 (retriever + ancillary
  bundle), 2.2 (`/chat`), 2.3 (manual scorecard). The prompts file
  landed inside 2.1, **not** as a separate 2.3a — there is no 2.3a.
- Phase 2 categories trimmed 7 → 4: keep `in_scope`,
  `adjacent_medical_oos`, `crisis`, `prompt_injection`. Dropped
  three (`in_scope_emotional`, `adjacent_admin_oos`, `off_topic`)
  as future work.
- Phase 3 ships **Branch A unconditionally**. Failed Phase 2 gates
  become entries in `docs/LIMITATIONS.md`. Branch B is deferred
  post-graduation.

## Files that ground future sessions

- `docs/chatbot-plan.md` — full implementation plan + provider-swap
  addenda.
- `docs/INTEGRATION.md` — front-end integration guide (Phase 1
  contract).
- `CLAUDE.md` — chatbot-side cross-cutting invariants.
- `eval/prompts.yaml` — 15-prompt manual eval input.
- `reference/CLAUDE.md` and `reference/README.md` — read-only
  snapshots of the parent ML repo's docs. **Never modify these.**
- `git show 5d938a7` — Commit 2.2.
- `git show 3a4c0c0` — Commit 2.1.
- `git show 605459d` — Phase 1 close-out.

## State-of-the-machine reminders

- **Parent ML API:** set up locally on `http://localhost:8000`,
  not required for Day 2's `/chat` work; required for Day 3's
  end-to-end testing. `/predict`'s multipart field is `file`.
- **Branch:** `experiment/gemini`. Last commit `5d938a7`.

## Hygiene note

Carried forward — three keys still pending revocation:

- One Anthropic key (zero-balance, inert).
- Two Google Gemini keys (Gemini 2.0 / 1.5 provisioning debug).
- A fourth Google key briefly leaked into `.env.example` was
  rotated by the user mid-session and is now invalid.

Revoke at:

- `https://console.anthropic.com/settings/keys`
- `https://aistudio.google.com/app/apikey`
