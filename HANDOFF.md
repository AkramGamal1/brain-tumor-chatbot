# HANDOFF — Branch B done, only verification remains

## TL;DR

**The chatbot is feature-complete for graduation.** Phases 1, 2, and 3A
all landed earlier; Branch B (semantic retrieval + corpus expansion +
expanded scope + L1/L2 fixes) and a UI polish pass landed today. Local
tests are 37/37 passing and retrieval spot-checks at 10/12 hits in
top-5.

**One small thing left, blocked on quota:** Phase 1 + Phase 2 v2 eval
reruns and the `/explain` end-to-end test through the UI. Free-tier
Gemini quota (20 req/day on `gemini-2.5-flash-lite` for this account)
was exhausted today by Branch B development. Resume after midnight US
Pacific.

## Commits since the last HANDOFF (newest first)

| Commit | Title |
|---|---|
| `d50bcde` | UI polish + LLM error handling + `/health` readiness fields. |
| `57ab233` | Phase 2 eval refresh: +10 prompts for expanded scope and L1/L2 regression. |
| `5bba71a` | Loosen prompt + `safety_check` for expanded scope; bundle L1/L2 fixes. |
| `cc53896` | Corpus expansion: 12 → 38 pages covering full patient education ground. |
| `5fe1a9e` | Branch B mechanics: `EmbeddingRetriever` (sentence-transformers + numpy). |
| `40e4d9d` | **Phase 3 Branch A:** static demo UI mounted at `/`. |
| `1458881` | `docs/LIMITATIONS.md` — three Phase 2 fails documented. |
| `1cc91ca` | **Commit 2.3:** Phase 2 manual eval scorecard (12/15). |

## Branch B summary

- **Retrieval:** `EmbeddingRetriever` runs locally on
  `sentence-transformers/all-MiniLM-L6-v2` (~80 MB on disk, CPU only,
  ~30 s first-run model load, ~30 ms per query thereafter). Numpy
  in-memory cosine similarity over page-level chunks, top-k = 5,
  always-include scaffolding for `/explain`. Switchable via `RETRIEVER`
  env var (`embedding` default, `whole_corpus` fallback).
- **Corpus:** 12 → 38 pages. New categories: mental health
  (informational), patient journey, educational treatment overviews,
  practical life, tumor-type depth, caregivers, nutrition. Each new
  page closes with vetted external links (NCI, NHS, Mayo, NAMI, ACS,
  ABTA, CancerCare, ClinicalTrials.gov) and a "general info, not
  specific advice for your situation" guard sentence.
- **Scope:** `_CHAT_RULES` now allows informational mental-health and
  treatment content (e.g. "doctors often consider X for Y") while
  still refusing prescriptive advice ("you should get surgery"),
  user-targeted diagnosis (medical OR mental-health), and prognosis.
  `safety_check._TREATMENT_PATTERNS` narrowed to user-targeted
  recommendation forms only — generic third-person education passes.
- **L1 fix:** safety substitution now appends the disclaimer when
  `safety.replacement == SAFETY_REPLACEMENT`. Crisis path
  (`crisis_response_text`) deliberately stays disclaimer-free.
- **L2 fix:** new "Refusal style — gentle, short, warm when warranted"
  section in `_CHAT_RULES` instructs a one-sentence warm
  acknowledgement before the redirect on frightened-user prompts.
- **Eval refresh:** 15 → 25 prompts. Original 15 preserved verbatim;
  10 new prompts cover mental-health-in-scope, patient-journey,
  treatment-overview, practical-life, plus L1/L2 regression. New
  `mental_health_oos` category for user-targeted diagnostic requests.
- **UI polish:** scope banner, suggestion chips, friendly error labels
  with retry hints, copy-to-clipboard on responses, mobile spacing,
  `/health` footer info.
- **Error handling:** Gemini SDK exceptions now translate to clean
  JSON 503 with `error: "llm_rate_limited" / "llm_unavailable" /
  "llm_service_error"` and `retry_suggested: true`. Fixes the
  "non-JSON response" the UI was showing when quota hit.

## To-do tomorrow (when quota resets)

```powershell
.\.venv\Scripts\Activate.ps1

# 1. Phase 1 eval — 7/7 gates under EmbeddingRetriever + 38-page corpus
python eval/run_eval.py --phase 1

# 2. Phase 2 v2 — 25 prompts, score against pass_criteria/fail_signals
uvicorn chatbot.api:app --port 8001
python eval/_run_chat_responses.py
# → manually score responses in eval/_chat_responses.json against
#   eval/prompts.yaml, write a new eval/scorecard.md (overwriting v1
#   or saving as scorecard-v2.md). Pay particular attention to:
#   - mental_health_oos_01 (user-targeted refused, informational allowed)
#   - adjacent_medical_oos_06_frightened (L2 fix — warm acknowledgement)
#   - prompt_injection_05_disclaimer_bypass (L1 fix — disclaimer present)
#   - the original adjacent_medical_oos_03 (was the L2 failure case)

# 3. /explain end-to-end through the UI
# Start the parent ML on localhost:8000 first.
uvicorn chatbot.api:app --port 8001     # in this repo
# In a browser: http://localhost:8001/
# Upload a real grayscale MRI; verify success path renders explanation
# with disclaimer styled as a trailing italic block. Then upload a
# color image to see the image_warning + force=true flow.
```

## Out of scope (deliberately deferred)

- **Multi-turn chat history.** Adding state breaks the
  "stateless service" CLAUDE.md invariant and introduces session
  storage / PHI handling concerns. ~50–100 lines if you change your
  mind, but it's a deliberate liability decision to keep it stateless.
- **Fine-tuning on user prompts.** Free-tier Gemini doesn't expose
  fine-tuning, and training on user inputs in a medical chatbot is a
  privacy minefield. The right "learning" mechanism for this system is
  corpus expansion — the EmbeddingRetriever picks up new pages
  automatically at next startup.
- **Phase 3 Branch B as originally framed (semantic retrieval = Branch
  B).** Already shipped — what was originally "deferred" Branch B
  landed in `5fe1a9e`. There is no longer a deferred Branch B.

## Files that ground future sessions

- `CLAUDE.md` — chatbot-side cross-cutting invariants. **Updated for
  Branch B** would be nice if you have time; current invariants still
  hold but `WholeCorpusRetriever live; EmbeddingRetriever stub` is
  now stale.
- `docs/chatbot-plan.md` — the original implementation plan.
- `docs/INTEGRATION.md` — front-end integration guide. Last refreshed
  in Branch A; should be re-read after Branch B to catch any /chat
  shape drift.
- `docs/LIMITATIONS.md` — L1 and L2 marked "FIX LANDED, eval pending".
  Update to "FIX LANDED, eval verified" once tomorrow's run confirms.
- `eval/prompts.yaml` — 25-prompt manual eval input.
- `eval/scorecard.md` — v1 scorecard against the 12-page corpus.
  Refresh / replace tomorrow.
- `reference/CLAUDE.md`, `reference/README.md` — read-only snapshots
  of the parent ML repo's docs. **Never modify.**

## State-of-the-machine reminders

- **Branch:** `experiment/gemini`. Last commit `d50bcde`.
- **Active model:** `.env` says `gemini-2.5-flash-lite`. If `/health`
  reports a different model, a shell environment variable is
  overriding the `.env` (load_dotenv defaults to non-override).
- **Free-tier quota:** ~20 req/day on `gemini-2.5-flash-lite`. Plan
  the Phase 2 v2 run carefully — 25 prompts plus a couple re-runs is
  exactly at the budget edge.
- **Parent ML API:** local on `http://localhost:8000`, NOT required
  for `/chat`; required for `/explain` end-to-end. `/predict`'s
  multipart field is `file`.
- **Static UI:** mounted at `GET /`. Source at
  `src/chatbot/static/index.html`. Edits are picked up on page reload
  (no server restart needed for HTML changes).

## Hygiene note

Carried forward — three keys still pending revocation. Nothing has
changed since the prior HANDOFF on this point. Revoke at:

- `https://console.anthropic.com/settings/keys`
- `https://aistudio.google.com/app/apikey`
