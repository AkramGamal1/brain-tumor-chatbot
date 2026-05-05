# HANDOFF — Brain Tumor Chatbot, end of Phase 1 work session

## TL;DR

Phase 1 is **built but not committed**. All 22 smoke tests still pass. The
Groq key is in `.env` and the eval has been run once end-to-end with real
LLM calls. Result: **5 of 7 gates green; 2 gates (notumor rule, overlap
rule) blocked by a `safety_check` false positive that has now been fixed
on disk but not yet re-verified under load** because we exhausted the
Groq free-tier daily token budget (TPD: 100k) iterating today. The fix
is small and the smoke suite confirms it preserves all the dangerous
phrasings the safety check is supposed to block; the diagnostic capture
showed the LLM's actual output already contains the required gate-1
language. Re-run the eval ~24h from the cap-hit time and gates 1–2 are
expected to flip green.

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
- **First end-to-end eval against Groq + Llama 3.3 70B** (this session):
  exit 1 with the scorecard below. Two issues surfaced and were
  resolved in code; one re-run remains:
  - Gates 3, 4, 5, 6, 7 PASS clean.
  - Gates 1 and 2 FAIL because `safety_check` was over-firing on benign
    LLM output and substituting the canned "I caught myself…" reply,
    which then lacked the gate-1 / gate-2 anchor phrases.
  - Two on-disk fixes landed (see "Today's fixes" below) and the smoke
    suite is still 22/22.

## Today's fixes (uncommitted, working tree)

Two files modified, both expected to land in the Phase 1 commit:

**`src/chatbot/safety_check.py` — false-positive narrowing.**

Two patterns in the forbidden-pattern set were too broad and were killing
otherwise-correct LLM output:

1. `\byou\s+(?:have|are\s+diagnosed\s+with)\b` matched the diagnostic
   phrasing the rule is meant to catch ("you have a glioma") but ALSO
   matched any benign "you have …" — e.g. **"if you have concerns about
   symptoms, talk to a doctor"** — which is gentle empathetic framing,
   not a diagnostic claim. This was the actual trip in the eval: the
   `notumor_high_conf` LLM output contained all three required gate-1
   phrases (`four specific conditions`, `does not mean … healthy`,
   `qualified clinician`), and the safety substitution wiped them.

2. `\brul(?:e|es|ed|ing)\s+out\b` was forbidden in either direction.
   But the corpus + system prompt actively encourage the LLM to say
   "the model **cannot** rule out other conditions" as the
   epistemically-correct framing — that's not a violation, it's the
   point.

The fix:

- Diagnostic pattern now requires a medical-condition noun within the
  same phrase: `you have <a|an|the> (glioma|meningioma|pituitary|tumor|
  cancer|growth|mass|lesion|brain tumor|brain disease|neurological
  condition|...)`. "You have concerns" no longer matches; "You have a
  glioma" still does (the smoke test
  `test_safety_blocks_diagnostic_phrasing` enforces this).
- Rule-out check moved to a function (`_has_unnegated_rule_out`) that
  iterates matches and skips any preceded within 50 chars by a negation
  word (`cannot`, `can't`, `does not`, `doesn't`, `unable to`,
  `no way to`, `never`, `without`, etc.). Smoke test
  `test_safety_blocks_ruling_out` (un-negated form) still trips, and
  the new behavior allows the corpus's standard "the model cannot rule
  out…" phrasing.

The diagnostic intent of `safety_check` is **unchanged** — only its
precision improved. All 22 smoke tests pass after the change.

**`eval/run_eval.py` — Windows console encoding fix.**

The gate-2 failure-reason string contained a `↔` character (U+2194),
which crashes `print` under Windows cp1252 stdout and prevented the
rest of the scorecard from rendering. Replaced with `/` in the reason
text only — the user-visible character in `prompts.py` and the corpus
is unchanged.

## Token budget situation (the reason we didn't re-verify)

Groq's free tier is **100k tokens per rolling 24h window** ("tokens per
day"). Each `/explain` call ships the full corpus in the system prompt
(~6.4k input tokens) and produces ~500 output tokens, so a single
fixture costs ~7k. A full Phase 1 eval (5 prediction fixtures + 3
ml-failure mocks that don't call the LLM + 4 image cases that do) =
roughly 35–50k tokens.

Today consumed: the first failed eval run (~35k), the diagnostic
capture (~7k before the cap fired mid-run), the polling probe and the
post-fix retry (~7k). Total ≈ 50k+ on top of any prior-session usage,
which put us at `Used 96281 / Limit 100000`. Token aging rate ≈ 70/min,
so freeing up enough headroom for another full eval requires waiting
roughly 10–24 hours from the cap-hit time, depending on when within
the 24h window the heavy usage actually landed.

The pragmatic call (from the user, this session): wait for the rolling
window to clear, re-run the eval tomorrow. No architecture change, no
provider swap, no Dev Tier upgrade.

## What's blocking

The Groq free-tier daily token cap is exhausted; re-running the eval
today would just hit `429 tokens-per-day rate_limit_exceeded` again.
The cap is a 24h rolling window, so headroom returns gradually.
**Recommended approach: wait ~24h from the cap-hit time and re-run
`python eval/run_eval.py --phase 1`.** On the next clean run, gates 1
and 2 are expected to flip green based on the diagnostic capture from
this session.

If a faster turnaround is needed, the alternative is a Groq Dev Tier
upgrade (pay-as-you-go, roughly $0.02 per full eval run) at
https://console.groq.com/settings/billing. The provider lock-in
decision was made earlier this session ("don't swap providers again"
— architecture has cleanly absorbed two swaps already). No new
provider work should happen.

### Earlier provider history (resolved, kept for archaeology)

| Provider | Outcome |
|---|---|
| Anthropic Claude Haiku 4.5 | Auth OK; 400 `Your credit balance is too low to access the Anthropic API`. No billing path. |
| Google Gemini (`gemini-2.0-flash`, then `gemini-1.5-flash`) | Auth OK on a fresh AI Studio key; 429 `RESOURCE_EXHAUSTED` with `limit: 0` on free-tier per-minute and per-day quotas. Same result on both models → project- or account-level provisioning issue, not model-specific. |
| **Groq (`llama-3.3-70b-versatile`)** | **Current. Key in `.env`. First end-to-end eval ran today; safety_check fix landed; re-run pending token reset.** |

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

## Pre-integration checklist

Parent ML API setup is **complete on this machine**:

- `outputs/models/best_model.pth` is in place at
  `E:\projects\brain-tumor-detection\outputs\models\best_model.pth`.
- The ML repo's venv is set up; `python scripts/run_api.py` brings the
  service up on `http://localhost:8000` with the model loaded.
- `/predict`'s multipart field name is confirmed as `file` (verified by
  reading `src/api/main.py` directly and against the running server).
  The chatbot's `ml_client.py` matches — no code change needed.

End-to-end testing now has both services available locally, so step 4
of "What to do next" below (field-name verification) is **done**. Phase 2
can call the real ML model from the chatbot when needed.

## What to do next

1. **Re-run the eval ~24h after the cap-hit time:**
   `.venv\Scripts\python.exe eval\run_eval.py --phase 1`. Expect: a
   scorecard table of seven gates, each PASS or FAIL with per-case
   reasons; exit 0 if all pass, 1 if any fail.
   - If a 429 still fires on the first fixture, the rolling window
     hasn't cleared enough yet — wait longer or upgrade to Dev Tier.
   - If gates 1 and 2 now flip to PASS as expected, proceed to step 4.
2. **If gates 1 or 2 still fail on exact-phrase compliance** after the
   safety_check fix is exercised under load, the next likely cause is
   the LLM's wording (e.g. saying `"three types"` instead of `"four
   conditions"`, or mentioning glioma and meningioma in separate
   paragraphs rather than the single-sentence pattern the scorer
   matches). Tighten `src/chatbot/prompts.py` to require the specific
   anchor phrases. The scorer patterns live in `eval/run_eval.py`
   (`NOTUMOR_*` and `OVERLAP_*` constants near line 120). Iterate
   until green. **Do not weaken the scorer to make tests pass** —
   tighten the prompt.
3. **If gates 1 or 2 fail because safety_check fires again** on a
   different benign phrase, the diagnostic loop is: dump raw LLM
   output, find which `_FORBIDDEN_PATTERNS` entry matched, narrow that
   pattern the same way today's two fixes did (require a medical-noun
   context, or function-check for negation). The shape of the fix is
   set; just identify the new offender.
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
6. **Commit Phase 1.** Single commit, only after all seven gates pass
   on a clean eval run. Message body must explicitly note the
   non-trivial fixes that landed during Phase 1:
   - the `eval/run_eval.py` lifespan-context fix (without it, `state`
     wasn't populated and `KeyError: 'llm_client'` crashed the eval),
   - the Anthropic → Gemini → Groq provider swap, with the rationale
     (no Anthropic billing, Gemini `limit: 0` provisioning issue),
   - the `safety_check` false-positive narrowing (the
     `you have <medical noun>` and unnegated-only `rule out` fixes —
     surfaced because the LLM legitimately writes "if you have
     concerns" and the corpus's correct framing is "the model cannot
     rule out…"),
   - the `eval/run_eval.py` Windows-console encoding fix (replaced a
     `↔` in the gate-2 reason string with `/` so cp1252 stdout doesn't
     crash mid-scorecard).
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
#    NOTE: if it's been less than ~24h since the last cap-hit, the first
#    fixture will likely 429 with "tokens per day". Wait or upgrade.
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
