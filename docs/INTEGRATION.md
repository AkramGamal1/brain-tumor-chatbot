# INTEGRATION.md — front-end integration guide

Service: **Brain Tumor Chatbot** (FastAPI). Default base URL during local
development: `http://localhost:8001`. The service wraps the parent ML model
(brain-tumor-detection, default `http://localhost:8000`); both must be
running for `/explain` to work end-to-end. `/chat` does not call the ML
service.

The service is **stateless** — no auth, no sessions, no chat history. Every
call is independent.

A built-in single-file demo UI is served at `GET /` (see §Demo UI below).

## Endpoints

### `GET /health`

Liveness probe. Returns `200 {"status": "ok"}` when the app is up.

### `POST /explain` *(Phase 1 — live)*

Accepts an MRI image, returns a layperson-friendly explanation of what the
ML model predicted.

- **Method:** POST
- **Content-Type:** `multipart/form-data`
- **Form fields:**
  - `image` *(file, required)* — JPEG, PNG, BMP, or TIFF, up to **10 MB**.
  - `force` *(string, optional, default `false`)* — set to `true` to skip the
    client-side image plausibility check (see §`force=true` below).

### `POST /chat` *(Phase 2 — live)*

Accepts a free-text question and returns a corpus-grounded educational
answer.

- **Method:** POST
- **Content-Type:** `application/json`
- **Body:**
  ```json
  { "message": "What is a meningioma, in plain language?" }
  ```
  `message` is required, 1–4000 chars after stripping. Empty messages
  return `400 {"error": "empty_message"}`.

#### `/chat` response shapes

All return `200 OK`. The `safety_substituted` and `reason` fields tell you
which path produced the response.

**1. Standard reply** — corpus-grounded answer with disclaimer appended:

```json
{
  "status": "ok",
  "response": "...layperson answer...\n\nThis is not medical advice. Please consult a qualified clinician for any medical decisions."
}
```

**2. Crisis substitution** — user input contained crisis indicators
(self-harm, suicidal language). The service substitutes a canned
crisis-resources response (988 + crisis-line text) **without an LLM
call** and **without appending the disclaimer**. Render as-is and treat
the visual presentation with care:

```json
{
  "status": "ok",
  "response": "...crisis resources, 988, compassionate acknowledgement...",
  "safety_substituted": true,
  "reason": "crisis"
}
```

**3. Forbidden-pattern substitution** — the LLM produced output that
tripped the post-LLM regex backstop (a diagnostic, treatment, or
prognostic phrase slipped through). The service substitutes a short
canned refusal. As of this writing, the disclaimer is **not** appended on
this path either — see `docs/LIMITATIONS.md` (L1) for the rationale and
the planned fix:

```json
{
  "status": "ok",
  "response": "I caught myself about to give medical advice I'm not qualified to give. Please consult a clinician.",
  "safety_substituted": true
}
```

You can disambiguate (3) from (2) by checking `reason === "crisis"`.

## `/explain` response shapes

There are **four** response shapes. All return JSON. The HTTP status code
distinguishes errors from successes; the `status` / `error` field
distinguishes successes from soft warnings.

### 1. Success — `200 OK`

```json
{
  "status": "ok",
  "prediction": {
    "predicted_class": "glioma",
    "confidence": 0.87,
    "probabilities": {
      "glioma": 0.87, "meningioma": 0.08,
      "notumor": 0.03, "pituitary": 0.02
    }
  },
  "explanation": "...layperson explanation...\n\nThis is not medical advice. Please consult a qualified clinician for any medical decisions."
}
```

The `explanation` string is what you should render to the user. The
disclaimer is **already appended** — do not add another one. The
`prediction` object is included for diagnostic/debug rendering only; do
**not** show raw probabilities or `confidence` to end users (the chatbot is
designed so users never see numeric confidence).

If a post-LLM safety substitution fired, the response also includes
`"safety_substituted": true`. Render `explanation` as-is regardless.

### 2. Soft warning — `200 OK` with `status: "image_warning"`

The uploaded file passed the MIME check but failed the heuristic plausibility
check (it's not grayscale, or has an extreme aspect ratio — i.e. it doesn't
look like an MRI). No ML call was made.

```json
{
  "status": "image_warning",
  "reason": "Image appears to be color, not a grayscale MRI scan.",
  "force_available": true
}
```

Show `reason` to the user, then offer a "Submit anyway" button that re-posts
the same file with `force=true` (see below).

### 3. ML service errors — `502` / `504`

The chatbot reached its own validation step but the parent ML service was
unreachable, slow, or returned an error. Body shape:

```json
{
  "error": "ml_service_unavailable",   // or _timeout, or _error
  "message": "human-readable text suitable to render",
  "retry_suggested": true               // false for 4xx-class upstream errors
}
```

Mapping:

| HTTP | `error`                  | `retry_suggested` | When                                        |
|------|--------------------------|-------------------|---------------------------------------------|
| 502  | `ml_service_unavailable` | `true`            | ML service unreachable (connection refused) |
| 504  | `ml_service_timeout`     | `true`            | ML service did not respond in time          |
| 502  | `ml_service_error`       | `false`           | ML service returned a non-2xx status        |

For `ml_service_error`, the `message` is status-aware: 415 from upstream
suggests JPEG/PNG/BMP/TIFF, 413 mentions the 10 MB limit, others are
generic.

### 4. Client errors — `415` / `413`

Pre-flight rejections by the chatbot itself, before any ML call.

```json
{
  "error": "unsupported_media",   // 415 — bad MIME type
  "message": "Image type 'image/gif' is not supported. Use JPEG, PNG, BMP, or TIFF.",
  "retry_suggested": false
}
```

```json
{
  "error": "payload_too_large",   // 413 — over 10 MB
  "message": "Image exceeds the 10 MB limit.",
  "retry_suggested": false
}
```

## `force=true` override flow

The image plausibility check is a heuristic and will reject some legitimate
MRIs. When the user gets an `image_warning` response with
`force_available: true`, you can offer them a "Submit anyway" affordance
that resubmits the same file with `force=true`.

```bash
curl -X POST http://localhost:8001/explain \
  -F "image=@scan.png" \
  -F "force=true"
```

The MIME type and 10 MB size limits **still apply** with `force=true`; only
the heuristic plausibility check is bypassed.

## Disclaimer text

The canonical disclaimer is appended to every successful `/explain`
response automatically:

> This is not medical advice. Please consult a qualified clinician for any
> medical decisions.

Do not strip it, restate it, or add a second one. If you want to style it
visually, parse it as the trailing paragraph after a blank line.

## Crisis handling

`/chat` runs a crisis pre-check on the incoming message. If it trips, the
service returns the canned crisis response (containing the **988 Suicide &
Crisis Lifeline**) without an LLM call and without the standard
disclaimer. The response carries `safety_substituted: true` and
`reason: "crisis"`. Your UI does not need to detect crisis language
itself; just style the crisis response distinctly when `reason ===
"crisis"`.

## Demo UI

A single-file static demo lives at `src/chatbot/static/index.html` and is
mounted at `GET /` via FastAPI's `StaticFiles`. It exercises both
endpoints (file upload + force toggle for `/explain`, text input for
`/chat`) using vanilla JS, no build step. Useful for end-to-end smoke
testing and as a reference for how to render the response shapes
documented above.

The demo expects the parent ML service on `localhost:8000` for the
`/explain` flow; `/chat` does not.

## CORS

**CORS is not configured yet.** Cross-origin requests from a browser will
be blocked until middleware is added. Before going live, add
`fastapi.middleware.cors.CORSMiddleware` with the front-end's specific
origin allow-listed — do **not** use `allow_origins=["*"]` in production.
This is intentionally a deployment-time decision, not a code default.

## Quick smoke test

```bash
# 1. health
curl http://localhost:8001/health

# 2. valid grayscale MRI
curl -X POST http://localhost:8001/explain -F "image=@scan.png"

# 3. force override on a non-MRI-looking image
curl -X POST http://localhost:8001/explain -F "image=@photo.jpg" -F "force=true"
```
