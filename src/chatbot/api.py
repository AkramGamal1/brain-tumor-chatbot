# ─── src/chatbot/api.py ───────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from chatbot.exceptions import register_exception_handlers
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── internal imports ───────────────────────────────────────────────────────────
from chatbot.corpus import load_corpus
from chatbot.retriever import WholeCorpusRetriever, EmbeddingRetriever
from chatbot.llm import LLMClient, LLMServiceError
from chatbot.ml_client import MLClient, MLClientError
from chatbot.prompts import build_chat_system_prompt, build_explain_request
from chatbot.confidence import derive_confidence
from chatbot.safety_check import scan, contains_crisis_language
from chatbot.disclaimer import append_disclaimer
from chatbot.image_check import looks_like_mri

load_dotenv()

# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brain Tumor Chatbot API",
    description=(
        "Layperson-facing service that wraps a brain-tumor MRI classifier "
        "with safety, medical-literacy, and refusal logic.\n\n"
        "**Safety contract**: this service never diagnoses, never recommends "
        "treatment, and never predicts outcomes for a specific person. "
        "Crisis language is intercepted before any ML or LLM call."
    ),
    version="1.0.0",
    contact={"name": "Back-end team"},
    license_info={"name": "Private"},
    openapi_tags=[
        {"name": "Chat", "description": "Text-based Q&A grounded in the tumor corpus."},
        {"name": "Explain", "description": "MRI image → layperson explanation via the ML classifier."},
        {"name": "System", "description": "Health and readiness probes."},
    ],
)

register_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    corpus_dir = Path(__file__).parent.parent.parent / "corpus"
    corpus = load_corpus(corpus_dir)
    retriever_type = os.getenv("RETRIEVER", "embedding").lower()
    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    if retriever_type == "embedding":
        retriever = EmbeddingRetriever(corpus, top_k=top_k)
        retriever.build()
    else:
        retriever = WholeCorpusRetriever(corpus)
    app.state.retriever = retriever
    app.state.ml_client = MLClient(base_url=os.getenv("ML_API_BASE_URL", "http://localhost:8000"))
    app.state.llm = LLMClient(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    app.state.crisis_text = corpus.crisis_response_text


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["What is a meningioma, in plain language?"],
        description=(
            "The user's free-text question. Must be in scope (tumor classes, "
            "MRI basics, patient journey, treatment overviews, mental health, "
            "practical life). Off-topic questions are politely refused."
        ),
    )


class ChatResponse(BaseModel):
    reply: str = Field(
        description="Corpus-grounded plain-language answer, or a polite refusal if out of scope."
    )
    crisis: bool = Field(
        description=(
            "True when crisis language was detected. The reply will be the "
            "canned crisis resource message instead of an LLM response."
        ),
        examples=[False],
    )


class ExplainResponse(BaseModel):
    predicted_class: str = Field(
        description="One of: glioma | meningioma | pituitary | notumor",
        examples=["glioma"],
    )
    confidence_band: str = Field(
        description="Verbal confidence band: 'fairly certain' | 'moderately confident' | 'uncertain' | 'suppressed'",
        examples=["fairly certain"],
    )
    explanation: str = Field(
        description="Plain-language explanation of the prediction with appropriate hedging."
    )
    override_applied: bool = Field(
        description="True when the glioma↔meningioma proximity override was triggered.",
        examples=[False],
    )


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    retriever: str = Field(examples=["EmbeddingRetriever"])
    corpus_pages: int = Field(examples=[38])
    model: str = Field(examples=["gemini-2.5-flash"])


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    retry_suggested: bool = Field(description="Whether a retry is likely to succeed.")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared error response definitions
# ═══════════════════════════════════════════════════════════════════════════════

COMMON_ERRORS = {
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": ErrorResponse,
        "description": "Validation error (e.g. empty message).",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Unexpected server error.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        **COMMON_ERRORS,
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Gemini LLM upstream error.",
        },
    },
    tags=["Chat"],
    summary="Ask a question about brain tumors",
    operation_id="chat",
)
async def chat(body: ChatRequest) -> ChatResponse:
    """
    Submit a free-text question and receive a corpus-grounded plain-language answer.

    **In-scope topics**: tumor classes (glioma, meningioma, pituitary, notumor),
    MRI basics, post-MRI patient journey, biopsies, second opinions, care team,
    follow-up imaging, recurrence, treatment overviews (surgery, radiation, chemo,
    watchful waiting, clinical trials), mental health, and practical life topics
    (work, school, driving, fatigue, finances, caregivers, nutrition).

    **Out-of-scope**: diagnosis for a specific person, treatment recommendations,
    prognosis predictions. These are politely refused.

    **Crisis detection**: if crisis language is detected the LLM is never called —
    a canned crisis-resource message is returned and `crisis` is `true`.
    """
    if contains_crisis_language(body.message):
        return ChatResponse(reply=app.state.crisis_text, crisis=True)

    retriever = app.state.retriever
    system_blocks, user_message = build_chat_system_prompt(retriever, body.message)

    try:
        raw = await app.state.llm.complete(system_blocks, user_message)
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "llm_unavailable", "message": exc.user_message, "retry_suggested": True},
        )

    result = scan(body.message, raw, app.state.crisis_text)
    safe = result.replacement if not result.passed else raw
    reply = append_disclaimer(safe)
    return ChatResponse(reply=reply, crisis=False)


@app.post(
    "/explain",
    response_model=ExplainResponse,
    responses={
        **COMMON_ERRORS,
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Image rejected (not an MRI, or no image supplied).",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "ML classifier service unreachable.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Gemini LLM upstream error.",
        },
    },
    tags=["Explain"],
    summary="Classify an MRI image and explain the result in plain language",
    operation_id="explain",
)
async def explain(
    image: Annotated[
        UploadFile,
        File(description="Brain MRI scan. Accepted formats: JPEG, PNG. Max size: 10 MB."),
    ],
    force: Annotated[
        bool,
        Form(description="Skip the 'looks like an MRI' heuristic gate. Useful for integration testing."),
    ] = False,
) -> ExplainResponse:
    """
    Upload a brain MRI image and receive a plain-language explanation of the
    classifier's prediction.

    **Pipeline**:
    1. Heuristic MRI gate (`image_check`) — rejects obviously non-MRI images (unless `force=true`).
    2. ML client sends the image to the parent classifier on `:8000` → raw probabilities.
    3. Confidence banding + optional glioma↔meningioma override (`confidence.py`).
    4. Retriever fetches corpus chunks for the predicted class.
    5. Gemini builds a plain-language explanation with appropriate hedging.
    6. Post-LLM regex backstop (`safety_check`) replaces any forbidden patterns.
    7. Canonical disclaimer appended.

    **Note**: the parent ML service must be running separately on port 8000 for
    this endpoint to work. `/chat` does not require it.
    """
    raw_bytes = await image.read()

    if not force and not looks_like_mri(raw_bytes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "not_an_mri",
                "message": "The uploaded image does not appear to be a brain MRI scan.",
                "retry_suggested": False,
            },
        )

    ml: MLClient = app.state.ml_client
    try:
        prediction = await ml.predict(
            image_bytes=raw_bytes,
            filename=image.filename or "upload",
            content_type=image.content_type,
        )
    except MLClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "ml_unreachable",
                "message": str(exc),
                "retry_suggested": True,
            },
        )

    confidence = derive_confidence(prediction)
    retriever = app.state.retriever
    system_blocks, user_message = build_explain_request(prediction, confidence, retriever)

    try:
        raw_text = await app.state.llm.complete(system_blocks, user_message)
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "llm_unavailable", "message": exc.user_message, "retry_suggested": True},
        )

    result = scan("", raw_text, app.state.crisis_text)
    safe = result.replacement if not result.passed else raw_text
    explanation = append_disclaimer(safe)

    return ExplainResponse(
        predicted_class=confidence.predicted_class,
        confidence_band=confidence.band or "suppressed",
        explanation=explanation,
        override_applied=confidence.is_glioma_meningioma_overlap,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Readiness probe",
    operation_id="health",
)
async def health() -> HealthResponse:
    """Returns the service readiness status and configuration summary."""
    retriever = app.state.retriever
    return HealthResponse(
        status="ok",
        retriever=type(retriever).__name__,
        corpus_pages=len(retriever._chunks) if hasattr(retriever, "_chunks") else -1,
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )


# ── demo UI (local dev only) ───────────────────────────────────────────────────

if os.getenv("SERVE_DEMO_UI", "false").lower() == "true":
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/static", StaticFiles(directory="src/chatbot/static"), name="static")

    @app.get("/", include_in_schema=False)
    async def demo_ui():
        return FileResponse("src/chatbot/static/index.html")