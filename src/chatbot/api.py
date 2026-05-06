"""FastAPI app exposing /explain (Phase 1) and /health.

Endpoint flow follows the plan exactly:
  1. crisis pre-check on user input  (Phase 1 /explain has no text input)
  2. image_check (skipped if force=true)
  3. ml_client.predict
  4. prompts → llm.complete
  5. safety_check → disclaimer append
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chatbot import prompts
from chatbot.confidence import derive_confidence
from chatbot.corpus import load_corpus
from chatbot.disclaimer import append_disclaimer
from chatbot.image_check import looks_like_mri
from chatbot.llm import LLMClient
from chatbot.ml_client import (
    MLClient,
    MLServiceError,
    MLServiceTimeout,
    MLServiceUnavailable,
)
from chatbot.retriever import WholeCorpusRetriever
from chatbot.safety_check import contains_crisis_language, scan as safety_scan

load_dotenv()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

state: dict = {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["corpus"] = load_corpus(_project_root() / "corpus")
    state["retriever"] = WholeCorpusRetriever(state["corpus"])
    state["ml_client"] = MLClient(
        base_url=os.environ.get("ML_API_BASE_URL", "http://localhost:8000"),
    )
    state["llm_client"] = LLMClient(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    )
    try:
        yield
    finally:
        await state["ml_client"].aclose()


app = FastAPI(title="Brain Tumor Chatbot", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/explain")
async def explain(
    image: UploadFile = File(...),
    force: bool = Form(False),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={
                "error": "unsupported_media",
                "message": (
                    f"Image type {image.content_type!r} is not supported. "
                    "Use JPEG, PNG, BMP, or TIFF."
                ),
                "retry_suggested": False,
            },
        )

    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": "payload_too_large",
                "message": "Image exceeds the 10 MB limit.",
                "retry_suggested": False,
            },
        )

    if not force:
        ok, reason = looks_like_mri(image_bytes)
        if not ok:
            return {
                "status": "image_warning",
                "reason": reason,
                "force_available": True,
            }

    try:
        prediction = await state["ml_client"].predict(
            image_bytes,
            image.filename or "upload",
            image.content_type,
        )
    except MLServiceUnavailable:
        return JSONResponse(
            status_code=502,
            content={
                "error": "ml_service_unavailable",
                "message": "The analysis service is not reachable right now. Please try again shortly.",
                "retry_suggested": True,
            },
        )
    except MLServiceTimeout:
        return JSONResponse(
            status_code=504,
            content={
                "error": "ml_service_timeout",
                "message": "The analysis service did not respond in time. Please try again.",
                "retry_suggested": True,
            },
        )
    except MLServiceError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": "ml_service_error",
                "message": exc.user_message,
                "retry_suggested": False,
            },
        )

    summary = derive_confidence(prediction)
    system_blocks, user_message = prompts.build_explain_request(
        prediction=prediction,
        confidence=summary,
        retriever=state["retriever"],
    )
    explanation = await state["llm_client"].complete(
        system_blocks=system_blocks,
        user_message=user_message,
    )

    safety = safety_scan(
        user_input="",
        llm_output=explanation,
        crisis_response_text=state["corpus"].crisis_response_text,
    )
    if safety.replacement is not None:
        return {
            "status": "ok",
            "prediction": prediction,
            "explanation": safety.replacement,
            "safety_substituted": True,
        }

    return {
        "status": "ok",
        "prediction": prediction,
        "explanation": append_disclaimer(explanation),
    }


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


@app.post("/chat")
async def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        return JSONResponse(
            status_code=400,
            content={
                "error": "empty_message",
                "message": "Message cannot be empty.",
            },
        )

    if contains_crisis_language(message):
        return {
            "status": "ok",
            "response": state["corpus"].crisis_response_text,
            "safety_substituted": True,
            "reason": "crisis",
        }

    system_blocks, user_msg = prompts.build_chat_system_prompt(
        retriever=state["retriever"],
        user_message=message,
    )
    response = await state["llm_client"].complete(
        system_blocks=system_blocks,
        user_message=user_msg,
    )

    safety = safety_scan(
        user_input=message,
        llm_output=response,
        crisis_response_text=state["corpus"].crisis_response_text,
    )
    if safety.replacement is not None:
        return {
            "status": "ok",
            "response": safety.replacement,
            "safety_substituted": True,
        }

    return {
        "status": "ok",
        "response": append_disclaimer(response),
    }
