"""Phase 1 smoke tests — non-network paths only.

The eval harness in eval/run_eval.py covers end-to-end /explain flow with real
LLM calls. These tests are fast unit checks for module-level invariants.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport
from PIL import Image, ImageDraw

from chatbot import CLASS_NAMES, prompts
from chatbot.confidence import derive_confidence
from chatbot.corpus import load_corpus
from chatbot.disclaimer import DISCLAIMER, append_disclaimer
from chatbot.image_check import looks_like_mri
from chatbot.retriever import EmbeddingRetriever, WholeCorpusRetriever
from chatbot.safety_check import SAFETY_REPLACEMENT, contains_crisis_language, scan


CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"


def _grayscale_bytes(size=(224, 224)) -> bytes:
    img = Image.new("L", size, color=120)
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, size[0] - 40, size[1] - 40), fill=200)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _color_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(255, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _extreme_aspect_bytes() -> bytes:
    img = Image.new("L", (60, 480), color=100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- class names ---


def test_class_names_canonical_order():
    assert CLASS_NAMES == ("glioma", "meningioma", "notumor", "pituitary")


# --- disclaimer ---


def test_append_disclaimer_appends_when_missing():
    out = append_disclaimer("Some explanation.")
    assert out.endswith(DISCLAIMER)


def test_append_disclaimer_idempotent():
    once = append_disclaimer("Some explanation.")
    twice = append_disclaimer(once)
    assert once == twice


# --- confidence ---


def test_band_high_confidence():
    pred = {
        "predicted_class": "glioma",
        "confidence": 0.95,
        "probabilities": {"glioma": 0.95, "meningioma": 0.03, "notumor": 0.01, "pituitary": 0.01},
    }
    s = derive_confidence(pred)
    assert s.band == "fairly certain"
    assert not s.is_glioma_meningioma_overlap


def test_band_uncertain():
    pred = {
        "predicted_class": "pituitary",
        "confidence": 0.50,
        "probabilities": {"glioma": 0.20, "meningioma": 0.20, "notumor": 0.10, "pituitary": 0.50},
    }
    s = derive_confidence(pred)
    assert s.band == "uncertain"


def test_glioma_meningioma_overlap_fires():
    pred = {
        "predicted_class": "glioma",
        "confidence": 0.46,
        "probabilities": {"glioma": 0.46, "meningioma": 0.41, "notumor": 0.08, "pituitary": 0.05},
    }
    s = derive_confidence(pred)
    assert s.is_glioma_meningioma_overlap is True
    assert s.band is None
    assert s.override_message is not None
    assert "glioma" in s.override_message and "meningioma" in s.override_message


def test_glioma_meningioma_no_overlap_when_gap_large():
    pred = {
        "predicted_class": "glioma",
        "confidence": 0.80,
        "probabilities": {"glioma": 0.80, "meningioma": 0.10, "notumor": 0.05, "pituitary": 0.05},
    }
    s = derive_confidence(pred)
    assert s.is_glioma_meningioma_overlap is False
    assert s.band is not None


def test_notumor_flag():
    pred = {
        "predicted_class": "notumor",
        "confidence": 0.92,
        "probabilities": {"glioma": 0.03, "meningioma": 0.03, "notumor": 0.92, "pituitary": 0.02},
    }
    s = derive_confidence(pred)
    assert s.is_notumor is True


# --- image_check ---


def test_image_check_passes_grayscale():
    ok, reason = looks_like_mri(_grayscale_bytes())
    assert ok is True
    assert reason is None


def test_image_check_rejects_color():
    ok, reason = looks_like_mri(_color_bytes())
    assert ok is False
    assert reason and "color" in reason.lower()


def test_image_check_rejects_extreme_aspect():
    ok, reason = looks_like_mri(_extreme_aspect_bytes())
    assert ok is False
    assert reason and "aspect" in reason.lower()


# --- safety_check ---


def test_safety_passes_clean_response():
    r = scan(user_input="", llm_output="Plain language explanation, nothing forbidden.", crisis_response_text="CRISIS")
    assert r.passed is True
    assert r.replacement is None


def test_safety_blocks_diagnostic_phrasing():
    r = scan(user_input="", llm_output="You have a glioma based on this scan.", crisis_response_text="CRISIS")
    assert r.passed is False
    assert r.replacement == SAFETY_REPLACEMENT


def test_safety_blocks_treatment_phrasing():
    r = scan(
        user_input="",
        llm_output="You should get surgery for this.",
        crisis_response_text="CRISIS",
    )
    assert r.passed is False
    assert r.replacement == SAFETY_REPLACEMENT


def test_safety_blocks_user_targeted_recommendation():
    r = scan(
        user_input="",
        llm_output="I recommend you start chemotherapy as soon as possible.",
        crisis_response_text="CRISIS",
    )
    assert r.passed is False


def test_safety_allows_educational_treatment_mention():
    """Generic 'doctors often recommend X for Y' is now in-scope education."""
    r = scan(
        user_input="",
        llm_output=(
            "Doctors often recommend surgery for high-grade gliomas, "
            "while watchful waiting may be appropriate for some small, "
            "asymptomatic meningiomas."
        ),
        crisis_response_text="CRISIS",
    )
    assert r.passed is True


def test_safety_allows_informational_mental_health_content():
    r = scan(
        user_input="I am scared about my diagnosis",
        llm_output=(
            "Many people experience anxiety and low mood after receiving "
            "imaging results. Talking to a therapist or joining a support "
            "group can help. If feelings of hopelessness persist, please "
            "reach out to a clinician."
        ),
        crisis_response_text="CRISIS",
    )
    assert r.passed is True


def test_safety_blocks_ruling_out():
    r = scan(
        user_input="",
        llm_output="The model rules out other conditions outside the four classes.",
        crisis_response_text="CRISIS",
    )
    assert r.passed is False


def test_safety_blocks_percent():
    r = scan(
        user_input="",
        llm_output="The model is 95% confident in this prediction.",
        crisis_response_text="CRISIS",
    )
    assert r.passed is False


def test_safety_allows_decimal_far_from_confidence():
    r = scan(
        user_input="",
        llm_output="The brain has 86 billion neurons; that 86.0 figure is approximate.",
        crisis_response_text="CRISIS",
    )
    # 86.0 is a decimal but no confidence word nearby — should pass
    assert r.passed is True


def test_crisis_substitution_fires():
    r = scan(
        user_input="I don't want to live with this",
        llm_output="Here is a tumor explanation that ignored the crisis.",
        crisis_response_text="CRISIS_RESOURCES_TEXT",
    )
    assert r.passed is False
    assert r.replacement == "CRISIS_RESOURCES_TEXT"


def test_crisis_does_not_fire_when_resources_present():
    r = scan(
        user_input="I don't want to live with this",
        llm_output="Please call 988 — the Suicide & Crisis Lifeline.",
        crisis_response_text="CRISIS_RESOURCES_TEXT",
    )
    assert r.passed is True


def test_contains_crisis_language():
    assert contains_crisis_language("I want to hurt myself")
    assert contains_crisis_language("what's the point of going on")
    assert not contains_crisis_language("I want to understand my MRI")


# --- corpus ---


def test_corpus_loads_all_pages():
    bundle = load_corpus(CORPUS_DIR)
    assert len(bundle.chunks) >= 12
    ids = {c.id for c in bundle.chunks}
    required = {
        "glioma", "meningioma", "pituitary", "notumor",
        "mri-basics", "model-capabilities", "what-this-model-cannot-tell-you",
        "seek-care", "confidence-meaning", "emotional-impact",
        "crisis-resources", "questions-to-ask-your-doctor",
    }
    assert required.issubset(ids)


def test_corpus_extracts_crisis_response_text():
    bundle = load_corpus(CORPUS_DIR)
    assert bundle.crisis_response_text
    assert "988" in bundle.crisis_response_text
    assert "crisis" in bundle.crisis_response_text.lower()


# --- retriever ---


def test_whole_corpus_retriever_returns_full_corpus():
    bundle = load_corpus(CORPUS_DIR)
    retriever = WholeCorpusRetriever(bundle)
    result = retriever.retrieve()
    assert len(result.chunks) == len(bundle.chunks)
    assert {c.id for c in result.chunks} == {c.id for c in bundle.chunks}


def test_whole_corpus_retriever_format_is_byte_stable():
    """Phase 2.1 must not perturb prompt-cache hashes."""
    bundle = load_corpus(CORPUS_DIR)
    expected = "# Educational corpus\n\n" + "\n\n---\n\n".join(
        f"## {c.title}\n\n{c.body}" for c in bundle.chunks
    )
    retriever = WholeCorpusRetriever(bundle)
    assert retriever.retrieve().formatted_text == expected


def test_whole_corpus_retriever_query_is_ignored():
    bundle = load_corpus(CORPUS_DIR)
    retriever = WholeCorpusRetriever(bundle)
    a = retriever.retrieve()
    b = retriever.retrieve(query="anything")
    assert a.formatted_text == b.formatted_text


def test_embedding_retriever_returns_topk_for_query():
    bundle = load_corpus(CORPUS_DIR)
    retriever = EmbeddingRetriever(bundle, top_k=3)
    retriever.build()
    result = retriever.retrieve("What is a glioma?")
    assert 1 <= len(result.chunks) <= 5  # top_k=3 + always_include_ids may merge
    ids = {c.id for c in result.chunks}
    assert "glioma" in ids, f"glioma should rank top-3 for its own definition, got {ids}"


def test_embedding_retriever_falls_back_to_full_corpus_when_no_query():
    bundle = load_corpus(CORPUS_DIR)
    retriever = EmbeddingRetriever(bundle, top_k=3)
    retriever.build()
    result = retriever.retrieve(None)
    assert len(result.chunks) == len(bundle.chunks)


def test_embedding_retriever_force_includes_ids():
    bundle = load_corpus(CORPUS_DIR)
    retriever = EmbeddingRetriever(bundle, top_k=2)
    retriever.build()
    # Query that has nothing to do with the crisis page; force-include it.
    result = retriever.retrieve(
        "How does an MRI scan work?",
        always_include_ids=("crisis-resources",),
    )
    assert "crisis-resources" in {c.id for c in result.chunks}


def test_embedding_retriever_format_preserves_corpus_order():
    bundle = load_corpus(CORPUS_DIR)
    retriever = EmbeddingRetriever(bundle, top_k=4)
    retriever.build()
    result = retriever.retrieve("brain tumor types")
    by_idx = {c.id: i for i, c in enumerate(bundle.chunks)}
    indices = [by_idx[c.id] for c in result.chunks]
    assert indices == sorted(indices), "result should preserve corpus order"


# --- /chat prompt builder + endpoint ---


def test_build_chat_system_prompt_returns_two_blocks_and_passes_message():
    bundle = load_corpus(CORPUS_DIR)
    retriever = WholeCorpusRetriever(bundle)
    system_blocks, user_msg = prompts.build_chat_system_prompt(
        retriever=retriever, user_message="What is a glioma?"
    )
    assert len(system_blocks) == 2
    assert system_blocks[0]["type"] == "text"
    assert "educational chatbot" in system_blocks[0]["text"].lower()
    assert system_blocks[1]["text"] == retriever.retrieve().formatted_text
    assert system_blocks[1].get("cache_control") == {"type": "ephemeral"}
    assert user_msg == "What is a glioma?"


def test_chat_rules_list_in_scope_and_oos_categories():
    rules = prompts._CHAT_RULES.lower()
    assert "in-scope" in rules
    assert "out-of-scope" in rules
    assert "refuse" in rules
    assert "forbidden" in rules
    assert "do not include it yourself" in rules


@pytest.mark.asyncio
async def test_chat_endpoint_short_circuits_on_crisis_input(monkeypatch):
    """Crisis pre-check must short-circuit before any LLM call."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("RETRIEVER", "whole_corpus")

    from chatbot import api as chatbot_api

    class TrackingLLM:
        def __init__(self):
            self.call_count = 0

        async def complete(self, system_blocks, user_message):
            self.call_count += 1
            return "should not be called"

    transport = ASGITransport(app=chatbot_api.app)
    async with chatbot_api.app.router.lifespan_context(chatbot_api.app):
        tracking = TrackingLLM()
        chatbot_api.state["llm_client"] = tracking

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": "I want to die"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["safety_substituted"] is True
    assert body.get("reason") == "crisis"
    assert "988" in body["response"]
    # Crisis substitution must NOT carry the educational disclaimer.
    assert DISCLAIMER not in body["response"]
    assert tracking.call_count == 0


@pytest.mark.asyncio
async def test_chat_endpoint_returns_clean_503_on_llm_error(monkeypatch):
    """Quota exhaustion / upstream 503 must return JSON, not bubble as 500."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("RETRIEVER", "whole_corpus")

    from chatbot import api as chatbot_api
    from chatbot.llm import LLMRateLimited

    class RateLimitedLLM:
        async def complete(self, system_blocks, user_message):
            raise LLMRateLimited("Daily limit reached.", status_code=429)

    transport = ASGITransport(app=chatbot_api.app)
    async with chatbot_api.app.router.lifespan_context(chatbot_api.app):
        chatbot_api.state["llm_client"] = RateLimitedLLM()

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": "What is a glioma?"})

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "llm_rate_limited"
    assert body["retry_suggested"] is True
    assert "limit" in body["message"].lower()


@pytest.mark.asyncio
async def test_chat_endpoint_appends_disclaimer_on_forbidden_pattern_substitution(
    monkeypatch,
):
    """L1 fix: SAFETY_REPLACEMENT path now carries the disclaimer."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("RETRIEVER", "whole_corpus")

    from chatbot import api as chatbot_api

    class ForbiddenLLM:
        async def complete(self, system_blocks, user_message):
            # This trips the diagnostic-claim regex.
            return "Based on what you describe, you have a glioma."

    transport = ASGITransport(app=chatbot_api.app)
    async with chatbot_api.app.router.lifespan_context(chatbot_api.app):
        chatbot_api.state["llm_client"] = ForbiddenLLM()

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={"message": "Tell me what's happening with my brain"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["safety_substituted"] is True
    assert body.get("reason") != "crisis"
    # L1 fix: the SAFETY_REPLACEMENT path now carries the disclaimer.
    assert DISCLAIMER in body["response"]
    assert SAFETY_REPLACEMENT.split(".")[0] in body["response"]
