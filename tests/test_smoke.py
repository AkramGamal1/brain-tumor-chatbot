"""Phase 1 smoke tests — non-network paths only.

The eval harness in eval/run_eval.py covers end-to-end /explain flow with real
LLM calls. These tests are fast unit checks for module-level invariants.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from chatbot import CLASS_NAMES
from chatbot.confidence import derive_confidence
from chatbot.corpus import load_corpus
from chatbot.disclaimer import DISCLAIMER, append_disclaimer
from chatbot.image_check import looks_like_mri
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
        llm_output="The recommended treatment is surgery followed by chemotherapy.",
        crisis_response_text="CRISIS",
    )
    assert r.passed is False
    assert r.replacement == SAFETY_REPLACEMENT


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
