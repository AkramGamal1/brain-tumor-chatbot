"""Verbal confidence bands and the glioma↔meningioma overlap override."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ConfidenceSummary:
    predicted_class: str
    band: str | None
    override_message: str | None
    top_two: tuple[str, str]
    is_notumor: bool
    is_glioma_meningioma_overlap: bool


GLIOMA_MENINGIOMA_OVERRIDE_TEXT = (
    "CRITICAL OVERRIDE: For this prediction, you MUST include the following sentence in your response, exactly as written, on a single line:\n\n"
    "\"The model isn't confident about distinguishing this case clearly between glioma and meningioma — please show this scan to a clinician for a definitive interpretation.\"\n\n"
    "Both class names ('glioma' and 'meningioma') must appear together in this single sentence. Do not split them across separate sentences."
)


def _band_for(probability: float) -> str:
    high = _env_float("BAND_HIGH", 0.85)
    low = _env_float("BAND_LOW", 0.65)
    if probability >= high:
        return "fairly certain"
    if probability >= low:
        return "moderately confident"
    return "uncertain"


def derive_confidence(prediction: dict) -> ConfidenceSummary:
    """Translate a /predict response into a ConfidenceSummary.

    The LLM never sees raw probabilities — it consumes the band string (or the
    override message) produced here.
    """
    probabilities: dict[str, float] = prediction["probabilities"]
    predicted_class: str = prediction["predicted_class"]
    sorted_pairs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top_two = (sorted_pairs[0][0], sorted_pairs[1][0])
    top_two_set = set(top_two)

    gap_threshold = _env_float("GLIOMA_MENINGIOMA_GAP", 0.20)
    gap = sorted_pairs[0][1] - sorted_pairs[1][1]
    overlap = top_two_set == {"glioma", "meningioma"} and gap < gap_threshold

    if overlap:
        return ConfidenceSummary(
            predicted_class=predicted_class,
            band=None,
            override_message=GLIOMA_MENINGIOMA_OVERRIDE_TEXT,
            top_two=top_two,
            is_notumor=(predicted_class == "notumor"),
            is_glioma_meningioma_overlap=True,
        )

    return ConfidenceSummary(
        predicted_class=predicted_class,
        band=_band_for(probabilities[predicted_class]),
        override_message=None,
        top_two=top_two,
        is_notumor=(predicted_class == "notumor"),
        is_glioma_meningioma_overlap=False,
    )
