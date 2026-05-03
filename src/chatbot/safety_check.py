"""Post-LLM forbidden-pattern scan + crisis enforcement.

Defense in depth — the LLM is also instructed not to produce these things, but
this module is the deterministic backstop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SAFETY_REPLACEMENT = (
    "I caught myself about to give medical advice I'm not qualified to give. "
    "Please consult a clinician."
)


_DIAGNOSTIC_PATTERNS = [
    re.compile(r"\byou\s+(?:have|are\s+diagnosed\s+with)\b", re.IGNORECASE),
    re.compile(r"\bI\s+diagnose\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(?:confirms|means\s+you\s+have)\b", re.IGNORECASE),
    re.compile(r"\byour\s+scan\s+shows\s+you\s+have\b", re.IGNORECASE),
]

_TREATMENT_PATTERNS = [
    re.compile(r"\bprescrib(?:e|es|ed|ing)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:you\s+(?:should|need\s+to)|I\s+recommend(?:ed)?)\s+(?:undergo|have|get|start)\s+"
        r"(?:surgery|chemo(?:therapy)?|radiation|radiotherapy)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brecommend(?:ed|ing)?\s+(?:treatment|medication|surgery|chemo(?:therapy)?|radiation|radiotherapy)\b",
        re.IGNORECASE,
    ),
]

_RULING_OUT_PATTERN = re.compile(r"\brul(?:e|es|ed|ing)\s+out\b", re.IGNORECASE)

_PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
_DECIMAL_NEAR_CONFIDENCE = re.compile(
    r"(?:confidence|probability|likelihood|chance|certain(?:ty)?)[^.\n]{0,30}\d+\.\d+"
    r"|\d+\.\d+[^.\n]{0,30}(?:confidence|probability|likelihood|chance|certain(?:ty)?)",
    re.IGNORECASE,
)

_MENTAL_HEALTH_DIAGNOSIS_PATTERNS = [
    re.compile(
        r"\byou\s+(?:are|have|seem\s+to\s+have|appear\s+to\s+have|are\s+suffering\s+from)\s+"
        r"(?:depression|anxiety\s+disorder|PTSD|bipolar)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+(?:recommend|suggest|prescribe)\s+(?:therapy|counseling|psychotherapy|"
        r"medication|antidepressants?|anxiolytics?)\b",
        re.IGNORECASE,
    ),
]

_FORBIDDEN_PATTERNS = (
    _DIAGNOSTIC_PATTERNS
    + _TREATMENT_PATTERNS
    + [_RULING_OUT_PATTERN, _PERCENT_PATTERN, _DECIMAL_NEAR_CONFIDENCE]
    + _MENTAL_HEALTH_DIAGNOSIS_PATTERNS
)


_CRISIS_PHRASES = (
    "i don't want to live",
    "i dont want to live",
    "i want to die",
    "i want to hurt myself",
    "i'm going to hurt myself",
    "im going to hurt myself",
    "i'm thinking of hurting myself",
    "im thinking of hurting myself",
    "i'm thinking of suicide",
    "im thinking of suicide",
    "thinking of killing myself",
    "kill myself",
    "end my life",
    "end it all",
    "what's the point of going on",
    "whats the point of going on",
    "what's the point of living",
    "whats the point of living",
    "i can't go on",
    "i cant go on",
    "no reason to live",
)

_CRISIS_RESOURCE_MARKERS = ("988", "crisis line", "emergency number", "crisis lifeline")


@dataclass(frozen=True)
class SafetyResult:
    passed: bool
    replacement: str | None


def contains_crisis_language(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _CRISIS_PHRASES)


def _has_crisis_resources(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CRISIS_RESOURCE_MARKERS)


def _matches_forbidden(text: str) -> bool:
    return any(p.search(text) for p in _FORBIDDEN_PATTERNS)


def scan(
    user_input: str,
    llm_output: str,
    crisis_response_text: str,
) -> SafetyResult:
    """Apply defense-in-depth checks to the LLM output.

    Order:
    1. Forbidden pattern in LLM output → substitute with the safety replacement.
    2. Crisis backstop: if user input had crisis indicators and LLM output lacks
       crisis resources → substitute with the canned crisis response.
    3. Otherwise pass.
    """
    if _matches_forbidden(llm_output):
        return SafetyResult(passed=False, replacement=SAFETY_REPLACEMENT)

    if contains_crisis_language(user_input) and not _has_crisis_resources(llm_output):
        return SafetyResult(passed=False, replacement=crisis_response_text)

    return SafetyResult(passed=True, replacement=None)
