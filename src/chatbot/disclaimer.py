DISCLAIMER = (
    "This is not medical advice. Please consult a qualified clinician for any "
    "medical decisions."
)


def append_disclaimer(text: str) -> str:
    if DISCLAIMER in text:
        return text
    stripped = text.rstrip()
    return f"{stripped}\n\n{DISCLAIMER}"
