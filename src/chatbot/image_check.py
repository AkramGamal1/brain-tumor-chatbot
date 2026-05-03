"""Lightweight pre-flight check that an upload looks plausibly like an MRI."""

from __future__ import annotations

import io
import os

from PIL import Image


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def looks_like_mri(image_bytes: bytes) -> tuple[bool, str | None]:
    """Return (passed, reason). reason is None on pass, a short string on reject."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        return False, "Could not open the image. Please upload a valid JPEG, PNG, BMP, or TIFF."

    width, height = img.size
    if width == 0 or height == 0:
        return False, "Image has zero width or height."

    longer, shorter = max(width, height), min(width, height)
    aspect_max = _env_float("IMAGE_ASPECT_RATIO_MAX", 2.0)
    if longer > aspect_max * shorter:
        return False, "Image aspect ratio is unusual for an MRI scan."

    sat_threshold = _env_float("IMAGE_SATURATION_THRESHOLD", 0.15)
    rgb = img.convert("RGB")
    hsv = rgb.convert("HSV")
    saturation_band = hsv.getchannel("S")
    pixels = list(saturation_band.getdata())
    if not pixels:
        return False, "Image contains no pixel data."
    mean_saturation = sum(pixels) / len(pixels) / 255.0
    if mean_saturation > sat_threshold:
        return False, "Image appears to be in color; MRI scans are typically grayscale."

    return True, None
