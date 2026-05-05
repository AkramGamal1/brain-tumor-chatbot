"""HTTP client for the parent ML repo's POST /predict endpoint.

Three failure modes are caught explicitly and translated into typed exceptions
so the API layer can return structured errors without leaking httpx internals.
"""

from __future__ import annotations

import httpx

from chatbot import CLASS_NAMES


class MLClientError(Exception):
    """Base class for ml_client failures."""


class MLServiceUnavailable(MLClientError):
    """The parent ML service could not be reached."""


class MLServiceTimeout(MLClientError):
    """The parent ML service did not respond in time."""


class MLServiceError(MLClientError):
    """The parent ML service returned a non-2xx status."""

    def __init__(self, status_code: int, body_excerpt: str) -> None:
        super().__init__(f"ML service returned HTTP {status_code}")
        self.status_code = status_code
        self.body_excerpt = body_excerpt

    @property
    def user_message(self) -> str:
        if self.status_code == 415:
            return (
                "The analysis service did not accept that image type. "
                "Please upload a JPEG, PNG, BMP, or TIFF image."
            )
        if self.status_code == 413:
            return (
                "The analysis service rejected the image as too large. "
                "Please use an image under 10 MB."
            )
        return "The analysis service rejected the request."


class MLClient:
    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def predict(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> dict:
        url = f"{self.base_url}/predict"
        files = {"file": (filename, image_bytes, content_type or "application/octet-stream")}

        try:
            response = await self._client.post(url, files=files)
        except httpx.ConnectError as e:
            raise MLServiceUnavailable(str(e)) from e
        except httpx.ReadTimeout as e:
            raise MLServiceTimeout(str(e)) from e
        except httpx.TimeoutException as e:
            raise MLServiceTimeout(str(e)) from e
        except httpx.RequestError as e:
            raise MLServiceUnavailable(str(e)) from e

        if response.status_code < 200 or response.status_code >= 300:
            body = response.text[:500] if response.text else ""
            raise MLServiceError(response.status_code, body)

        try:
            data = response.json()
        except ValueError as e:
            raise MLServiceError(response.status_code, "non-JSON response body") from e

        self._validate_shape(data)
        return data

    @staticmethod
    def _validate_shape(data: dict) -> None:
        for key in ("predicted_class", "confidence", "probabilities"):
            if key not in data:
                raise MLServiceError(200, f"response missing key: {key}")
        probs = data["probabilities"]
        if not isinstance(probs, dict):
            raise MLServiceError(200, "probabilities is not a dict")
        if set(probs.keys()) != set(CLASS_NAMES):
            raise MLServiceError(
                200,
                f"probabilities keys {set(probs.keys())} != expected {set(CLASS_NAMES)}",
            )
