"""Thin client for a Chat Completions-compatible LLM API, configured via env vars.

Reads API_BASE_URL, API_KEY, and MODEL_NAME from the environment (see app.config).
No API key is hard-coded here or anywhere else in this project.
"""
import logging
from typing import List, Optional

import httpx

from app import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM API call fails or is misconfigured."""


class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"
    return headers


def chat_completion(
    messages: List[ChatMessage],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Call the configured Chat Completions endpoint and return the reply text."""
    if not config.API_BASE_URL:
        raise LLMError(
            "API_BASE_URL is not configured. Set it in your environment or .env file."
        )

    url = config.API_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.MODEL_NAME,
        "messages": [m.to_dict() for m in messages],
        "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
    }

    try:
        with httpx.Client(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            response = client.post(url, json=payload, headers=_build_headers())
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise LLMError(f"LLM request timed out after {config.LLM_TIMEOUT_SECONDS}s") from exc
    except httpx.HTTPStatusError as exc:
        raise LLMError(
            f"LLM API returned an error: {exc.response.status_code} {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Failed to reach LLM API: {exc}") from exc
    except ValueError as exc:
        raise LLMError(f"LLM API returned invalid JSON: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM API response shape: {data}") from exc
