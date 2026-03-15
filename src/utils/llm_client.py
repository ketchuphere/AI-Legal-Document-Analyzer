"""
src/utils/llm_client.py
=======================
Thin wrapper around the Anthropic Messages API.
Provides a simple `call()` interface used by all agent nodes.
"""

import os
import logging
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Stateless wrapper around anthropic.Anthropic.

    Usage:
        client = LLMClient(api_key="sk-…", model="claude-sonnet-4-20250514")
        text = client.call("Summarise this document: …", temperature=0.2)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
    ):
        self.api_key    = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = anthropic.Anthropic(api_key=self.api_key)

    # ──────────────────────────────────────────────────────────────────────
    def call(
        self,
        prompt: str,
        *,
        system: str = "You are an expert document analysis assistant.",
        temperature: float = 0.2,
    ) -> str:
        """
        Send a single-turn prompt to the model and return the text response.

        Args:
            prompt:      The user-turn message.
            system:      System prompt for role/persona.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).

        Returns:
            Stripped text content of the model's reply.

        Raises:
            anthropic.APIError: On API-level failures (rate limit, auth, etc.).
        """
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except anthropic.APIStatusError as e:
            logger.error("Anthropic API error %s: %s", e.status_code, e.message)
            raise
        except Exception as e:
            logger.error("Unexpected LLM error: %s", e)
            raise

    # ──────────────────────────────────────────────────────────────────────
    def call_with_history(
        self,
        messages: list[dict],
        *,
        system: str = "You are an expert document analysis assistant.",
        temperature: float = 0.3,
    ) -> str:
        """
        Multi-turn call — pass a list of {"role": …, "content": …} dicts.
        Used by the Q&A chat chain.
        """
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            return response.content[0].text.strip()
        except anthropic.APIStatusError as e:
            logger.error("Anthropic API error %s: %s", e.status_code, e.message)
            raise
