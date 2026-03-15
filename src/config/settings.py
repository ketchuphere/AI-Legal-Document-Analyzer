"""
src/config/settings.py
======================
Centralised configuration using Pydantic BaseSettings.
All tuneable values live here; override via environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration.

    Environment variables take precedence over .env values,
    which take precedence over the defaults defined here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""

    # Available Claude models exposed in the UI dropdown
    AVAILABLE_MODELS: dict = {
        "Claude Sonnet 4 (Recommended)": "claude-sonnet-4-20250514",
        "Claude Haiku 4.5 (Fast)":       "claude-haiku-4-5-20251001",
        "Claude Opus 4 (Most Capable)":  "claude-opus-4-20250514",
    }

    # Default model
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"

    # ── LLM generation parameters ────────────────────────────────────────
    SUMMARY_TEMPERATURE:     float = 0.2
    RISKS_TEMPERATURE:       float = 0.2
    SUGGESTIONS_TEMPERATURE: float = 0.3
    CLAUSES_TEMPERATURE:     float = 0.1   # deterministic extraction
    COMPLIANCE_TEMPERATURE:  float = 0.1
    QA_TEMPERATURE:          float = 0.3

    # ── Document chunking ─────────────────────────────────────────────────
    DEFAULT_CHUNK_CHARS:   int = 6_000
    DEFAULT_CHUNK_OVERLAP: int = 400
    MAX_TOKENS_PER_CALL:   int = 2_048

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
