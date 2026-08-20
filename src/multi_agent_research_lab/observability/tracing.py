"""Tracing hooks.

This file intentionally avoids hard-binding to one provider:
- A minimal built-in JSON span always runs (no account needed).
- If `LANGSMITH_API_KEY` is set, LangChain/LangGraph's native tracing is enabled via env
  vars, so any LangChain-based calls in this process show up in the LangSmith UI.
- If Langfuse keys are set, the span is additionally exported to Langfuse when the
  optional `langfuse` package is installed.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger("multi_agent_research_lab.trace")

_langsmith_env_configured = False


def _ensure_langsmith_env() -> None:
    """Enable LangChain/LangGraph native tracing by setting the standard env vars once."""

    global _langsmith_env_configured
    if _langsmith_env_configured:
        return
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    _langsmith_env_configured = True


def _export_to_langfuse(span: dict[str, Any]) -> None:
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("Langfuse keys set but `langfuse` package is not installed; skipping")
        return

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    client.event(
        name=span["name"],
        metadata=span["attributes"],
        start_time=None,
    )


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Time a unit of work and record it as a span.

    Always produces a lightweight structured log line. Additionally forwards to
    LangSmith (via env vars) and/or Langfuse when the corresponding API keys are set.
    """

    _ensure_langsmith_env()
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.info(
            "trace_span name=%s duration=%.3fs attrs=%s",
            name,
            span["duration_seconds"],
            span["attributes"],
        )
        _export_to_langfuse(span)
