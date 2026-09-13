"""
Optional LangSmith tracing, so one turn can be opened up and looked at.

This is the other half of observability, and it answers a different question
from `telemetry.py`. The SQLite store answers "how has the model behaved over
the last week" -- aggregates, trends, whether today is worse than Tuesday.
LangSmith answers "what exactly happened on *that* turn" -- the triage call
and what it replied, which specialist ran, every tool call with its arguments
and its result, nested and timed. You want the first to notice a problem and
the second to understand it.

Nothing here is required. With no LANGSMITH_API_KEY, or without the package
installed, `traced` is an identity decorator and the app behaves exactly as it
did before -- no import cost, no network call, no wrapper in the stack. That
matters more than it sounds: an observability layer that can break the thing
it observes is a liability, and tracing is the layer most likely to be
misconfigured in production.

    pip install langsmith

    LANGSMITH_API_KEY=lsv2_...        # from smith.langchain.com
    LANGSMITH_TRACING=true            # the switch; absent means off
    LANGSMITH_PROJECT=customerchatbot # optional, groups the runs
    LANGSMITH_ENDPOINT=...            # optional, for self-hosted

This app does not use LangChain, and does not need to -- `@traceable` works on
any function. Runs appear as a tree: one `chat_turn` per customer message with
`triage`, the specialist, and each tool nested beneath it.

Traces carry the full message and reply text to LangSmith's servers, the same
content the local database holds. If that is not acceptable for a deployment,
leave LANGSMITH_TRACING unset -- the local store keeps working on its own.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Resolved once at import. Tracing is a deployment-level decision, and
# re-checking the environment on every model call would cost more than the
# feature.
_enabled: bool = False
_status: str = "off — LANGSMITH_TRACING is not set"
_traceable: Optional[Callable] = None


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _resolve() -> None:
    global _enabled, _status, _traceable

    wants = _truthy(os.environ.get("LANGSMITH_TRACING")
                    or os.environ.get("LANGCHAIN_TRACING_V2"))
    if not wants:
        _status = "off — LANGSMITH_TRACING is not set"
        return

    if not (os.environ.get("LANGSMITH_API_KEY")
            or os.environ.get("LANGCHAIN_API_KEY")):
        # Asked for and not usable is worth saying out loud. Silently doing
        # nothing here is how you discover at the end of a debugging session
        # that no traces were ever sent.
        _status = "off — LANGSMITH_TRACING is set but LANGSMITH_API_KEY is missing"
        logger.warning(_status)
        return

    try:
        from langsmith import traceable as _ls_traceable
    except ImportError:
        _status = "off — LANGSMITH_TRACING is set but langsmith is not installed"
        logger.warning("%s (pip install langsmith)", _status)
        return

    # The SDK reads LANGCHAIN_* names; mirror the LANGSMITH_* ones onto them so
    # either spelling works in a .env.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    if os.environ.get("LANGSMITH_ENDPOINT"):
        os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
    project = os.environ.get("LANGSMITH_PROJECT", "customerchatbot")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)

    _traceable = _ls_traceable
    _enabled = True
    _status = f"on — project {project!r}"
    logger.info("LangSmith tracing %s", _status)


_resolve()


def enabled() -> bool:
    return _enabled


def status() -> str:
    """One line for /status, so you can see whether traces are being sent."""
    return _status


def traced(name: str, run_type: str = "chain") -> Callable:
    """Decorate a function so its call shows up as a span in LangSmith.

    A no-op passthrough when tracing is off, which is the normal case. The
    decorator is applied at import time, so when tracing is off the function
    is left exactly as written -- there is no wrapper on the hot path at all.
    """
    def decorate(func: Callable) -> Callable:
        if not _enabled or _traceable is None:
            return func
        try:
            return _traceable(name=name, run_type=run_type)(func)
        except Exception as exc:                        # pragma: no cover
            # A tracing decorator must never be the reason the app fails to
            # import.
            logger.warning("could not trace %s: %s", name, exc)
            return func
    return decorate


def label(text: str) -> None:
    """Rename the span currently being traced.

    `@traceable` fixes a span's name at decoration time, which is why every
    tool call arrived in LangSmith as a row called "tool" -- three identical
    rows per turn, each needing a click to find out which tool it was. The
    name is the one part of a span you read without opening it, so it should
    say `search_knowledge`, not `tool`.

    Silent no-op when tracing is off or there is no active span.
    """
    if not _enabled:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.name = text
    except Exception:                                   # pragma: no cover
        # Naming is cosmetic. It must never be the reason a turn fails.
        pass


def add_usage(input_tokens: int, output_tokens: int,
              model: Optional[str] = None,
              provider: Optional[str] = None) -> None:
    """Attach token usage to the span currently being traced.

    Without this, LangSmith receives spans marked `run_type="llm"` that carry
    no token counts at all -- its cost and token views stay empty, and the
    only place the numbers exist is the local turn log. It infers usage
    automatically only when it wraps the model client itself; here it is
    tracing our own functions, so the numbers have to be handed to it.

    Accumulates, because one specialist span covers however many model calls
    its tool loop took. `ls_model_name` and `ls_provider` are what LangSmith
    prices the tokens against.

    Silent no-op when tracing is off.
    """
    if not _enabled:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return
        if run.metadata is None:
            run.metadata = {}
        usage = run.metadata.get("usage_metadata") or {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        usage["input_tokens"] += int(input_tokens or 0)
        usage["output_tokens"] += int(output_tokens or 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        run.metadata["usage_metadata"] = usage
        if model:
            run.metadata.setdefault("ls_model_name", model)
        if provider:
            run.metadata.setdefault("ls_provider", provider)
    except Exception:                                   # pragma: no cover
        pass


def reset_for_tests() -> None:
    """Re-read the environment. Only tests should need this."""
    global _enabled, _status, _traceable
    _enabled, _status, _traceable = False, "off — LANGSMITH_TRACING is not set", None
    _resolve()
