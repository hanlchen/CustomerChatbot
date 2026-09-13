"""
Runtime configuration.

Every value reads from the environment with a sensible default, so

    API_PORT=8001 python app.py

works as documented. Previously these were hardcoded, which silently ignored
any environment variable you set.

Model and retrieval settings are NOT here -- they are read where they are used
(`llm_providers.py`, `retrieval.py`) so each module owns its own configuration.
See `.env.example` for the full list.
"""

import os


def _flag(name: str, default: bool) -> bool:
    """Read a boolean env var, accepting the usual spellings."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- API ---------------------------------------------------------------------
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Enables uvicorn's auto-reload. Off in production, where it would restart the
# process on every file touch and drop all in-memory sessions.
DEBUG = _flag("DEBUG", ENVIRONMENT == "development")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# Removed: DATABASE_URL, DATABASE_POOL_SIZE, DATABASE_MAX_OVERFLOW, REDIS_HOST,
# REDIS_PORT, REDIS_PASSWORD, CACHE_TTL_SECONDS, SECRET_KEY, API_KEY_REQUIRED,
# LOG_FILE, STRUCTURED_LOGGING, ENABLE_CACHING, ENABLE_COMPRESSION,
# MAX_REQUEST_SIZE, REQUEST_TIMEOUT.
#
# None of them were imported anywhere. They described a SQL database, a Redis
# cache and an auth layer that this project does not have, which made the
# configuration look like it did more than it does. The last two survived a
# first pass and were just as decorative: nothing read either, so setting
# MAX_REQUEST_SIZE bought you nothing. Request size is bounded where it is
# actually enforced -- `max_length` on the Pydantic request models.
#
# The rule for this file: a constant lives here only if something imports it.
# `test_production.py` checks that, so a knob cannot go back to being scenery.
