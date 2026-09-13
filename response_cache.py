"""
A small TTL cache for knowledge-base lookups.

What is cached and what is not is the whole design. Policy answers are the
same for every customer and change when someone edits a document, so a five
minute window costs nothing and saves a retrieval on every repeat question.
Account data is the opposite: it is one person's, it changes when their order
ships, and a cache that returned it across sessions would be a data leak
wearing a performance costume.

So this cache is only ever wired to `search_knowledge` and
`list_policy_topics`, and `assert_cacheable` refuses anything else loudly
rather than letting a future edit quietly make that mistake.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300.0     # five minutes
DEFAULT_MAX_ENTRIES = 512

# Only these. Everything else is somebody's private data.
CACHEABLE_TOOLS = frozenset({"search_knowledge", "list_policy_topics"})


class NotCacheable(RuntimeError):
    """Raised when something tries to cache per-customer data."""


def assert_cacheable(tool_name: str) -> None:
    if tool_name not in CACHEABLE_TOOLS:
        raise NotCacheable(
            f"{tool_name} returns customer-specific data and must never be "
            f"cached. Cacheable tools: {sorted(CACHEABLE_TOOLS)}"
        )


@dataclass
class _Entry:
    value: Any
    stored_at: float


class ResponseCache:
    """Keyed on the tool and its arguments. Thread-safe, bounded, expiring."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                 max_entries: int = DEFAULT_MAX_ENTRIES):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # -- keys --------------------------------------------------------------

    @staticmethod
    def key(tool_name: str, arguments: Dict[str, Any]) -> str:
        """A stable key. Arguments are sorted so ordering cannot miss."""
        assert_cacheable(tool_name)
        blob = json.dumps(arguments or {}, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:32]
        return f"{tool_name}:{digest}"

    # -- access ------------------------------------------------------------

    def get(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Any]:
        """(hit, value). Never raises on a miss."""
        cache_key = self.key(tool_name, arguments)
        now = time.time()
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                self.misses += 1
                return False, None
            if (now - entry.stored_at) > self.ttl:
                del self._entries[cache_key]
                self.misses += 1
                return False, None
            self.hits += 1
            return True, entry.value

    def put(self, tool_name: str, arguments: Dict[str, Any], value: Any) -> None:
        cache_key = self.key(tool_name, arguments)
        with self._lock:
            if len(self._entries) >= self.max_entries:
                # Oldest first. A policy corpus this size never reaches the
                # cap in practice; the bound exists so a pathological run
                # cannot grow the process without limit.
                oldest = min(self._entries.items(), key=lambda kv: kv[1].stored_at)
                del self._entries[oldest[0]]
                self.evictions += 1
            self._entries[cache_key] = _Entry(value=value, stored_at=time.time())

    async def through(self, tool_name: str, arguments: Dict[str, Any],
                      produce: Callable[[], Any]) -> Tuple[Any, bool]:
        """Read through the cache. Returns (value, was_a_hit).

        A failed lookup is not cached: caching an error would turn a blip into
        five minutes of the same wrong answer.
        """
        hit, value = self.get(tool_name, arguments)
        if hit:
            return value, True

        value = await produce()
        if isinstance(value, dict) and value.get("ok") is False:
            return value, False
        self.put(tool_name, arguments, value)
        return value, False

    # -- reporting ---------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "ttl_seconds": self.ttl,
            "cacheable_tools": sorted(CACHEABLE_TOOLS),
        }


_cache: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache


def reset_cache() -> None:
    """Fresh cache (used by tests)."""
    global _cache
    _cache = ResponseCache()
