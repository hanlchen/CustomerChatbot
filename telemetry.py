"""
Persistent per-turn telemetry, and the health numbers computed from it.

The in-process counters in `app.py` answer "what is happening right now" and
lose everything on restart. This answers "how has the model been behaving",
which is a question about last week as much as this minute -- so it goes to
disk, in SQLite, one row per customer turn.

Why SQLite and not a log file: every health number here is an aggregate over
a time window, and re-reading a growing JSONL file to compute a p95 gets
slower every day the app stays up. SQL does the aggregation where the data
is. It is also stdlib, so this adds no dependency.

Why one row per *turn* and not per model call: a turn is the unit a customer
experiences. One turn may be four model calls and three tool calls; averaging
model calls tells you about the machine, averaging turns tells you about the
service.

    CHATBOT_TELEMETRY_DB    path to the database (default: telemetry.db)
    CHATBOT_TELEMETRY       off | 0 | false disables recording entirely

The file contains full conversation transcripts, including any phone digits a
customer typed during verification. It is in .gitignore for that reason;
treat it as credential-bearing and do not commit or ship it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB = "telemetry.db"

# Health thresholds. These are the lines between "fine", "worth a look" and
# "something is wrong", and they are here rather than in the dashboard so the
# API and the page cannot disagree about what amber means.
THRESHOLDS = {
    # Share of turns where any guard fired. Guards catch the model faking tool
    # use, so this rising means the model is degrading, not the app.
    "guard_rate": (0.05, 0.15),
    # Share of turns triage could not route. It is repaired by falling back to
    # the general agent, so customers do not see an error -- which is exactly
    # why it needs watching somewhere.
    "triage_gave_up_rate": (0.02, 0.10),
    # Policy turns answered with no lookup. Should be near zero: the policy
    # agent has no knowledge of its own, and the grounding guard now repairs
    # any answer with neither of its tools behind it. Was (0.35, 0.60) when
    # this was measured over every turn including greetings.
    "ungrounded_rate": (0.05, 0.20),
    # Share of turns the model server could not be reached for.
    "unavailable_rate": (0.01, 0.05),
    # Share of turns where a tool ran and raised.
    "tool_error_rate": (0.02, 0.10),
}


def _enabled() -> bool:
    return os.environ.get("CHATBOT_TELEMETRY", "on").lower() not in (
        "off", "0", "false", "no")


def _db_path() -> str:
    return os.environ.get("CHATBOT_TELEMETRY_DB", DEFAULT_DB)


SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL    NOT NULL,
    iso               TEXT    NOT NULL,
    day               TEXT    NOT NULL,
    hour              TEXT    NOT NULL,
    session_id        TEXT,
    turn_index        INTEGER,

    -- routing
    engine            TEXT,
    agent             TEXT,
    routed_to         TEXT,
    intent            TEXT,
    triage_topic      TEXT,
    answer_source     TEXT,
    triage_attempts   INTEGER DEFAULT 0,
    triage_gave_up    INTEGER DEFAULT 0,

    -- work done
    tools_used        TEXT,
    tool_count        INTEGER DEFAULT 0,
    tool_errors       INTEGER DEFAULT 0,
    wrong_agent_tools INTEGER DEFAULT 0,

    -- behaviour guards
    guards            TEXT,
    guard_count       INTEGER DEFAULT 0,

    -- identity
    verified          INTEGER DEFAULT 0,
    gate_blocks       INTEGER DEFAULT 0,

    -- cost and speed
    latency_ms        REAL,
    model_calls       INTEGER DEFAULT 0,
    input_tokens      INTEGER DEFAULT 0,
    output_tokens     INTEGER DEFAULT 0,
    cache_hits        INTEGER DEFAULT 0,
    cache_misses      INTEGER DEFAULT 0,
    model             TEXT,
    provider          TEXT,

    -- outcome
    ok                INTEGER DEFAULT 1,
    error             TEXT,

    -- content
    message           TEXT,
    reply             TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_ts    ON turns(ts);
CREATE INDEX IF NOT EXISTS idx_turns_day   ON turns(day);
CREATE INDEX IF NOT EXISTS idx_turns_agent ON turns(agent);
"""


@dataclass
class TurnRecord:
    """One customer turn, as it will be stored.

    Built from the response dict rather than assembled by hand at the call
    site, so a new field in the agent's telemetry reaches the database by
    being named here once.
    """
    session_id: Optional[str] = None
    turn_index: int = 0
    engine: Optional[str] = None
    agent: Optional[str] = None
    routed_to: Optional[str] = None
    intent: Optional[str] = None
    triage_topic: Optional[str] = None
    answer_source: Optional[str] = None
    triage_attempts: int = 0
    triage_gave_up: bool = False
    tools_used: List[str] = field(default_factory=list)
    tool_errors: int = 0
    wrong_agent_tools: int = 0
    guards: List[str] = field(default_factory=list)
    verified: bool = False
    gate_blocks: int = 0
    latency_ms: float = 0.0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    model: Optional[str] = None
    provider: Optional[str] = None
    ok: bool = True
    error: Optional[str] = None
    message: Optional[str] = None
    reply: Optional[str] = None

    @classmethod
    def from_response(cls, response: Dict[str, Any], *, session_id: str,
                      turn_index: int, latency_ms: float,
                      message: str, verified: bool,
                      error: Optional[str] = None) -> "TurnRecord":
        stats = response.get("telemetry") or {}
        return cls(
            session_id=session_id,
            turn_index=turn_index,
            engine=response.get("engine"),
            agent=response.get("agent"),
            routed_to=response.get("routed_to"),
            intent=response.get("intent"),
            triage_topic=response.get("triage_topic"),
            answer_source=response.get("answer_source"),
            triage_attempts=stats.get("triage_attempts", 0),
            triage_gave_up=bool(stats.get("triage_gave_up")),
            tools_used=list(response.get("actions_taken") or []),
            tool_errors=stats.get("tool_errors", 0),
            wrong_agent_tools=stats.get("wrong_agent_tools", 0),
            guards=list(stats.get("guards") or []),
            verified=verified,
            gate_blocks=stats.get("gate_blocks", 0),
            latency_ms=latency_ms,
            model_calls=stats.get("model_calls", 0),
            input_tokens=stats.get("input_tokens", 0),
            output_tokens=stats.get("output_tokens", 0),
            cache_hits=stats.get("cache_hits", 0),
            cache_misses=stats.get("cache_misses", 0),
            model=response.get("model"),
            provider=response.get("provider"),
            ok=error is None,
            error=error,
            message=message,
            reply=response.get("response"),
        )


class TelemetryStore:
    """SQLite-backed turn log.

    One connection, guarded by a lock. The app runs a single worker because
    sessions live in process memory, so there is exactly one writer; the lock
    is for the async event loop interleaving two turns, not for processes.

    Every public method swallows its own errors. Telemetry that can take the
    application down with it is worse than no telemetry -- a failed write is
    logged and the customer still gets their answer.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or _db_path()
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self.write_failures = 0
        self._open()

    def _open(self) -> None:
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # WAL lets the dashboard read while a turn is being written.
            self._conn.execute("PRAGMA journal_mode=WAL")
            # The durability of a metrics row is not worth an fsync on the
            # customer's latency path.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        except Exception as exc:                       # pragma: no cover
            logger.error("telemetry store unavailable (%s): %s", self.path, exc)
            self._conn = None

    @property
    def available(self) -> bool:
        return self._conn is not None

    # -- writing ----------------------------------------------------------

    def record(self, turn: TurnRecord) -> None:
        if self._conn is None:
            return
        now = time.time()
        stamp = datetime.fromtimestamp(now, timezone.utc)
        row = {
            "ts": now,
            "iso": stamp.isoformat(),
            "day": stamp.strftime("%Y-%m-%d"),
            "hour": stamp.strftime("%Y-%m-%dT%H:00"),
            "session_id": turn.session_id,
            "turn_index": turn.turn_index,
            "engine": turn.engine,
            "agent": turn.agent,
            "routed_to": turn.routed_to,
            "intent": turn.intent,
            "triage_topic": turn.triage_topic,
            "answer_source": turn.answer_source,
            "triage_attempts": turn.triage_attempts,
            "triage_gave_up": int(turn.triage_gave_up),
            "tools_used": json.dumps(turn.tools_used),
            "tool_count": len(turn.tools_used),
            "tool_errors": turn.tool_errors,
            "wrong_agent_tools": turn.wrong_agent_tools,
            "guards": json.dumps(turn.guards),
            "guard_count": len(turn.guards),
            "verified": int(turn.verified),
            "gate_blocks": turn.gate_blocks,
            "latency_ms": round(turn.latency_ms, 2),
            "model_calls": turn.model_calls,
            "input_tokens": turn.input_tokens,
            "output_tokens": turn.output_tokens,
            "cache_hits": turn.cache_hits,
            "cache_misses": turn.cache_misses,
            "model": turn.model,
            "provider": turn.provider,
            "ok": int(turn.ok),
            "error": turn.error,
            "message": turn.message,
            "reply": turn.reply,
        }
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO turns ({columns}) VALUES ({placeholders})", row)
                self._conn.commit()
        except Exception as exc:                       # pragma: no cover
            self.write_failures += 1
            logger.warning("telemetry write failed: %s", exc)

    # -- reading ----------------------------------------------------------

    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        if self._conn is None:
            return []
        try:
            with self._lock:
                return self._conn.execute(sql, params).fetchall()
        except Exception as exc:                       # pragma: no cover
            logger.warning("telemetry query failed: %s", exc)
            return []

    def total_turns(self) -> int:
        rows = self._query("SELECT COUNT(*) AS n FROM turns")
        return rows[0]["n"] if rows else 0

    def recent(self, limit: int = 50, include_text: bool = True) -> List[dict]:
        """The last N turns, newest first. The transcript view."""
        rows = self._query(
            "SELECT * FROM turns ORDER BY ts DESC LIMIT ?", (limit,))
        out = []
        for row in rows:
            item = dict(row)
            item["tools_used"] = json.loads(item.get("tools_used") or "[]")
            item["guards"] = json.loads(item.get("guards") or "[]")
            if not include_text:
                item.pop("message", None)
                item.pop("reply", None)
            out.append(item)
        return out

    def health(self, hours: int = 24, buckets: str = "hour") -> dict:
        """Health over a window, as one summary plus a time series.

        `buckets` is "hour" or "day" -- the series is what makes a number
        readable: a 6% guard rate is meaningless until you can see whether it
        was 2% last week.
        """
        since = time.time() - hours * 3600
        column = "hour" if buckets == "hour" else "day"

        summary = self._summarise(
            "SELECT %s FROM turns WHERE ts >= ?" % _AGGREGATES, (since,))
        # Real percentiles, not the average wearing a percentile's name. SQLite
        # has no percentile function, so these are separate ordered lookups --
        # cheap, and honest, which the alternative was not.
        summary["p50_ms"] = self.percentile(hours, 0.50)
        summary["p95_ms"] = self.percentile(hours, 0.95)
        summary["p99_ms"] = self.percentile(hours, 0.99)

        series_rows = self._query(
            f"SELECT {column} AS bucket, {_AGGREGATES} FROM turns "
            f"WHERE ts >= ? GROUP BY {column} ORDER BY {column}", (since,))
        series = [self._shape(dict(row), bucket=row["bucket"])
                  for row in series_rows]

        return {
            "window_hours": hours,
            "bucket": column,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "series": series,
            "by_agent": self._by_agent(since),
            "guards": self._guard_breakdown(since),
            "top_intents": self._top_intents(since),
            "thresholds": {k: {"warn": w, "bad": b}
                           for k, (w, b) in THRESHOLDS.items()},
            "verdict": _verdict(summary),
        }

    def _summarise(self, sql: str, params: tuple) -> dict:
        rows = self._query(sql, params)
        return self._shape(dict(rows[0]) if rows else {})

    @staticmethod
    def _shape(row: dict, bucket: Optional[str] = None) -> dict:
        """Turn raw SUMs into the rates a person actually reads."""
        turns = row.get("turns") or 0

        def rate(key: str) -> float:
            return round((row.get(key) or 0) / turns, 4) if turns else 0.0

        out = {
            "turns": turns,
            "guard_rate": rate("guarded_turns"),
            "triage_gave_up_rate": rate("gave_up"),
            # Over POLICY turns only. Measured across everything it read
            # 81.6% and meant nothing: a greeting correctly needs no lookup,
            # and so does the first data turn, where the agent is still asking
            # who it is talking to. A policy turn with no lookup is the one
            # case that is unambiguously wrong -- the policy agent knows
            # nothing except what search_knowledge returns.
            "ungrounded_rate": (
                round((row.get("ungrounded") or 0) / row["policy_turns"], 4)
                if row.get("policy_turns") else 0.0),
            "policy_turns": row.get("policy_turns") or 0,
            "unavailable_rate": rate("unavailable"),
            "tool_error_rate": rate("tool_error_turns"),
            "verified_rate": rate("verified_turns"),
            "gate_blocks": row.get("gate_blocks") or 0,
            "wrong_agent_tools": row.get("wrong_agent_tools") or 0,
            "avg_ms": round(row.get("avg_ms") or 0, 1),
            "max_ms": round(row.get("max_ms") or 0, 1),
            "input_tokens": row.get("input_tokens") or 0,
            "output_tokens": row.get("output_tokens") or 0,
            "tokens_per_turn": round(
                ((row.get("input_tokens") or 0) + (row.get("output_tokens") or 0))
                / turns, 1) if turns else 0.0,
            "model_calls_per_turn": round(
                (row.get("model_calls") or 0) / turns, 2) if turns else 0.0,
            "cache_hit_rate": round(
                (row.get("cache_hits") or 0)
                / ((row.get("cache_hits") or 0) + (row.get("cache_misses") or 0)), 4)
            if ((row.get("cache_hits") or 0) + (row.get("cache_misses") or 0))
            else 0.0,
        }
        if bucket is not None:
            out["bucket"] = bucket
        return out

    def percentile(self, hours: int = 24, fraction: float = 0.95) -> float:
        """Nearest-rank percentile over the window, computed in SQL."""
        since = time.time() - hours * 3600
        rows = self._query(
            "SELECT COUNT(*) AS n FROM turns WHERE ts >= ?", (since,))
        count = rows[0]["n"] if rows else 0
        if not count:
            return 0.0
        offset = min(count - 1, max(0, int(round(fraction * count + 0.5)) - 1))
        rows = self._query(
            "SELECT latency_ms FROM turns WHERE ts >= ? "
            "ORDER BY latency_ms LIMIT 1 OFFSET ?", (since, offset))
        return round(rows[0]["latency_ms"] or 0.0, 1) if rows else 0.0

    def _by_agent(self, since: float) -> List[dict]:
        rows = self._query(
            "SELECT COALESCE(routed_to, agent, 'unknown') AS agent, "
            "COUNT(*) AS turns, AVG(latency_ms) AS avg_ms, "
            "SUM(input_tokens + output_tokens) AS tokens, "
            "SUM(guard_count > 0) AS guarded "
            "FROM turns WHERE ts >= ? GROUP BY 1 ORDER BY turns DESC", (since,))
        return [{
            "agent": row["agent"],
            "turns": row["turns"],
            "avg_ms": round(row["avg_ms"] or 0, 1),
            "tokens": row["tokens"] or 0,
            "guard_rate": round((row["guarded"] or 0) / row["turns"], 4)
            if row["turns"] else 0.0,
        } for row in rows]

    def _guard_breakdown(self, since: float) -> Dict[str, int]:
        """Which guards fired, by name.

        Guards are stored as a JSON list because a turn can trip more than
        one, so they are counted in Python. The row count here is bounded by
        the window, and only guarded turns are read.
        """
        rows = self._query(
            "SELECT guards FROM turns WHERE ts >= ? AND guard_count > 0",
            (since,))
        counts: Dict[str, int] = {}
        for row in rows:
            for name in json.loads(row["guards"] or "[]"):
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def _top_intents(self, since: float, limit: int = 8) -> List[dict]:
        rows = self._query(
            "SELECT COALESCE(intent, 'unknown') AS intent, COUNT(*) AS turns "
            "FROM turns WHERE ts >= ? GROUP BY 1 ORDER BY turns DESC LIMIT ?",
            (since, limit))
        return [{"intent": row["intent"], "turns": row["turns"]} for row in rows]

    def prune(self, keep_days: int = 90) -> int:
        """Drop rows older than the retention window. Returns rows removed."""
        if self._conn is None:
            return 0
        cutoff = time.time() - keep_days * 86400
        try:
            with self._lock:
                cursor = self._conn.execute(
                    "DELETE FROM turns WHERE ts < ?", (cutoff,))
                self._conn.commit()
                return cursor.rowcount
        except Exception as exc:                       # pragma: no cover
            logger.warning("telemetry prune failed: %s", exc)
            return 0

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


# The SUM(...) expressions every aggregate query shares. Defined once so the
# summary row and the per-bucket rows cannot drift into meaning different
# things -- which is exactly how a dashboard ends up lying.
_AGGREGATES = """
    COUNT(*)                            AS turns,
    SUM(guard_count > 0)                AS guarded_turns,
    SUM(triage_gave_up)                 AS gave_up,
    SUM(routed_to = 'policy')                             AS policy_turns,
    SUM(routed_to = 'policy' AND answer_source = 'none')  AS ungrounded,
    SUM(engine = 'unavailable')         AS unavailable,
    SUM(tool_errors > 0)                AS tool_error_turns,
    SUM(verified)                       AS verified_turns,
    SUM(gate_blocks)                    AS gate_blocks,
    SUM(wrong_agent_tools)              AS wrong_agent_tools,
    AVG(latency_ms)                     AS avg_ms,
    MAX(latency_ms)                     AS max_ms,
    SUM(input_tokens)                   AS input_tokens,
    SUM(output_tokens)                  AS output_tokens,
    SUM(model_calls)                    AS model_calls,
    SUM(cache_hits)                     AS cache_hits,
    SUM(cache_misses)                   AS cache_misses
"""


def _verdict(summary: dict) -> dict:
    """One word for the whole window, and why.

    A dashboard of twelve numbers still needs someone to know which one is
    bad. This applies THRESHOLDS and names the worst offender.
    """
    if not summary.get("turns"):
        return {"level": "unknown", "reason": "no turns recorded in this window"}

    worst = "good"
    reasons: List[str] = []
    for key, (warn, bad) in THRESHOLDS.items():
        value = summary.get(key, 0.0)
        if value >= bad:
            worst = "bad"
            reasons.append(f"{key} {value:.1%} (bad above {bad:.0%})")
        elif value >= warn and worst != "bad":
            worst = "warn"
            reasons.append(f"{key} {value:.1%} (warn above {warn:.0%})")
    return {
        "level": worst,
        "reason": "; ".join(reasons) if reasons else "every indicator within range",
    }


_store: Optional[TelemetryStore] = None


def get_store() -> Optional[TelemetryStore]:
    """The process-wide store, or None when telemetry is switched off."""
    global _store
    if not _enabled():
        return None
    if _store is None:
        _store = TelemetryStore()
    return _store


def reset_store() -> None:
    """Drop the cached store. For tests that point at a temp database."""
    global _store
    if _store is not None:
        _store.close()
    _store = None
