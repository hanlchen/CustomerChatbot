"""
Tests for the persistent turn log and the health numbers computed from it.

The point of this layer is to be trustworthy when someone reads it at 2am and
decides whether the model is broken. So these tests are mostly about the
arithmetic being right and the failure modes being quiet: a rate that is
subtly wrong is worse than no dashboard, and a telemetry write that can take
down a customer's turn is worse than no telemetry.
"""

import json
import os
import sqlite3

import pytest

import telemetry
from telemetry import TelemetryStore, TurnRecord, _verdict


@pytest.fixture
def store(tmp_path):
    """A store on its own file, so tests never touch the real database."""
    db = TelemetryStore(str(tmp_path / "t.db"))
    yield db
    db.close()


def turn(**overrides) -> TurnRecord:
    base = dict(session_id="s1", turn_index=0, engine="llm", agent="policy",
                routed_to="policy", intent="returns", answer_source="knowledge_base",
                tools_used=["search_knowledge"], latency_ms=120.0,
                input_tokens=300, output_tokens=60, model_calls=2,
                model="mock/Qwen3-8B", provider="openai-compatible")
    base.update(overrides)
    return TurnRecord(**base)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class TestItActuallyPersists:
    """The entire reason this module exists."""

    def test_a_turn_survives_a_new_store_on_the_same_file(self, tmp_path):
        path = str(tmp_path / "t.db")
        first = TelemetryStore(path)
        first.record(turn())
        first.close()

        # A new process, in every way that matters here.
        second = TelemetryStore(path)
        assert second.total_turns() == 1
        assert second.recent()[0]["intent"] == "returns"
        second.close()

    def test_lists_round_trip_as_lists_not_strings(self, store):
        store.record(turn(tools_used=["search_knowledge", "list_policy_topics"],
                          guards=["narration"]))
        row = store.recent()[0]
        assert row["tools_used"] == ["search_knowledge", "list_policy_topics"]
        assert row["guards"] == ["narration"]
        assert row["tool_count"] == 2
        assert row["guard_count"] == 1

    def test_recent_is_newest_first(self, store):
        for i in range(3):
            store.record(turn(intent=f"i{i}"))
        assert [r["intent"] for r in store.recent()] == ["i2", "i1", "i0"]

    def test_text_can_be_withheld_from_a_caller(self, store):
        store.record(turn(message="where is my order", reply="let me look"))
        assert store.recent(include_text=False)[0].get("message") is None
        assert store.recent(include_text=True)[0]["message"] == "where is my order"


class TestTelemetryNeverBreaksATurn:
    """A metrics layer that can take the app down is a liability, not an asset.

    Every path here is one that would otherwise raise inside the request
    handler, after the model has already been paid for and the customer is
    waiting on an answer.
    """

    def test_an_unwritable_database_does_not_raise(self, tmp_path):
        # A directory where the file should be: open fails, and must do so
        # quietly.
        path = tmp_path / "cannot"
        path.mkdir()
        store = TelemetryStore(str(path))
        assert not store.available
        store.record(turn())          # must not raise
        assert store.total_turns() == 0
        assert store.health(hours=1)["summary"]["turns"] == 0

    def test_a_write_failure_is_counted_not_raised(self, store):
        store._conn.close()           # simulate the file vanishing underneath
        store.record(turn())
        assert store.write_failures == 1

    def test_a_query_failure_returns_empty(self, store):
        store._conn.close()
        assert store.recent() == []
        assert store.total_turns() == 0

    def test_switching_it_off_yields_no_store(self, monkeypatch):
        monkeypatch.setenv("CHATBOT_TELEMETRY", "off")
        telemetry.reset_store()
        assert telemetry.get_store() is None
        telemetry.reset_store()


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------

class TestTheRatesAreRight:
    """Arithmetic, checked against counts a person can do in their head.

    A dashboard rate that is quietly wrong is worse than no dashboard: it is
    believed.
    """

    def test_guard_rate_counts_turns_not_guards(self, store):
        # One turn with two guards is one guarded turn, not two.
        store.record(turn(guards=["narration", "ungrounded_fact"]))
        store.record(turn())
        store.record(turn())
        store.record(turn())
        assert store.health(hours=1)["summary"]["guard_rate"] == 0.25

    def test_guard_breakdown_counts_every_firing(self, store):
        store.record(turn(guards=["narration", "ungrounded_fact"]))
        store.record(turn(guards=["narration"]))
        assert store.health(hours=1)["guards"] == {
            "narration": 2, "ungrounded_fact": 1}

    def test_ungrounded_means_no_tool_behind_the_answer(self, store):
        store.record(turn(answer_source="none", tools_used=[]))
        store.record(turn(answer_source="knowledge_base"))
        assert store.health(hours=1)["summary"]["ungrounded_rate"] == 0.5

    def test_turns_that_need_no_lookup_are_not_counted_against_it(self, store):
        """A greeting correctly needs no tool. Counting it made the metric lie.

        Measured over every turn this read 81.6% on real traffic and meant
        nothing: greetings, and the first data turn where the agent is still
        asking who it is talking to, both legitimately answer with no lookup.
        A POLICY turn with no lookup is the one unambiguous failure -- that
        agent knows nothing except what search_knowledge returns.
        """
        for _ in range(10):                       # greetings: no lookup needed
            store.record(turn(routed_to="general", answer_source="none",
                              tools_used=[]))
        for _ in range(4):                        # data asking for an id
            store.record(turn(routed_to="data", answer_source="none",
                              tools_used=[]))
        for _ in range(3):                        # policy that searched
            store.record(turn(routed_to="policy",
                              answer_source="knowledge_base"))
        store.record(turn(routed_to="policy",     # policy that did not
                          answer_source="none", tools_used=[]))

        summary = store.health(hours=1)["summary"]
        assert summary["turns"] == 18
        assert summary["policy_turns"] == 4
        assert summary["ungrounded_rate"] == 0.25, (
            "must be 1 of 4 policy turns, not 15 of 18 overall")

    def test_no_policy_turns_means_no_ungrounded_rate(self, store):
        """Not a division by zero, and not a misleading 100%."""
        store.record(turn(routed_to="general", answer_source="none",
                          tools_used=[]))
        assert store.health(hours=1)["summary"]["ungrounded_rate"] == 0.0

    def test_rates_are_zero_not_an_error_with_no_turns(self, store):
        summary = store.health(hours=1)["summary"]
        assert summary["turns"] == 0
        assert summary["guard_rate"] == 0.0
        assert summary["tokens_per_turn"] == 0.0

    def test_tokens_per_turn_covers_both_directions(self, store):
        store.record(turn(input_tokens=100, output_tokens=50))
        store.record(turn(input_tokens=200, output_tokens=50))
        assert store.health(hours=1)["summary"]["tokens_per_turn"] == 200.0

    def test_cache_hit_rate_is_hits_over_lookups(self, store):
        store.record(turn(cache_hits=3, cache_misses=1))
        assert store.health(hours=1)["summary"]["cache_hit_rate"] == 0.75

    def test_percentiles_are_percentiles_and_not_the_mean(self, store):
        """The bug this pins: labelling AVG(latency_ms) as p50.

        With one huge outlier the mean and the median are far apart, so a
        p50 that is really a mean is caught here rather than on a dashboard.
        """
        for value in [10, 10, 10, 10, 1000]:
            store.record(turn(latency_ms=value))
        summary = store.health(hours=1)["summary"]
        assert summary["p50_ms"] == 10.0, "p50 must be the median, not the mean"
        assert summary["avg_ms"] == 208.0
        assert summary["p95_ms"] == 1000.0

    def test_a_window_excludes_what_is_outside_it(self, store):
        import time
        store.record(turn())
        # Age one row past the window by hand.
        store._conn.execute("UPDATE turns SET ts = ?", (time.time() - 7200,))
        store._conn.commit()
        assert store.health(hours=1)["summary"]["turns"] == 0
        assert store.health(hours=24)["summary"]["turns"] == 1


class TestTheSeriesAndTheSummaryAgree:
    """Two code paths compute the same aggregates; they must not diverge.

    They share `_AGGREGATES` for exactly this reason -- if someone splits
    them, a bucket total will stop matching the headline number and the page
    will contradict itself.
    """

    def test_bucket_turns_sum_to_the_summary_total(self, store):
        for _ in range(7):
            store.record(turn())
        report = store.health(hours=24, buckets="hour")
        assert sum(b["turns"] for b in report["series"]) == report["summary"]["turns"]

    def test_every_bucket_carries_the_same_keys_as_the_summary(self, store):
        store.record(turn())
        report = store.health(hours=24)
        for bucket in report["series"]:
            missing = set(report["summary"]) - set(bucket) - {
                "p50_ms", "p95_ms", "p99_ms"}
            assert not missing, f"bucket is missing {missing}"


class TestTheVerdict:
    """One word for the window. It is what someone reads first."""

    def test_no_turns_is_unknown_not_healthy(self):
        assert _verdict({"turns": 0})["level"] == "unknown"

    def test_clean_numbers_read_as_good(self):
        assert _verdict({"turns": 100, "guard_rate": 0.0,
                         "triage_gave_up_rate": 0.0, "ungrounded_rate": 0.02,
                         "unavailable_rate": 0.0, "tool_error_rate": 0.0,
                         })["level"] == "good"

    def test_one_bad_indicator_makes_the_whole_window_bad(self):
        verdict = _verdict({"turns": 100, "guard_rate": 0.9,
                            "triage_gave_up_rate": 0.0, "ungrounded_rate": 0.0,
                            "unavailable_rate": 0.0, "tool_error_rate": 0.0})
        assert verdict["level"] == "bad"
        assert "guard_rate" in verdict["reason"]

    def test_the_reason_names_the_number(self):
        """A verdict you cannot act on is decoration."""
        verdict = _verdict({"turns": 100, "guard_rate": 0.07,
                            "triage_gave_up_rate": 0.0, "ungrounded_rate": 0.0,
                            "unavailable_rate": 0.0, "tool_error_rate": 0.0})
        assert verdict["level"] == "warn"
        assert "guard_rate" in verdict["reason"] and "7.0%" in verdict["reason"]


# ---------------------------------------------------------------------------
# The bridge from a response to a row
# ---------------------------------------------------------------------------

class TestTurnRecordFromResponse:

    def test_it_reads_the_agent_telemetry_block(self):
        record = TurnRecord.from_response(
            {"agent": "policy", "routed_to": "policy", "engine": "llm",
             "intent": "returns", "answer_source": "knowledge_base",
             "actions_taken": ["search_knowledge"], "response": "sure",
             "telemetry": {"input_tokens": 10, "output_tokens": 5,
                           "model_calls": 2, "guards": ["narration"],
                           "gate_blocks": 1, "triage_attempts": 1,
                           "triage_gave_up": False, "tool_errors": 0,
                           "wrong_agent_tools": 0,
                           "cache_hits": 1, "cache_misses": 0}},
            session_id="s", turn_index=3, latency_ms=99.0,
            message="hello", verified=True)
        assert record.guards == ["narration"]
        assert record.gate_blocks == 1
        assert record.input_tokens == 10
        assert record.verified is True
        assert record.reply == "sure"

    def test_a_response_with_no_telemetry_block_still_records(self):
        """The unavailable path returns no telemetry; it must still be logged.

        Turns the model could not answer are the ones you most want counted.
        """
        record = TurnRecord.from_response(
            {"engine": "unavailable", "answer_source": "none",
             "response": "I can't reach our systems"},
            session_id="s", turn_index=0, latency_ms=5.0,
            message="hi", verified=False)
        assert record.engine == "unavailable"
        assert record.guards == []
        assert record.input_tokens == 0


class TestRetention:

    def test_prune_drops_only_what_is_older_than_the_window(self, store):
        import time
        store.record(turn())
        store.record(turn())
        store._conn.execute("UPDATE turns SET ts = ? WHERE id = 1",
                            (time.time() - 200 * 86400,))
        store._conn.commit()
        assert store.prune(keep_days=90) == 1
        assert store.total_turns() == 1


class TestTheSchemaMatchesWhatIsWritten:
    """A column that exists but is never populated is a dashboard bug waiting.

    This pins that every column the writer names is a column the table has --
    the failure otherwise is an sqlite3.OperationalError inside a request.
    """

    def test_every_written_column_exists(self, store):
        store.record(turn())
        columns = {row[1] for row in
                   store._conn.execute("PRAGMA table_info(turns)")}
        row = store.recent()[0]
        assert set(row) - {"tools_used", "guards"} <= columns
