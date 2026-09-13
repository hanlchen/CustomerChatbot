"""
Tests for the eval scoreboard.

A scoreboard is only worth having if it is right, and "right" here means two
things: the numbers are computed correctly, and a regression is actually
noticed. A comparison that quietly reports everything as steady is worse than
no comparison, because it will be believed.
"""

import json

import pytest

import eval_metrics as EM


def card(suite="retrieval", **metrics) -> EM.Scorecard:
    sc = EM.Scorecard(suite=suite, model="test-model", provider="test")
    for name, value in metrics.items():
        sc.add(name.replace("__", "."), value, 16)
    return sc


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class TestMetricsAreDeclaredBeforeUse:
    """An undeclared metric cannot be compared, so it is a bug not a number."""

    def test_an_unknown_metric_is_refused(self):
        sc = EM.Scorecard(suite="x")
        with pytest.raises(KeyError, match="not a declared metric"):
            sc.add("retrieval.made_up", 1.0)

    def test_every_declared_metric_has_a_direction_and_a_suite(self):
        for name, spec in EM.METRICS.items():
            assert spec.name == name, f"{name} keyed under the wrong name"
            assert spec.suite, f"{name} has no suite"
            assert spec.description.strip(), f"{name} has no description"
            assert isinstance(spec.higher_is_better, bool)

    def test_safety_metrics_are_absolutes_not_trends(self):
        """A breach must not be gradeable as 'only slightly worse'."""
        for name, spec in EM.METRICS.items():
            if spec.suite == "safety":
                assert spec.must_equal == 0, f"{name} must be pinned to zero"
                assert not spec.higher_is_better


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:

    def test_a_real_drop_is_a_regression(self):
        before = card(retrieval__mrr=0.90)
        after = card(retrieval__mrr=0.60)
        change = EM.compare(after, before)[0]
        assert change.verdict == "regressed"
        assert change.delta == pytest.approx(-0.30)

    def test_a_drop_inside_tolerance_is_steady(self):
        """Small suites are jumpy; crying wolf gets the board ignored."""
        before = card(retrieval__mrr=0.90)
        after = card(retrieval__mrr=0.88)
        assert EM.compare(after, before)[0].verdict == "steady"

    def test_a_rise_is_an_improvement(self):
        change = EM.compare(card(retrieval__mrr=0.95),
                            card(retrieval__mrr=0.60))[0]
        assert change.verdict == "improved"

    def test_direction_is_respected_for_lower_is_better(self):
        """More tokens per turn is worse, not better."""
        before = EM.Scorecard(suite="cost")
        before.add("cost.tokens_per_turn", 400, 10)
        after = EM.Scorecard(suite="cost")
        after.add("cost.tokens_per_turn", 900, 10)
        assert EM.compare(after, before)[0].verdict == "regressed"

        cheaper = EM.Scorecard(suite="cost")
        cheaper.add("cost.tokens_per_turn", 200, 10)
        assert EM.compare(cheaper, before)[0].verdict == "improved"

    def test_magnitude_tolerance_scales_with_the_value(self):
        """5% of 20,000ms and 5% of 0.9 are not the same quantity."""
        before = EM.Scorecard(suite="cost")
        before.add("cost.p95_latency_ms", 20000, 10)
        after = EM.Scorecard(suite="cost")
        after.add("cost.p95_latency_ms", 21000, 10)     # +5%, inside 25%
        assert EM.compare(after, before)[0].verdict == "steady"

        worse = EM.Scorecard(suite="cost")
        worse.add("cost.p95_latency_ms", 40000, 10)     # doubled
        assert EM.compare(worse, before)[0].verdict == "regressed"

    def test_a_new_metric_is_marked_new_not_improved(self):
        change = EM.compare(card(retrieval__mrr=0.9), card())[0]
        assert change.verdict == "new"

    def test_no_baseline_means_everything_is_new(self):
        changes = EM.compare(card(retrieval__mrr=0.9), None)
        assert [c.verdict for c in changes] == ["new"]


class TestSafetyIsJudgedAgainstZero:
    """Not against the trend. 'No worse than last week' is not a safety bar."""

    def test_any_bypass_is_a_breach(self):
        sc = EM.Scorecard(suite="conversation")
        sc.add("safety.gate_bypasses", 1, 40)
        assert EM.compare(sc, None)[0].verdict == "breach"

    def test_a_breach_is_still_a_breach_when_the_baseline_had_one_too(self):
        before = EM.Scorecard(suite="conversation")
        before.add("safety.gate_bypasses", 3, 40)
        after = EM.Scorecard(suite="conversation")
        after.add("safety.gate_bypasses", 1, 40)
        change = EM.compare(after, before)[0]
        assert change.verdict == "breach", \
            "improving from 3 breaches to 1 is not a pass"

    def test_zero_is_steady(self):
        sc = EM.Scorecard(suite="conversation")
        sc.add("safety.isolation_violations", 0, 40)
        assert EM.compare(sc, None)[0].verdict == "steady"

    def test_regressions_include_breaches(self):
        sc = EM.Scorecard(suite="conversation")
        sc.add("safety.gate_bypasses", 2, 40)
        sc.add("conversation.turn_pass_rate", 1.0, 40)
        assert [c.name for c in EM.regressions(EM.compare(sc, None))] == \
            ["safety.gate_bypasses"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_a_saved_run_round_trips(self, tmp_path):
        original = card(retrieval__mrr=0.87)
        original.notes = "lexical-only"
        path = original.save(tmp_path)
        restored = EM.Scorecard.from_dict(json.loads(path.read_text()))
        assert restored.model == original.model
        assert restored.notes == "lexical-only"
        assert restored.get("retrieval.mrr").value == pytest.approx(0.87)

    def test_history_is_oldest_first(self, tmp_path):
        for i, value in enumerate([0.5, 0.7, 0.9]):
            sc = card(retrieval__mrr=value)
            sc.started_at = 1000.0 + i
            sc.save(tmp_path)
        values = [c.get("retrieval.mrr").value for c in EM.history(None, tmp_path)]
        assert values == [0.5, 0.7, 0.9]

    def test_latest_returns_the_newest(self, tmp_path):
        for i, value in enumerate([0.5, 0.9]):
            sc = card(retrieval__mrr=value)
            sc.started_at = 2000.0 + i
            sc.save(tmp_path)
        assert EM.latest("retrieval", tmp_path).get("retrieval.mrr").value == 0.9

    def test_a_corrupt_file_does_not_break_the_history(self, tmp_path):
        """One truncated write must not blind the whole board."""
        good = card(retrieval__mrr=0.8)
        good.save(tmp_path)
        (tmp_path / "20200101T000000_retrieval_broken.json").write_text("{not json")
        assert len(EM.history(None, tmp_path)) == 1

    def test_history_of_an_empty_directory_is_empty(self, tmp_path):
        assert EM.history(None, tmp_path / "nothing") == []


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------

class TestRetrievalScoring:

    def test_it_scores_the_real_corpus(self):
        sc = EM.score_retrieval()
        names = {m.name for m in sc.metrics}
        assert names == {
            "retrieval.hit_rate@1", "retrieval.hit_rate@3", "retrieval.mrr",
            "retrieval.hard.hit_rate@3", "retrieval.hard.mrr",
        }
        for metric in sc.metrics:
            assert 0.0 <= metric.value <= 1.0, f"{metric.name} out of range"

    def test_mrr_is_bounded_by_hit_rate(self):
        """MRR can never exceed hit-rate: a hit at rank 2 contributes 0.5."""
        sc = EM.score_retrieval()
        assert sc.get("retrieval.mrr").value <= sc.get("retrieval.hit_rate@3").value

    def test_hit_rate_at_1_never_exceeds_hit_rate_at_3(self):
        sc = EM.score_retrieval()
        assert (sc.get("retrieval.hit_rate@1").value
                <= sc.get("retrieval.hit_rate@3").value)

    def test_ranking_maths_is_right(self):
        """A hit at rank 2 must score 0.5, not 1.0 -- MRR's whole purpose."""
        class FakeChunk:
            def __init__(self, title):
                self.title, self.section, self.category = title, "", ""

        class FakeHit:
            def __init__(self, title):
                self.chunk = FakeChunk(title)

        class FakeRetriever:
            def search(self, query, top_k=3):
                return [FakeHit("Shipping"), FakeHit("Returns")]

        assert EM._hit_rank(FakeRetriever(), "q", r"returns", 3) == 2
        assert EM._hit_rank(FakeRetriever(), "q", r"nothing", 3) is None

    def test_the_hard_set_is_harder_than_the_core_set(self):
        """If both saturate, the board has no headroom and measures nothing."""
        sc = EM.score_retrieval()
        assert (sc.get("retrieval.hard.mrr").value
                < sc.get("retrieval.mrr").value), (
            "the adversarial set scores as well as the core set -- it is no "
            "longer adversarial, so add harder queries")


class TestBenchmarkIsSingleSourced:
    """Two copies of a question set drift, and then the suite and the board
    are measuring different things while both look green."""

    def test_the_test_suite_imports_the_same_list(self):
        import test_retrieval

        assert test_retrieval.BENCHMARK is EM.BENCHMARK
        assert test_retrieval.HARD_BENCHMARK is EM.HARD_BENCHMARK
