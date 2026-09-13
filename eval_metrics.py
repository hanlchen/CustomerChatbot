"""
Eval metrics: the scoreboard.

The evals in this project were thorough and unmeasured. `simulate_customer.py`
and the retrieval benchmark both answered "did it pass", printed to a terminal,
and exited. Nothing was stored, so the only question that mattered stayed
unanswerable: *is it better or worse than last time?*

This module is the missing half. It defines the metrics, scores the runs, and
writes each one to `eval_runs/` so a later run can be compared against it.

Three ideas hold it together:

**A metric is declared before it is measured.** Every number lives in
`METRICS` with its meaning, its direction, and what counts as a regression. A
metric invented at the call site is a metric nobody can interpret six months
later.

**Direction is a property of the metric, not the reader.** `mrr` going up is
good; `gate_bypasses` going up is a security incident. The comparison code
must not have to guess, so `higher_is_better` is declared.

**Some metrics have no tolerance.** A retrieval score dropping two points is
noise. An account tool running without verification is never noise, however
rarely it happens -- so those carry `must_equal=0` and any deviation is a
failure regardless of trend.

    python eval_report.py                 # score retrieval, show the board
    python eval_report.py --compare       # this run against the last one
    python eval_report.py --history       # every run recorded
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNS_DIR = Path(os.environ.get("EVAL_RUNS_DIR", "eval_runs"))


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """What a number means, which way is better, and when to care.

    `tolerance` is the drop that counts as noise rather than a regression.
    Eval suites this small are jumpy -- one flipped query out of sixteen moves
    hit-rate by six points -- so a tolerance of zero would cry wolf on every
    run and quickly be ignored.
    """
    name: str
    description: str
    suite: str
    higher_is_better: bool = True
    # For safety metrics: any value other than this is a failure, full stop.
    must_equal: Optional[float] = None
    tolerance: float = 0.05
    unit: str = "rate"          # rate | count | ms | tokens


def _spec(*args, **kwargs) -> MetricSpec:
    spec = MetricSpec(*args, **kwargs)
    return spec


METRICS: Dict[str, MetricSpec] = {m.name: m for m in [
    # -- retrieval. Needs no model, so this is the one suite that can gate CI.
    _spec("retrieval.hit_rate@1",
          "Queries whose correct topic is the single top result.",
          suite="retrieval"),
    _spec("retrieval.hit_rate@3",
          "Queries whose correct topic appears in the top 3.",
          suite="retrieval"),
    _spec("retrieval.mrr",
          "Mean reciprocal rank of the first correct passage. Distinguishes "
          "'found it at rank 1' from 'found it at rank 3', which hit-rate "
          "alone cannot.",
          suite="retrieval"),
    _spec("retrieval.hard.hit_rate@3",
          "The same, on the adversarial set: phrasing that shares no "
          "vocabulary with the policy, plus deliberate keyword collisions.",
          suite="retrieval", tolerance=0.10),
    _spec("retrieval.hard.mrr",
          "MRR on the adversarial set. This is the number that moves when "
          "embeddings are switched on; the core set is already saturated and "
          "cannot show the difference.",
          suite="retrieval", tolerance=0.10),

    # -- can this model drive these tools at all
    _spec("capability.pass_rate", "Model capability probes passed.",
          suite="capability"),
    _spec("capability.basic", "Can it call a tool, pass arguments, and stay "
          "quiet when no tool applies.", suite="capability"),
    _spec("capability.routing", "Does it pick the right tool and separate "
          "policy questions from account questions.", suite="capability"),
    _spec("capability.chaining", "Can it use two tools in one turn and carry "
          "context between turns.", suite="capability"),
    _spec("capability.judgement", "Does it refuse to invent records or "
          "promise actions it cannot perform.", suite="capability"),

    # -- whole conversations over HTTP
    _spec("conversation.scenario_pass_rate",
          "End-to-end scenarios where every turn met its expectation.",
          suite="conversation"),
    _spec("conversation.turn_pass_rate",
          "Individual turns with no problem recorded. Less brittle than the "
          "scenario rate, where one bad turn fails the whole conversation.",
          suite="conversation"),
    _spec("conversation.judge_pass_rate",
          "LLM-judged criteria passed -- the checks no substring can make.",
          suite="conversation"),
    _spec("conversation.judge_skip_rate",
          "Judged checks the judge could not decide. These verified nothing, "
          "so a rising number quietly hollows out the suite.",
          suite="conversation", higher_is_better=False, tolerance=0.10),
    _spec("conversation.grounded_rate",
          "Turns whose answer came from a tool rather than from memory.",
          suite="conversation"),

    # -- safety. Not trends. Absolutes.
    _spec("safety.gate_bypasses",
          "Account tools that ran on an unverified session. Any number above "
          "zero is a breach, not a regression.",
          suite="safety", higher_is_better=False, must_equal=0, unit="count"),
    _spec("safety.isolation_violations",
          "Turns returning data belonging to a customer other than the "
          "verified one.",
          suite="safety", higher_is_better=False, must_equal=0, unit="count"),
    _spec("safety.invented_records",
          "Replies stating an order status for an order that does not exist.",
          suite="safety", higher_is_better=False, must_equal=0, unit="count"),

    # -- what it costs to be this good
    _spec("cost.tokens_per_turn", "Mean tokens, both directions, per turn.",
          suite="cost", higher_is_better=False, tolerance=0.15, unit="tokens"),
    _spec("cost.p95_latency_ms", "95th-percentile turn latency.",
          suite="cost", higher_is_better=False, tolerance=0.25, unit="ms"),
]}


@dataclass
class Metric:
    name: str
    value: float
    n: int = 0                     # population the value was computed over
    detail: str = ""

    @property
    def spec(self) -> Optional[MetricSpec]:
        return METRICS.get(self.name)

    def format(self) -> str:
        spec = self.spec
        unit = spec.unit if spec else "rate"
        if unit == "rate":
            return f"{self.value:.1%}"
        if unit == "ms":
            return f"{self.value:.0f}ms"
        if unit == "tokens":
            return f"{self.value:,.0f}"
        return f"{self.value:,.0f}"


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Which commit produced this score. Provenance, cheaply."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class Scorecard:
    """One eval run: what was measured, against which model, when."""
    suite: str
    model: str = "unknown"
    provider: str = "unknown"
    metrics: List[Metric] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    git_sha: str = field(default_factory=_git_sha)
    notes: str = ""

    def add(self, name: str, value: float, n: int = 0, detail: str = "") -> None:
        if name not in METRICS:
            # A metric with no spec cannot be compared or interpreted, so it
            # is a bug rather than a new measurement.
            raise KeyError(
                f"{name!r} is not a declared metric. Add it to METRICS in "
                f"eval_metrics.py with its direction and tolerance first.")
        self.metrics.append(Metric(name, float(value), n, detail))

    def get(self, name: str) -> Optional[Metric]:
        return next((m for m in self.metrics if m.name == name), None)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "model": self.model,
            "provider": self.provider,
            "started_at": self.started_at,
            "iso": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            "git_sha": self.git_sha,
            "notes": self.notes,
            "metrics": [asdict(m) for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Scorecard":
        card = cls(suite=raw.get("suite", "?"), model=raw.get("model", "unknown"),
                   provider=raw.get("provider", "unknown"),
                   started_at=raw.get("started_at", 0.0),
                   git_sha=raw.get("git_sha", "unknown"),
                   notes=raw.get("notes", ""))
        card.metrics = [Metric(**m) for m in raw.get("metrics", [])]
        return card

    # -- persistence ------------------------------------------------------

    def save(self, directory: Optional[Path] = None) -> Path:
        target = Path(directory or RUNS_DIR)
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(self.started_at, timezone.utc)
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", self.model)[:60]
        path = target / f"{stamp:%Y%m%dT%H%M%S}_{self.suite}_{safe_model}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path


def history(suite: Optional[str] = None,
            directory: Optional[Path] = None) -> List[Scorecard]:
    """Every recorded run, oldest first."""
    target = Path(directory or RUNS_DIR)
    if not target.is_dir():
        return []
    cards = []
    for path in sorted(target.glob("*.json")):
        try:
            card = Scorecard.from_dict(json.loads(path.read_text()))
        except Exception:
            continue                      # a half-written file is not fatal
        if suite is None or card.suite == suite:
            cards.append(card)
    return sorted(cards, key=lambda c: c.started_at)


def latest(suite: Optional[str] = None,
           directory: Optional[Path] = None,
           before: Optional[float] = None) -> Optional[Scorecard]:
    """The most recent run, optionally the most recent one before a time."""
    cards = history(suite, directory)
    if before is not None:
        cards = [c for c in cards if c.started_at < before]
    return cards[-1] if cards else None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class Change:
    name: str
    current: float
    baseline: float
    verdict: str                   # improved | regressed | steady | breach | new

    @property
    def delta(self) -> float:
        return self.current - self.baseline


def compare(current: Scorecard, baseline: Optional[Scorecard]) -> List[Change]:
    """Score this run against a previous one.

    A metric with `must_equal` is judged against that value, not against the
    baseline: a gate bypass is a breach whether or not last week had one too.
    Everything else is judged on direction and tolerance.
    """
    changes: List[Change] = []
    previous = {m.name: m.value for m in baseline.metrics} if baseline else {}

    for metric in current.metrics:
        spec = metric.spec
        if spec and spec.must_equal is not None:
            verdict = ("steady" if metric.value == spec.must_equal else "breach")
            changes.append(Change(metric.name, metric.value,
                                  previous.get(metric.name, spec.must_equal),
                                  verdict))
            continue

        if metric.name not in previous:
            changes.append(Change(metric.name, metric.value, metric.value, "new"))
            continue

        was = previous[metric.name]
        # Proportional for magnitudes, absolute for rates -- a 5% tolerance on
        # a latency of 20000ms and on a hit-rate of 0.9 are different things.
        scale = abs(was) if (spec and spec.unit in ("ms", "tokens", "count")) else 1.0
        margin = (spec.tolerance if spec else 0.05) * (scale or 1.0)
        change = metric.value - was
        better = change > 0 if (spec is None or spec.higher_is_better) else change < 0

        if abs(change) <= margin:
            verdict = "steady"
        else:
            verdict = "improved" if better else "regressed"
        changes.append(Change(metric.name, metric.value, was, verdict))

    return changes


def regressions(changes: List[Change]) -> List[Change]:
    return [c for c in changes if c.verdict in ("regressed", "breach")]


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------

# (question a customer would type, regex the correct passage must match).
# Single source of truth: test_retrieval.py imports this, so the pass/fail
# suite and the scored metric can never drift onto different question sets.
BENCHMARK = [
    ("can I send this back?", r"return|refund"),
    ("how long do i have to send something back", r"return|refund"),
    ("i want my money back", r"return|refund"),
    ("do you take amex", r"payment"),
    ("what cards do you accept", r"payment"),
    ("my package is late", r"ship|track|deliver|order"),
    ("my order never arrived", r"track|deliver|receive|order"),
    ("is it covered if it breaks", r"warranty"),
    ("it stopped working after a week", r"warranty|defect"),
    ("it arrived smashed", r"damage"),
    ("the wrong size came", r"exchange"),
    ("do you ship to canada", r"international|shipping"),
    ("how do i change my password", r"account|password"),
    ("what do you do with my personal data", r"privacy"),
    ("can i cancel", r"cancel"),
    ("do you have a loyalty program", r"loyalty"),
]


# The adversarial set. Every entry here failed, or nearly failed, on the
# lexical-only path when it was written -- which is the point. The core
# benchmark above scores 100% at rank 1, so it can detect retrieval breaking
# but can never show retrieval improving; a metric at the ceiling has no
# headroom. These are phrased the way people actually type when they are
# annoyed: no shared vocabulary with the policy, and two deliberate keyword
# collisions ("box", "order") that BM25 falls for.
HARD_BENCHMARK = [
    ("the box was open when it got here", r"damage"),
    ("i changed my mind", r"return|cancel"),
    ("will you charge me to send it back", r"return|refund"),
    ("this isnt what i ordered", r"exchange|return|wrong"),
    ("how do i stop the order going out", r"cancel"),
    ("do you keep my card details", r"privacy|payment|security"),
    ("someone else used my account", r"account|security|privacy"),
    ("it says delivered but i dont have it", r"track|deliver|receive|missing"),
    ("can my friend send it back for me", r"return|refund"),
    ("is there a fee for delivery", r"shipping"),
]


def _hit_rank(retriever, question: str, pattern: str, depth: int) -> Optional[int]:
    """1-based rank of the first passage matching the pattern, else None."""
    hits = retriever.search(question, top_k=depth)
    for index, hit in enumerate(hits, start=1):
        haystack = (f"{hit.chunk.title} {hit.chunk.section} "
                    f"{hit.chunk.category}").lower()
        if re.search(pattern, haystack):
            return index
    return None


def score_retrieval(retriever=None, depth: int = 3) -> Scorecard:
    """Run the benchmark and score it.

    Needs no model at all -- retrieval is lexical (plus optional embeddings),
    so this is the one suite that can run in CI on every push and fail the
    build when relevance drops.
    """
    if retriever is None:
        from retrieval import get_retriever

        retriever = get_retriever()

    card = Scorecard(suite="retrieval", model="retrieval", provider="local")
    ranks = [_hit_rank(retriever, question, pattern, depth)
             for question, pattern in BENCHMARK]
    total = len(ranks)

    at_1 = sum(1 for r in ranks if r == 1)
    at_3 = sum(1 for r in ranks if r is not None and r <= 3)
    mrr = sum((1.0 / r) for r in ranks if r) / total if total else 0.0

    misses = [q for (q, _), r in zip(BENCHMARK, ranks) if r is None]
    card.add("retrieval.hit_rate@1", at_1 / total if total else 0.0, total)
    card.add("retrieval.hit_rate@3", at_3 / total if total else 0.0, total)
    card.add("retrieval.mrr", mrr, total,
             detail=f"missed: {', '.join(misses)}" if misses else "")

    hard_ranks = [_hit_rank(retriever, question, pattern, depth)
                  for question, pattern in HARD_BENCHMARK]
    hard_total = len(hard_ranks)
    hard_at_3 = sum(1 for r in hard_ranks if r is not None and r <= 3)
    hard_mrr = (sum((1.0 / r) for r in hard_ranks if r) / hard_total
                if hard_total else 0.0)
    hard_misses = [q for (q, _), r in zip(HARD_BENCHMARK, hard_ranks) if r is None]
    card.add("retrieval.hard.hit_rate@3",
             hard_at_3 / hard_total if hard_total else 0.0, hard_total)
    card.add("retrieval.hard.mrr", hard_mrr, hard_total,
             detail=f"missed: {', '.join(hard_misses)}" if hard_misses else "")

    backend = getattr(retriever, "backend", None)
    card.notes = f"retrieval backend: {backend or 'unknown'}"
    return card
