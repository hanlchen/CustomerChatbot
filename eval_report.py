#!/usr/bin/env python3
"""
The eval scoreboard: score a run, store it, compare it to the last one.

Without this, every eval in the project answered "did it pass" and forgot the
answer. The question that decides whether a change was an improvement -- *is
this better than last time?* -- had no data behind it.

    python eval_report.py                    # score retrieval, save, compare
    python eval_report.py --no-save          # score without recording
    python eval_report.py --history          # every run, oldest first
    python eval_report.py --compare-only     # last two runs, no new scoring
    python eval_report.py --fail-on-regression   # exit 1 if anything dropped

Retrieval needs no model, so the default run is safe in CI on every push.
`simulate_customer.py --scorecard` writes its own card into the same
directory when you have a model up.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import eval_metrics as EM


class S:
    on = sys.stdout.isatty()

    @staticmethod
    def _w(code, text):
        return f"\033[{code}m{text}\033[0m" if S.on else text

    @staticmethod
    def bold(t): return S._w("1", t)
    @staticmethod
    def dim(t): return S._w("2", t)
    @staticmethod
    def red(t): return S._w("31", t)
    @staticmethod
    def green(t): return S._w("32", t)
    @staticmethod
    def yellow(t): return S._w("33", t)
    @staticmethod
    def cyan(t): return S._w("36", t)


MARK = {
    "improved":  ("▲", S.green),
    "regressed": ("▼", S.red),
    "breach":    ("✗", S.red),
    "steady":    ("·", S.dim),
    "new":       ("+", S.cyan),
}


def show_card(card: EM.Scorecard) -> None:
    stamp = card.as_dict()["iso"][:19].replace("T", " ")
    print(S.bold(f"\n{card.suite}  ·  {card.model}"))
    print(S.dim(f"  {stamp}Z · {card.git_sha}"
                + (f" · {card.notes}" if card.notes else "")))
    print()
    for metric in card.metrics:
        spec = metric.spec
        line = f"    {metric.name:30} {metric.format():>8}"
        if metric.n:
            line += S.dim(f"  n={metric.n}")
        print(line)
        if metric.detail:
            print(S.dim(f"        {metric.detail}"))
        if spec and spec.must_equal is not None and metric.value != spec.must_equal:
            print(S.red(f"        must be {spec.must_equal:g} — this is a breach"))


def show_comparison(changes: List[EM.Change],
                    baseline: Optional[EM.Scorecard]) -> None:
    if baseline is None:
        print(S.dim("\n  No earlier run to compare against — this is the "
                    "baseline. Run it again after a change."))
        return

    stamp = baseline.as_dict()["iso"][:19].replace("T", " ")
    print(S.bold(f"\n  vs {stamp}Z ({baseline.model}, {baseline.git_sha})"))
    print()
    for change in changes:
        glyph, colour = MARK[change.verdict]
        spec = EM.METRICS.get(change.name)
        unit = spec.unit if spec else "rate"
        fmt = (lambda v: f"{v:.1%}") if unit == "rate" else (lambda v: f"{v:,.0f}")
        if change.verdict in ("steady", "new"):
            detail = fmt(change.current)
        else:
            detail = f"{fmt(change.baseline)} → {fmt(change.current)}"
            if unit == "rate":
                detail += f"  ({change.delta:+.1%})"
        print(f"    {colour(glyph)} {change.name:30} {detail}")


def verdict_line(changes: List[EM.Change]) -> int:
    bad = EM.regressions(changes)
    breaches = [c for c in bad if c.verdict == "breach"]
    print()
    if breaches:
        print(S.red(S.bold(
            f"  BREACH — {len(breaches)} safety metric(s) are not at zero.")))
        print("  These are not trends. Stop and fix them.")
        return 2
    if bad:
        print(S.yellow(S.bold(f"  {len(bad)} regression(s).")))
        print(S.dim("  A drop inside a metric's tolerance is reported as "
                    "steady; these are outside it."))
        return 1
    print(S.green(S.bold("  No regressions.")))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-save", action="store_true",
                        help="Score without recording the run.")
    parser.add_argument("--history", action="store_true",
                        help="List every recorded run and exit.")
    parser.add_argument("--compare-only", action="store_true",
                        help="Compare the last two recorded runs; score nothing.")
    parser.add_argument("--suite", default="retrieval",
                        help="Which suite to compare against (default: retrieval).")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit non-zero when a metric regresses. For CI.")
    args = parser.parse_args()

    if args.history:
        cards = EM.history(None if args.suite == "all" else args.suite)
        if not cards:
            print("No runs recorded yet.")
            return 0

        # One table per suite. Mixing them put `retrieval.mrr` and
        # `retrieval.hard.mrr` in adjacent columns both labelled "mrr", which
        # is a table that actively misleads.
        suites: dict = {}
        for card in cards:
            suites.setdefault(card.suite, []).append(card)

        print(S.bold(f"\n{len(cards)} run(s) across {len(suites)} suite(s)"))
        for suite, group in sorted(suites.items()):
            names = sorted({m.name for c in group for m in c.metrics})
            # Drop the suite prefix, keep everything that distinguishes them.
            labels = [n.split(".", 1)[1] if "." in n else n for n in names]
            width = max(13, max((len(l) for l in labels), default=13) + 2)
            print(S.bold(f"\n  {suite}"))
            print(S.dim("    " + f"{'date':17}{'model':22}"
                        + "".join(f"{l:>{width}}" for l in labels)))
            for card in group:
                stamp = card.as_dict()["iso"][:16].replace("T", " ")
                row = f"    {stamp:17}{card.model[:20]:22}"
                for name in names:
                    metric = card.get(name)
                    row += f"{(metric.format() if metric else '—'):>{width}}"
                print(row)
        print()
        return 0

    if args.compare_only:
        cards = EM.history(args.suite)
        if len(cards) < 2:
            print(f"Need two {args.suite} runs to compare; have {len(cards)}.")
            return 0
        current, baseline = cards[-1], cards[-2]
        show_card(current)
        show_comparison(EM.compare(current, baseline), baseline)
        code = verdict_line(EM.compare(current, baseline))
        return code if args.fail_on_regression else 0

    # Default: score retrieval now.
    card = EM.score_retrieval()
    baseline = EM.latest("retrieval")
    show_card(card)
    changes = EM.compare(card, baseline)
    show_comparison(changes, baseline)
    code = verdict_line(changes)

    if not args.no_save:
        path = card.save()
        print(S.dim(f"\n  recorded → {path}"))

    return code if args.fail_on_regression else 0


if __name__ == "__main__":
    sys.exit(main())
