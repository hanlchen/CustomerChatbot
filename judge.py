"""
An LLM judge for the parts of a reply that substring matching cannot check.

Why this exists
---------------
`contains("customer id")` is a bad proxy for "did it ask the customer to
identify themselves". It fails a correct answer phrased differently ("which
account is this?") and passes a wrong one that happens to contain the magic
words. Half the simulator's failures were that: the bot was right and the
assertion was too literal.

What it is *not* for
--------------------
Facts and security stay deterministic, and always will:

    did a tool actually run?              did verification get enforced?
    did another customer's data appear?   are the buttons resolvable?
    is there a "None" in the text?        did the intent label match?

Those are checks with a right answer, and a judge would only make them slower,
non-deterministic and occasionally wrong. A judge is for the questions that are
genuinely judgement: did this reply answer what was asked, given everything
said before it.

On self-grading
---------------
By default the judge runs on whatever model the app runs on, which means the
model grades its own homework -- it is lenient about its own mistakes, and a
model that misunderstands a question will usually think its answer to the
wrong question was fine. Point JUDGE_BASE_URL / JUDGE_MODEL at something
larger (or set ANTHROPIC_API_KEY) and the grading becomes worth more. The
report says which of the two you got.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JUDGE_MAX_TOKENS = 160


@dataclass(frozen=True)
class Criterion:
    """One thing a reply has to achieve, in plain language.

    Written as what a reasonable support supervisor would want, not as a
    string to find: "asks the customer to identify themselves, by customer ID
    or email" rather than `contains("customer id")`.
    """
    requirement: str


JUDGE_PROMPT = """You are reviewing a customer support conversation.

You will be shown the conversation so far, the customer's latest message, and
what the support bot replied. You will be given ONE requirement that the reply
had to meet.

Decide whether the reply meets it.

Judge the reply as a reasonable support supervisor would:

- Meaning, not wording. If the requirement is "asks the customer to identify
  themselves" then "which account is this?" meets it, and so does "what's your
  customer ID or the email on the account?". The exact words do not matter.
- In context. A reply that would be wrong out of nowhere may be exactly right
  as a follow-up to what was already said.
- Only this requirement. If the reply is unhelpful in some other way, that is
  not your problem right now. If it meets the requirement, it passes.
- Be fair, not generous. A reply that dodges the requirement while sounding
  pleasant does not pass. Neither does one that promises to do the thing
  instead of doing it.

Answer in exactly this shape, and nothing else:

VERDICT: PASS
REASON: one short sentence

or

VERDICT: FAIL
REASON: one short sentence saying what was missing"""


class Judge:
    """Grades one reply against one criterion."""

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else self._resolve()
        self.calls = 0
        self.failures = 0

    # -- setup -------------------------------------------------------------

    @staticmethod
    def _resolve():
        """A separate judge model if one is configured, else the app's own."""
        from llm_providers import resolve_backend

        judge_url = os.environ.get("JUDGE_BASE_URL")
        judge_model = os.environ.get("JUDGE_MODEL")
        if judge_url or judge_model:
            saved = {k: os.environ.get(k) for k in
                     ("CHATBOT_BASE_URL", "CHATBOT_MODEL", "CHATBOT_PROVIDER")}
            try:
                if judge_url:
                    os.environ["CHATBOT_BASE_URL"] = judge_url
                    os.environ["CHATBOT_PROVIDER"] = "openai"
                if judge_model:
                    os.environ["CHATBOT_MODEL"] = judge_model
                return resolve_backend()
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        return resolve_backend()

    @property
    def available(self) -> bool:
        return self.backend is not None and getattr(self.backend, "available", False)

    @property
    def description(self) -> str:
        if not self.available:
            return "no judge model configured"
        own = os.environ.get("CHATBOT_MODEL")
        marking_own_work = (own or "") == getattr(self.backend, "model", "")
        suffix = "  (grading its own work — see judge.py)" if marking_own_work else ""
        return f"{self.backend.description}{suffix}"

    # -- grading -----------------------------------------------------------

    @staticmethod
    def _transcript(history: List[Dict[str, str]], limit: int = 8) -> str:
        lines = []
        for entry in history[-limit:]:
            who = "CUSTOMER" if entry.get("role") == "customer" else "BOT"
            lines.append(f"{who}: {(entry.get('content') or '').strip()}")
        return "\n".join(lines) or "(this is the first message)"

    async def assess(self, history: List[Dict[str, str]], said: str,
                     reply: str, criterion: Criterion) -> Tuple[Optional[bool], str]:
        """(passed, reason). `None` means the judge could not be asked."""
        if not self.available:
            return None, "no judge model"

        question = (
            f"## The conversation before this turn\n\n"
            f"{self._transcript(history)}\n\n"
            f"## The customer's latest message\n\n{said}\n\n"
            f"## What the bot replied\n\n{reply}\n\n"
            f"## The requirement\n\n{criterion.requirement}"
        )

        try:
            self.calls += 1
            answer = await self.backend.complete(
                system=JUDGE_PROMPT,
                messages=self.backend.build_messages(
                    [{"role": "user", "content": question}]),
                tools=[],
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as exc:
            self.failures += 1
            logger.warning("judge call failed: %s", exc)
            return None, f"judge unavailable: {exc}"

        return self._read_verdict(answer.text)

    @staticmethod
    def _read_verdict(text: str) -> Tuple[Optional[bool], str]:
        """Read PASS/FAIL out of the answer, however it is dressed up.

        An unreadable verdict returns None rather than a guess: a judge that
        silently defaults to PASS is worse than no judge, because the suite
        goes green while nothing is being checked.
        """
        body = (text or "").strip()
        verdict = re.search(r"VERDICT\s*[:\-]?\s*\**\s*(PASS|FAIL)", body, re.IGNORECASE)
        if not verdict:
            # Some models drop the label and just say the word.
            loose = re.search(r"\b(PASS|FAIL)\b", body, re.IGNORECASE)
            if not loose:
                return None, f"unreadable verdict: {body[:80]!r}"
            verdict = loose

        reason = re.search(r"REASON\s*[:\-]?\s*(.+)", body, re.IGNORECASE | re.DOTALL)
        detail = (reason.group(1).strip().split("\n")[0][:200]
                  if reason else body[:120].replace("\n", " "))
        return verdict.group(1).upper() == "PASS", detail
