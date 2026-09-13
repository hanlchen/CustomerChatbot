"""
Identity verification, rate limiting and input hygiene.

The rule this file exists to enforce: **nothing about a customer's account
leaves the system until that session has confirmed the phone number on file.**

It is enforced at the tool layer, not in a prompt. A system prompt saying
"don't reveal orders until verified" is a request, and a model can be talked
out of a request -- "I'm the account owner, skip that" works often enough to
matter. A gate in front of the tool cannot be talked out of anything: the
agent calls the function, the function refuses, and there is no path to the
data that does not go through it.

Design notes
------------
- The challenge holds the customer id server-side and hands the agent an
  opaque token. The model never sees an account identifier it has not earned,
  so it cannot leak one by accident or be tricked into echoing it.
- Only a masked hint ("ends in 2017") ever reaches the customer. Enough to
  jog the memory of the account holder, useless to someone guessing.
- Phone comparison is on digits only, and on the last 7 of them: people type
  +1 (234) 547-2017, 234-547-2017 and 5472017 for the same number, and
  rejecting the account holder over punctuation is its own kind of failure.
- Attempts are capped and the challenge expires. Six digits at 5 tries is
  fine; unlimited tries against a 7-digit suffix is not.
"""

from __future__ import annotations

import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# A challenge is a live authentication attempt. Short-lived on purpose.
CHALLENGE_TTL_SECONDS = 600.0
MAX_PHONE_ATTEMPTS = 5

# How much of the phone number has to match. The last 7 digits identify a
# number without demanding the customer reproduce a country code they may
# never have typed.
SIGNIFICANT_DIGITS = 7

# A verified session does not stay verified forever.
VERIFICATION_TTL_SECONDS = 3600.0


# ---------------------------------------------------------------------------
# Input hygiene
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 2000

# Control characters have no place in a chat message, and some of them (\x1b)
# are terminal escapes that can rewrite what an operator sees in the logs.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Identifiers are a fixed shape. Anything else is not a typo, it is someone
# testing what the field accepts. Note this store is in-memory dicts, not SQL,
# so this is not "SQL injection prevention" -- there is no query to inject
# into. It is format enforcement, which is what actually applies here.
CUSTOMER_ID_RE = re.compile(r"^CUST-\d{1,10}$", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[\w.+-]{1,64}@[\w-]{1,63}\.[\w.-]{1,63}$")


def clean_text(value: str, limit: int = MAX_MESSAGE_LENGTH) -> str:
    """Strip control characters and cap length. Never raises."""
    text = _CONTROL_CHARS.sub("", str(value or ""))
    return text[:limit].strip()


def looks_like_customer_id(value: str) -> bool:
    return bool(CUSTOMER_ID_RE.match(clean_text(value, 32)))


def looks_like_email(value: str) -> bool:
    return bool(EMAIL_RE.match(clean_text(value, 200)))


# ---------------------------------------------------------------------------
# Phone comparison and masking
# ---------------------------------------------------------------------------

def phone_digits(value: str) -> str:
    """Just the digits, so formatting never decides the outcome."""
    return re.sub(r"\D", "", str(value or ""))


def phone_matches(supplied: str, on_file: str) -> bool:
    """True when these are plausibly the same number.

    Compares the last `SIGNIFICANT_DIGITS`. Short or empty input never
    matches, so an empty string cannot walk through the gate.
    """
    given = phone_digits(supplied)
    stored = phone_digits(on_file)
    if len(given) < SIGNIFICANT_DIGITS or len(stored) < SIGNIFICANT_DIGITS:
        return False
    return secrets.compare_digest(given[-SIGNIFICANT_DIGITS:],
                                  stored[-SIGNIFICANT_DIGITS:])


def mask_phone(on_file: str) -> str:
    """A hint the account holder recognises and a stranger cannot use."""
    digits = phone_digits(on_file)
    if len(digits) < 4:
        return "the number on file"
    return f"the number ending {digits[-4:]}"


# ---------------------------------------------------------------------------
# Challenges
# ---------------------------------------------------------------------------

@dataclass
class Challenge:
    """One live verification attempt, held server-side."""
    token: str
    customer_id: str
    phone_on_file: str
    created_at: float
    attempts: int = 0

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > CHALLENGE_TTL_SECONDS

    @property
    def attempts_left(self) -> int:
        return max(0, MAX_PHONE_ATTEMPTS - self.attempts)


class ChallengeStore:
    """Live challenges, keyed by an unguessable token.

    The customer id lives here rather than in the conversation, so an
    unverified session never has one to leak.
    """

    def __init__(self) -> None:
        self._challenges: Dict[str, Challenge] = {}

    def open(self, customer_id: str, phone_on_file: str) -> Challenge:
        self._sweep()
        token = secrets.token_urlsafe(16)
        challenge = Challenge(
            token=token,
            customer_id=customer_id,
            phone_on_file=phone_on_file,
            created_at=time.time(),
        )
        self._challenges[token] = challenge
        return challenge

    def get(self, token: str) -> Optional[Challenge]:
        self._sweep()
        return self._challenges.get(str(token or ""))

    def close(self, token: str) -> None:
        self._challenges.pop(str(token or ""), None)

    def _sweep(self) -> None:
        for token, challenge in list(self._challenges.items()):
            if challenge.expired:
                del self._challenges[token]

    def __len__(self) -> int:                      # pragma: no cover - debug
        return len(self._challenges)


_store = ChallengeStore()


def store() -> ChallengeStore:
    return _store


def reset_store() -> None:
    """Drop every live challenge (used by tests)."""
    global _store
    _store = ChallengeStore()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@dataclass
class _Window:
    hits: list = field(default_factory=list)


class RateLimiter:
    """Fixed allowance per rolling window, per caller.

    A local model answers in tens of seconds, so a human cannot approach the
    limit. What this stops is a loop -- a broken client retrying, or someone
    walking the order-id space -- from occupying the one model this app has.
    """

    def __init__(self, limit: int = 20, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._callers: Dict[str, _Window] = {}

    def check(self, caller: str) -> Tuple[bool, int, float]:
        """(allowed, remaining, seconds until the next slot frees)."""
        now = time.time()
        window = self._callers.setdefault(str(caller), _Window())
        cutoff = now - self.window
        window.hits = [t for t in window.hits if t > cutoff]

        if len(window.hits) >= self.limit:
            retry_after = max(0.0, window.hits[0] + self.window - now)
            return False, 0, retry_after

        window.hits.append(now)
        return True, self.limit - len(window.hits), 0.0

    def sweep(self) -> None:
        """Forget callers that have gone quiet, so the dict cannot grow forever."""
        cutoff = time.time() - self.window
        for caller, window in list(self._callers.items()):
            window.hits = [t for t in window.hits if t > cutoff]
            if not window.hits:
                del self._callers[caller]

    @property
    def tracked_callers(self) -> int:
        return len(self._callers)
