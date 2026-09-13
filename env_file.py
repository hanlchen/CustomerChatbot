"""
Load `.env` into the environment.

The project documented a `.env` file and shipped a `.env.example`, but nothing
ever read it -- so every setting had to be exported by hand, in every terminal,
and a fresh shell silently fell back to the rules engine. This closes that gap.

Deliberately dependency-free: `python-dotenv` is one more thing to install
before the app will start, and the format is four lines of parsing.

Rules
-----
- A variable already set in the environment always wins. `.env` is a default,
  not an override, so `CHATBOT_MODEL=x python app.py` still does what it says.
- `export FOO=bar` is accepted, since people paste shell lines in.
- Quotes around the value are stripped; `#` starts a comment only at the
  beginning of a line, so a value can contain one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_FILENAME = ".env"


def parse(text: str) -> List[Tuple[str, str]]:
    """Parse .env content into (key, value) pairs, in file order."""
    pairs: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        pairs.append((key, value))
    return pairs


def load(path: str | os.PathLike | None = None,
         override: bool = False) -> Dict[str, str]:
    """Read `.env` (if present) into os.environ. Returns what it applied.

    Missing file is not an error -- the app runs fine without one.
    """
    target = Path(path) if path else Path(__file__).with_name(DEFAULT_FILENAME)
    if not target.is_file():
        return {}

    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}

    applied: Dict[str, str] = {}
    for key, value in parse(text):
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
