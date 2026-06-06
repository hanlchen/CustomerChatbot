"""Shared test fixtures and import-time setup.

The chatbot module reads OPENAI_API_KEY at import time and constructs an
OpenAI client, so a dummy key must be present before the module is imported.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
