"""
Model backends for the support agent.

The agent logic — decide, call tools, answer — is identical whichever model
serves it. What differs is wire format: Anthropic puts tool calls in content
blocks, OpenAI puts them in a `tool_calls` array on the assistant message.
This module hides that difference behind one small interface so
`llm_agent.py` never branches on provider.

Backends
--------
AnthropicBackend   Claude via api.anthropic.com
OpenAIBackend      Anything speaking the OpenAI chat-completions API:
                   vLLM, SGLang, llama.cpp server, Ollama, LM Studio,
                   Docker Model Runner, or OpenAI itself

Selection (see resolve_backend)
-------------------------------
    CHATBOT_PROVIDER   auto (default) | anthropic | openai
    CHATBOT_BASE_URL   http://localhost:8000/v1   -> selects openai
    CHATBOT_MODEL      model id
    CHATBOT_API_KEY    key; local servers accept any non-empty string
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# A local server needs a non-empty key to satisfy the SDK, but ignores it.
LOCAL_PLACEHOLDER_KEY = "local-no-key-required"

# A local 8B model on a laptop answers in tens of seconds, not minutes. Long
# enough to be patient, short enough that a hang surfaces as an error.
DEFAULT_TIMEOUT_SECONDS = 120.0


def _disable_thinking() -> bool:
    """Whether to ask reasoning models to skip their chain of thought.

    Qwen3 and similar emit <think> tokens before answering. On a laptop at
    ~13 tokens/sec that is a large share of the wait, and a support answer
    grounded in tool results does not need it. Set CHATBOT_THINKING=1 to keep it.
    """
    return os.environ.get("CHATBOT_THINKING", "").strip().lower() not in (
        "1", "true", "yes", "on"
    )


def _timeout_seconds() -> float:
    try:
        return float(os.environ.get("CHATBOT_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "Qwen/Qwen3-8B"


# ---------------------------------------------------------------------------
# Provider-neutral types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """One tool the model wants run."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelReply:
    """What came back from one round trip."""
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    # Provider-native assistant message, appended verbatim to the transcript
    # so the next request stays consistent with what the model produced.
    raw_message: Any = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolOutcome:
    """The result of running one tool, ready to send back."""
    call: ToolCall
    payload: Dict[str, Any]

    def as_json(self, limit: int = 12000) -> str:
        return json.dumps(self.payload, default=str)[:limit]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ModelBackend:
    """Common interface. Subclasses own their wire format entirely."""

    name = "base"

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url
        self.client = None
        self.status = "not initialised"
        self._api_key = api_key

    @property
    def available(self) -> bool:
        return self.client is not None

    @property
    def description(self) -> str:
        target = self.base_url or "hosted API"
        return f"{self.name} · {self.model} · {target}"

    # -- to implement -----------------------------------------------------

    def tool_schemas(self, tools: Sequence[Dict[str, Any]]) -> Any:
        """Translate neutral tool definitions to this provider's format."""
        raise NotImplementedError

    def build_messages(self, turns: Sequence[Dict[str, str]]) -> List[Any]:
        """Turn (role, content) history into provider messages."""
        raise NotImplementedError

    async def complete(self, system: str, messages: List[Any],
                       tools: Any, max_tokens: int,
                       tool_choice: str = "auto") -> ModelReply:
        """Ask the model for one reply.

        `tool_choice` is "auto" (the model decides) or "required" (it must
        call one of the tools it was given). Required exists because deciding
        is the part a small model is worst at: measured on Qwen3-8B-4bit the
        policy agent answered from memory on three quarters of turns despite
        the prompt telling it, in three places, to search first. A constraint
        cannot be talked out of anything -- the same reason the verification
        gate lives in the tool layer rather than the prompt.
        """
        raise NotImplementedError

    def append_tool_results(self, messages: List[Any], reply: ModelReply,
                            outcomes: Sequence[ToolOutcome]) -> None:
        """Append the assistant turn and its tool results to `messages`."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicBackend(ModelBackend):
    name = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(model, api_key, base_url)
        try:
            from anthropic import AsyncAnthropic

            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "timeout": _timeout_seconds(),
                "max_retries": int(os.environ.get("CHATBOT_MAX_RETRIES", "1")),
            }
            if base_url:
                kwargs["base_url"] = base_url
            self.client = AsyncAnthropic(**kwargs)
            self.status = f"ready ({self.description})"
        except ImportError:
            self.status = "anthropic package not installed (pip install anthropic)"
        except Exception as exc:
            self.status = f"failed to initialise: {exc}"

    def tool_schemas(self, tools):
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    def build_messages(self, turns):
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    async def complete(self, system, messages, tools, max_tokens,
                       tool_choice="auto"):
        request: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        # Omitted rather than empty, for the same reason as the OpenAI path:
        # an agent with no tools must not send an empty tools array.
        if tools:
            request["tools"] = tools
            # Anthropic spells "you must call something" as {"type": "any"}.
            if tool_choice == "required":
                request["tool_choice"] = {"type": "any"}

        response = await self.client.messages.create(**request)

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()

        calls = [
            ToolCall(id=block.id, name=block.name, arguments=block.input or {})
            for block in response.content
            if getattr(block, "type", "") == "tool_use"
        ]

        usage = getattr(response, "usage", None)
        return ModelReply(
            text=text,
            tool_calls=calls,
            raw_message=response.content,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def append_tool_results(self, messages, reply, outcomes):
        messages.append({"role": "assistant", "content": reply.raw_message})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": outcome.call.id,
                    "content": outcome.as_json(),
                }
                for outcome in outcomes
            ],
        })

# ---------------------------------------------------------------------------
# OpenAI-compatible (vLLM, SGLang, llama.cpp, Ollama, OpenAI)
# ---------------------------------------------------------------------------

class OpenAIBackend(ModelBackend):
    name = "openai-compatible"

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(model, api_key, base_url)
        try:
            from openai import AsyncOpenAI

            # The SDK defaults to a 600s read timeout with 2 retries, which
            # means a hung request can spin for half an hour. A local model is
            # slow but not that slow -- fail fast enough to fall back.
            self.client = AsyncOpenAI(
                api_key=api_key or LOCAL_PLACEHOLDER_KEY,
                base_url=base_url,
                timeout=_timeout_seconds(),
                max_retries=int(os.environ.get("CHATBOT_MAX_RETRIES", "1")),
            )
            self._send_thinking_flag = _disable_thinking()
            # Set False the first time a server rejects tool_choice, so one
            # unsupported server costs one failed request rather than every
            # policy turn from then on.
            self._tool_choice_supported = True
            self.status = f"ready ({self.description})"
        except ImportError:
            self._send_thinking_flag = False
            self._tool_choice_supported = False
            self.status = "openai package not installed (pip install openai)"
        except Exception as exc:
            self._tool_choice_supported = False
            self.status = f"failed to initialise: {exc}"

    def tool_schemas(self, tools):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    def build_messages(self, turns):
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    async def complete(self, system, messages, tools, max_tokens,
                       tool_choice="auto"):
        # OpenAI carries the system prompt as the first message rather than a
        # separate parameter.
        payload = [{"role": "system", "content": system}] + messages

        request = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": payload,
        }
        # An empty tools array is not valid: the OpenAI schema requires at
        # least one entry, and vLLM rejects `"tools": []` with a 400. The
        # triage agent and the general agent both have no tools by design, so
        # the field has to be absent rather than empty -- otherwise every one
        # of their turns fails.
        if tools:
            request["tools"] = tools
            # `required` is in the OpenAI schema, but support varies across
            # self-hosted servers -- the MLX and CUDA vLLM paths do not always
            # agree. `_tool_choice_supported` degrades to "auto" on the first
            # rejection rather than failing every policy turn from then on,
            # the same way the thinking flag below does.
            request["tool_choice"] = (
                tool_choice if self._tool_choice_supported else "auto")

        # Reasoning models are told to skip their chain of thought. Not every
        # chat template accepts the argument, so the first rejection turns it
        # off permanently rather than failing every turn.
        if self._send_thinking_flag:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        try:
            response = await self.client.chat.completions.create(**request)
        except Exception as exc:
            if (self._tool_choice_supported
                    and request.get("tool_choice") not in (None, "auto")
                    and "tool_choice" in str(exc).lower()):
                logger.warning(
                    "this server rejected tool_choice=%r; falling back to "
                    "'auto' for the rest of the process. The policy agent "
                    "will decide for itself whether to search, which is the "
                    "behaviour this was meant to remove.",
                    request["tool_choice"])
                self._tool_choice_supported = False
                request["tool_choice"] = "auto"
                response = await self.client.chat.completions.create(**request)
            elif self._send_thinking_flag and "chat_template" in str(exc).lower():
                self._send_thinking_flag = False
                request.pop("extra_body", None)
                response = await self.client.chat.completions.create(**request)
            else:
                raise

        choice = response.choices[0]
        message = choice.message
        text = (message.content or "").strip()

        calls: List[ToolCall] = []
        for call in (message.tool_calls or []):
            raw_args = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                # Smaller open models occasionally emit malformed JSON. Report
                # it to the model as a tool error rather than crashing the turn.
                arguments = {"__malformed_arguments__": raw_args}
            calls.append(ToolCall(id=call.id, name=call.function.name,
                                  arguments=arguments))

        usage = getattr(response, "usage", None)
        return ModelReply(
            text=text,
            tool_calls=calls,
            raw_message=message,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def append_tool_results(self, messages, reply, outcomes):
        message = reply.raw_message
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in (message.tool_calls or [])
            ],
        })
        # Each tool result is its own message, keyed by call id.
        for outcome in outcomes:
            messages.append({
                "role": "tool",
                "tool_call_id": outcome.call.id,
                "content": outcome.as_json(),
            })

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

# Values of CHATBOT_ENGINE that mean "use no model at all". `rules` is the old
# spelling from when there was a second engine to fall back to; it is honoured
# so an existing .env keeps working, but `off` is what it does.
ENGINE_OFF = frozenset({"off", "none", "rules"})

def resolve_backend() -> Optional[ModelBackend]:
    """Pick a backend from the environment, or None if there is no model.

    Order:
      1. CHATBOT_ENGINE=off                -> None
      2. CHATBOT_PROVIDER, if set          -> that provider
      3. CHATBOT_BASE_URL set              -> OpenAI-compatible (local server)
      4. ANTHROPIC_API_KEY set             -> Anthropic
      5. otherwise                         -> None

    None means no model, which means every turn answers with the unavailable
    message. It used to mean "fall back to the rules engine"; that engine is
    gone, so `rules` is kept only as a spelling of `off` for existing .env
    files, and it no longer describes anything that happens.
    """
    if os.environ.get("CHATBOT_ENGINE", "auto").lower() in ENGINE_OFF:
        return None

    provider = os.environ.get("CHATBOT_PROVIDER", "auto").lower()
    base_url = os.environ.get("CHATBOT_BASE_URL") or None
    model = os.environ.get("CHATBOT_MODEL")
    explicit_key = os.environ.get("CHATBOT_API_KEY")

    if provider == "auto":
        if base_url:
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            return None  # why_no_backend() explains this to the operator

    if provider == "anthropic":
        key = explicit_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        return AnthropicBackend(model or DEFAULT_ANTHROPIC_MODEL, key, base_url)

    if provider in ("openai", "openai-compatible", "vllm", "local"):
        key = (explicit_key or os.environ.get("OPENAI_API_KEY")
               or LOCAL_PLACEHOLDER_KEY)
        # A local server is the common case; default to the vLLM address.
        return OpenAIBackend(
            model or DEFAULT_OPENAI_MODEL,
            key,
            base_url or "http://localhost:8000/v1",
        )

    return None


def why_no_backend() -> str:
    """Explain, specifically, why no model resolved.

    "no model backend configured" is true but useless when you are three
    environment variables deep and one of them is misspelled.
    """
    engine = os.environ.get("CHATBOT_ENGINE", "auto").lower()
    if engine in ENGINE_OFF:
        return (f"CHATBOT_ENGINE={engine} is set, so no model will be used "
                "and every turn will answer with the unavailable message")

    provider = os.environ.get("CHATBOT_PROVIDER", "auto").lower()
    base_url = os.environ.get("CHATBOT_BASE_URL")
    model = os.environ.get("CHATBOT_MODEL")

    if base_url or provider in ("openai", "vllm", "local"):
        try:
            import openai  # noqa: F401  (probe: is it installed?)
        except ImportError:
            return ("CHATBOT_BASE_URL is set but the openai package is missing "
                    "-- run: pip install openai")
        return f"configured for {base_url or 'a local server'} but the client failed to start"

    if provider == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401  (probe: is it installed?)
        except ImportError:
            return ("ANTHROPIC_API_KEY is set but the anthropic package is "
                    "missing -- run: pip install anthropic")
        return "ANTHROPIC_API_KEY did not produce a working client"

    hint = ""
    if model:
        hint = (f" (CHATBOT_MODEL={model} is set, but on its own it selects "
                "nothing -- the provider is chosen by CHATBOT_BASE_URL)")
    return ("no CHATBOT_BASE_URL and no ANTHROPIC_API_KEY" + hint)
