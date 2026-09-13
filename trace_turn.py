#!/usr/bin/env python3
"""
Print everything that happens in one turn: prompts, routing, tools, replies.

The chat API shows you the answer. This shows you the whole exchange -- what
triage was asked, what it said, which specialist ran, exactly what that
specialist was sent, every tool call and every tool result. When a reply is
wrong, this is how you find out whether triage misrouted it, the specialist
picked the wrong tool, or the tool returned something unexpected.

    python trace_turn.py "how much is shipping"
    python trace_turn.py "my id is CUST-10000" "can I return my most recent order?"
    python trace_turn.py --full "what is your return policy"   # untruncated
    python trace_turn.py --prompts "hello"                     # system prompts too

Several messages make one conversation: context carries between them, exactly
as it would in the UI.

Runs the agent in-process against your configured model. It does NOT need the
app to be running, but it does need the model server to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class S:
    on = sys.stdout.isatty()

    @staticmethod
    def _w(code: str, text: str) -> str:
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
    def blue(t): return S._w("34", t)
    @staticmethod
    def magenta(t): return S._w("35", t)
    @staticmethod
    def cyan(t): return S._w("36", t)


WIDTH = 78
LIMIT = 700          # characters per block before truncating


def rule(title: str = "", colour=S.dim) -> None:
    if title:
        bar = "─" * max(4, WIDTH - len(title) - 3)
        print(colour(f"── {title} {bar}"))
    else:
        print(colour("─" * WIDTH))


def block(text: str, indent: str = "    ", limit: Optional[int] = LIMIT) -> str:
    """Indent a block, collapsing the middle of anything very long."""
    text = str(text)
    if limit and len(text) > limit:
        head, tail = text[: limit // 2], text[-limit // 4:]
        skipped = len(text) - len(head) - len(tail)
        text = f"{head}\n{indent}  … {skipped} more characters …\n{tail}"
    return "\n".join(indent + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Recording backend
# ---------------------------------------------------------------------------

class TracingBackend:
    """Wraps the real backend and prints every round trip as it happens.

    A wrapper rather than a flag inside the agent: tracing is a debugging
    concern, and the agent should not carry code that only exists to be
    watched.
    """

    def __init__(self, inner, show_prompts: bool = False,
                 limit: Optional[int] = LIMIT):
        self._inner = inner
        self.show_prompts = show_prompts
        self.limit = limit
        self.call_index = 0
        self.seen_systems: set = set()

    # -- pass-through ------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def tool_schemas(self, tools):
        return self._inner.tool_schemas(tools)

    def build_messages(self, turns):
        return self._inner.build_messages(turns)

    def append_tool_results(self, messages, reply, outcomes):
        return self._inner.append_tool_results(messages, reply, outcomes)

    # -- the interesting one ----------------------------------------------

    async def complete(self, system, messages, tools, max_tokens,
                       tool_choice="auto"):
        self.call_index += 1
        index = self.call_index
        names = [self._tool_name(t) for t in (tools or [])]

        which = "TRIAGE" if index == 1 else f"SPECIALIST (round {index - 1})"
        rule(f"MODEL CALL {index} — {which}", S.cyan)
        print(S.dim(f"    tools offered : {', '.join(names) if names else '(none)'}"))
        print(S.dim(f"    max_tokens    : {max_tokens}"))

        if self.show_prompts:
            first_time = system not in self.seen_systems
            self.seen_systems.add(system)
            print(S.dim(f"\n    system prompt ({len(system)} chars)"
                        f"{'' if first_time else '  [same as an earlier call]'}"))
            if first_time:
                print(S.dim(block(system, limit=self.limit)))

        print(S.dim("\n    conversation sent to the model:"))
        for position, message in enumerate(messages):
            self._print_message(position, message)

        started = time.time()
        reply = await self._inner.complete(system, messages, tools, max_tokens)
        elapsed = time.time() - started

        print()
        if reply.tool_calls:
            for call in reply.tool_calls:
                print(S.yellow(f"    ◀ model wants: {call.name}"
                               f"({json.dumps(call.arguments)})"))
        if reply.text:
            print(S.green("    ◀ model said:"))
            print(S.green(block(reply.text, limit=self.limit)))
        if not reply.tool_calls and not reply.text:
            print(S.red("    ◀ model said nothing at all"))

        print(S.dim(f"\n    {elapsed:.1f}s · in {reply.input_tokens} tok"
                    f" · out {reply.output_tokens} tok"))
        print()
        return reply

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _tool_name(schema) -> str:
        if isinstance(schema, dict):
            if "function" in schema:               # OpenAI shape
                return schema["function"].get("name", "?")
            return schema.get("name", "?")          # Anthropic shape
        return str(schema)

    def _print_message(self, position: int, message: Any) -> None:
        role = message.get("role", "?") if isinstance(message, dict) else "?"
        colour = {"user": S.blue, "assistant": S.magenta,
                  "tool": S.yellow}.get(role, S.dim)
        content = message.get("content") if isinstance(message, dict) else message

        header = f"      [{position}] {role}"
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if calls:
            for call in calls:
                fn = call.get("function", {})
                print(colour(f"{header}  CALL {fn.get('name')}({fn.get('arguments')})"))
            if not content:
                return
            header = f"      [{position}] {role} (text)"

        if isinstance(content, list):
            # Anthropic content blocks.
            for chunk in content:
                kind = chunk.get("type") if isinstance(chunk, dict) else "?"
                if kind == "tool_use":
                    print(colour(f"{header}  CALL {chunk.get('name')}"
                                 f"({json.dumps(chunk.get('input'))})"))
                elif kind == "tool_result":
                    print(colour(f"{header}  RESULT"))
                    print(colour(block(chunk.get("content", ""), "          ",
                                       self.limit)))
                else:
                    print(colour(f"{header}"))
                    print(colour(block(chunk.get("text", chunk), "          ",
                                       self.limit)))
            return

        print(colour(header))
        print(colour(block(content or "(empty)", "          ", self.limit)))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def trace(messages: List[str], args) -> int:
    from conversation_manager import ConversationContext
    from mcp_server import CustomerChatbotTools

    tools = CustomerChatbotTools()
    context = ConversationContext()
    history: List[Dict[str, str]] = []

    from llm_agent import LLMAgent
    from llm_providers import resolve_backend, why_no_backend

    inner = resolve_backend()
    if inner is None or not inner.available:
        print(S.red(f"\nNo model configured: "
                    f"{inner.status if inner else why_no_backend()}"))
        print(S.dim("Put CHATBOT_BASE_URL and CHATBOT_MODEL in .env. There is "
                    "no rules engine to fall back to -- the agents are the app."))
        return 1
    backend = TracingBackend(inner, show_prompts=args.prompts,
                             limit=None if args.full else LIMIT)
    agent = LLMAgent(tools, backend=backend)
    print(S.bold(f"\nTracing {inner.description}\n"))

    turn_started = time.time()

    for number, message in enumerate(messages, start=1):
        rule()
        print(S.bold(f"TURN {number} — customer says: {message!r}"))
        rule()
        print(S.dim(f"  context in : customer_id={context.customer_id} "
                    f"order={context.current_order_id} "
                    f"known_orders={context.known_order_ids or '[]'}"))
        print()

        history.append({"role": "customer", "content": message})
        started = time.time()
        try:
            backend.call_index = 0
            result = await agent.respond(message, context, history)
        except Exception as exc:
            print(S.red(f"  TURN FAILED: {type(exc).__name__}: {exc}"))
            print(S.dim("  In the app the customer is told plainly that we "
                        "cannot answer -- there is no keyword engine to guess."))
            return 1
        elapsed = time.time() - started

        history.append({"role": "bot", "content": result["response"]})

        rule("REPLY", S.green)
        print(S.green(block(result["response"], "    ", None)))
        print()
        print(f"    agent        : {S.bold(str(result.get('agent')))}"
              f"   (engine: {result.get('engine', 'unknown')})")
        print(f"    intent       : {result.get('intent')}")
        print(f"    answer_source: {result.get('answer_source')}")
        print(f"    tools ran    : {result.get('actions_taken') or '(none)'}")
        print(f"    buttons      : {result.get('suggested_actions')}")
        print(S.dim(f"    took {elapsed:.1f}s"))
        print(S.dim(f"  context out: customer_id={context.customer_id} "
                    f"order={context.current_order_id} "
                    f"known_orders={context.known_order_ids or '[]'}"))
        print()

    rule()
    total = time.time() - turn_started
    print(S.bold(f"{len(messages)} turn(s) in {total:.1f}s"))
    if True:
        usage = agent.usage()
        print(S.dim(f"  triage calls : {usage['triage_calls']}"))
        print(S.dim(f"  routed to    : {usage['routed_to']}"))
        print(S.dim(f"  tokens       : in {usage['input_tokens']}, "
                    f"out {usage['output_tokens']}"))
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("messages", nargs="+",
                        help="One or more customer messages, in order")
    parser.add_argument("--full", action="store_true",
                        help="Do not truncate anything")
    parser.add_argument("--prompts", action="store_true",
                        help="Include the system prompt each agent was given")
    parser.add_argument("--base-url", help="Override CHATBOT_BASE_URL")
    parser.add_argument("--model", help="Override CHATBOT_MODEL")
    args = parser.parse_args()

    from env_file import load as load_env_file

    load_env_file()
    if args.base_url:
        os.environ["CHATBOT_BASE_URL"] = args.base_url
        os.environ["CHATBOT_PROVIDER"] = "openai"
    if args.model:
        os.environ["CHATBOT_MODEL"] = args.model

    return asyncio.run(trace(args.messages, args))


if __name__ == "__main__":
    sys.exit(main())
