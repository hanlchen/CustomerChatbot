"""
Tests for the knowledge retrieval layer and the LLM agent.

The retrieval tests are written as a small relevance benchmark: paraphrased
questions a real customer would ask, and the topic that must come back. Keyword
search scored 4/15 on these; the hybrid retriever must not regress below the
bar set here.

The LLM agent is tested with a scripted fake client, so the tool loop, context
bookkeeping and fallback behaviour are all verified without an API key or a
network call.
"""

import asyncio
import json
import re
import time
import types

import pytest

from conversation_manager import ConversationContext, ResponseGenerator
# The benchmark lives in eval_metrics.py, which also scores it. One list, so
# the pass/fail suite and the tracked metric cannot drift onto different
# questions -- two copies of a question set is how a green suite ends up
# measuring something the scoreboard does not.
from eval_metrics import BENCHMARK, HARD_BENCHMARK
from llm_agent import TOOL_DEFINITIONS, LLMAgent
from llm_providers import (
    AnthropicBackend,
    ModelBackend,
    ModelReply,
    OpenAIBackend,
    ToolCall,
    ToolOutcome,
    resolve_backend,
)
from mcp_server import CustomerChatbotTools
from retrieval import (
    BM25,
    HybridRetriever,
    build_corpus,
    get_retriever,
    phrase_concepts,
    reset_retriever,
    stem,
    tokenize,
)


@pytest.fixture(scope="module")
def retriever():
    reset_retriever()
    return get_retriever()


@pytest.fixture
def tools():
    return CustomerChatbotTools()


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

class TestTextProcessing:
    @pytest.mark.parametrize("word,expected", [
        ("returns", "return"), ("returning", "return"), ("returned", "return"),
        ("shipping", "ship"), ("policies", "policy"), ("cancelled", "cancel"),
        ("refunds", "refund"), ("delivery", "deliver"),
    ])
    def test_inflections_reduce_to_one_stem(self, word, expected):
        assert stem(word) == expected

    def test_stopwords_are_dropped(self):
        assert "the" not in tokenize("the return policy")

    def test_short_words_are_not_mangled(self):
        assert stem("is") == "is"
        assert stem("box") == "box"

    @pytest.mark.parametrize("query,concept", [
        ("can I send this back", "return"),
        ("I want my money back", "refund"),
        ("it stopped working", "defect"),
        ("my order never arrived", "track"),
    ])
    def test_phrases_imply_their_concept(self, query, concept):
        implied = phrase_concepts(query)
        assert implied, f"no concept found for {query!r}"
        assert any(concept in term or term in concept for term in implied), (
            f"{query!r} -> {implied}, expected something like {concept!r}"
        )


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class TestBM25:
    def test_ranks_the_document_containing_the_term(self):
        corpus = [["return", "refund"], ["ship", "deliver"], ["warranty"]]
        scores = BM25(corpus).scores(["return"])
        assert scores[0] > scores[1] and scores[0] > scores[2]

    def test_missing_term_scores_zero_everywhere(self):
        scores = BM25([["a"], ["b"]]).scores(["zzz"])
        assert all(s == 0 for s in scores)

    def test_empty_corpus_does_not_crash(self):
        assert BM25([]).scores(["anything"]) == []


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

class TestCorpus:
    def test_policies_are_split_into_sections(self, retriever):
        from database import _db
        # Far more chunks than documents, because documents are sectioned.
        assert len(retriever.chunks) > len(_db.policies)

    def test_faqs_are_indexed_too(self, retriever):
        assert any(c.doc_type == "faq" for c in retriever.chunks), (
            "FAQs are in the database but were never searchable"
        )

    def test_section_headings_are_captured(self, retriever):
        headings = {c.section for c in retriever.chunks if c.section}
        assert "Eligibility" in headings or "Process" in headings

    def test_every_chunk_has_text_and_tokens(self, retriever):
        for chunk in retriever.chunks:
            assert chunk.text.strip()
            assert chunk.tokens


# ---------------------------------------------------------------------------
# Relevance benchmark
# ---------------------------------------------------------------------------

class TestRelevanceBenchmark:
    @pytest.mark.parametrize("question,expected", BENCHMARK)
    def test_paraphrased_question_finds_the_right_topic(self, retriever, question, expected):
        hits = retriever.search(question, top_k=3)
        assert hits, f"no result at all for {question!r}"
        haystack = " ".join(
            f"{h.chunk.title} {h.chunk.section} {h.chunk.category}" for h in hits[:2]
        ).lower()
        assert re.search(expected, haystack), (
            f"{question!r} returned {[h.chunk.label for h in hits[:2]]}"
        )

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_gibberish_returns_nothing_rather_than_noise(self, retriever):
        assert retriever.search("xqzjvw ptkgh", top_k=3) == []

    def test_top_k_is_respected(self, retriever):
        assert len(retriever.search("return", top_k=2)) <= 2

    def test_no_duplicate_sections(self, retriever):
        hits = retriever.search("shipping cost", top_k=5)
        keys = [(h.chunk.doc_id, h.chunk.section) for h in hits]
        assert len(keys) == len(set(keys))

    def test_doc_type_filter(self, retriever):
        hits = retriever.search("return", top_k=5, doc_type="faq")
        assert all(h.chunk.doc_type == "faq" for h in hits)

    def test_works_without_an_embedding_backend(self):
        """Embeddings are an enhancement; absence must not break retrieval."""
        from database import _db

        class NoEmbeddings:
            available = False
            kind = "none"

        retriever = HybridRetriever(
            build_corpus(_db.policies, _db.faqs), embedder=NoEmbeddings()
        )
        assert retriever.backend == "lexical-only"
        assert retriever.search("can I send this back", top_k=2)


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------

class TestKnowledgeTools:
    @pytest.mark.asyncio
    async def test_search_knowledge_tool(self, tools):
        result = await tools.search_knowledge("can I send this back", top_k=3)
        assert result.success
        assert result.data["passages"]
        assert "text" in result.data["passages"][0]

    @pytest.mark.asyncio
    async def test_return_eligibility_verdicts_match_the_record(self, tools):
        from database import _db

        for order in list(_db.orders.values())[:25]:
            result = await tools.check_return_eligibility(order["order_id"])
            assert result.success
            verdict = result.data["verdict"]
            status = order["status"].lower()

            if status in ("pending", "processing", "shipped"):
                assert verdict == "not_yet", f"{status} -> {verdict}"
            elif status in ("cancelled", "returned"):
                assert verdict == "not_applicable"
            elif order.get("return_eligible"):
                assert verdict == "yes"
            else:
                assert verdict == "no"

    @pytest.mark.asyncio
    async def test_return_eligibility_unknown_order(self, tools):
        result = await tools.check_return_eligibility("ORD-999999")
        assert not result.success


# ---------------------------------------------------------------------------
# LLM agent (scripted client -- no API key, no network)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LLM agent — provider-neutral loop
# ---------------------------------------------------------------------------

class _ScriptedBackend(ModelBackend):
    """A backend that plays back scripted turns.

    Exercises the agent loop without any wire format, so these tests stay
    valid whichever provider is configured.
    """

    name = "scripted"

    def __init__(self, script):
        super().__init__(model="scripted-model", api_key="none")
        self.script = list(script)
        self.client = object()          # marks the backend as available
        self.status = "ready (scripted)"
        self.sent = []

    def tool_schemas(self, tools):
        return [{"name": t["name"]} for t in tools]

    def build_messages(self, turns):
        return [dict(t) for t in turns]

    async def complete(self, system, messages, tools, max_tokens,
                       tool_choice="auto"):
        self.sent.append({"system": system, "messages": [dict(m) for m in messages],
                          "tool_choice": tool_choice})
        if not self.script:
            raise AssertionError("scripted backend ran out of turns")
        step = self.script.pop(0)
        calls = [
            ToolCall(id=f"call-{i}", name=c["name"], arguments=c.get("arguments", {}))
            for i, c in enumerate(step.get("tools", []))
        ]
        return ModelReply(
            text=step.get("text", ""),
            tool_calls=calls,
            raw_message=step,
            input_tokens=step.get("input_tokens", 10),
            output_tokens=step.get("output_tokens", 5),
        )

    def append_tool_results(self, messages, reply, outcomes):
        messages.append({"role": "assistant", "content": "[tool calls]"})
        for outcome in outcomes:
            messages.append({
                "role": "tool",
                "tool_call_id": outcome.call.id,
                "content": outcome.as_json(),
            })


def verified_context(customer_id="CUST-10000"):
    """A session that has already confirmed the phone number.

    Most agent tests are about the tool loop, not the gate. Without this they
    would all fail at verification -- which is the gate working, but it tells
    you nothing about the thing under test.
    """
    context = ConversationContext()
    context.verified_customer_id = customer_id
    context.customer_id = customer_id
    context.verified_at = time.time()
    return context


def _agent(script, tools, route="data"):
    """An agent whose first scripted reply is the triage routing decision.

    Every turn is now triage-then-specialist, so a script that does not start
    with a route would have its first specialist reply eaten by triage.
    """
    return LLMAgent(tools, backend=_ScriptedBackend([{"text": route}] + list(script)))


class TestSpecialistLoop:
    """One specialist's tool loop, with triage already decided."""

    @pytest.mark.asyncio
    async def test_returns_text_when_no_tools_wanted(self, tools):
        agent = _agent([{"text": "Hello!"}], tools, route="general")
        result = await agent.respond("hi", ConversationContext(),
                                     [{"role": "customer", "content": "hi"}])
        assert result["response"] == "Hello!"
        assert result["agent"] == "triage"
        assert result["engine"] == "llm"

    @pytest.mark.asyncio
    async def test_runs_tools_then_answers(self, tools):
        agent = _agent([
            {"tools": [{"name": "list_my_orders",
                        }]},
            {"text": "You have 5 orders."},
        ], tools)
        result = await agent.respond("orders", verified_context(),
                                     [{"role": "customer", "content": "orders"}])
        assert result["response"] == "You have 5 orders."
        assert "list_my_orders" in result["actions_taken"]

    @pytest.mark.asyncio
    async def test_tool_results_are_returned_to_the_model(self, tools):
        backend = _ScriptedBackend([
            {"text": "data"},
            {"tools": [{"name": "list_my_orders",
                        }]},
            {"text": "ok"},
        ])
        agent = LLMAgent(tools, backend=backend)
        await agent.respond("x", verified_context(),
                            [{"role": "customer", "content": "x"}])
        final = backend.sent[-1]["messages"]
        tool_messages = [m for m in final if m.get("role") == "tool"]
        assert tool_messages, "tool results never reached the model"
        payload = json.loads(tool_messages[-1]["content"])
        assert payload["ok"] is True
        assert payload["data"]["customer_id"] == "CUST-10000"

    @pytest.mark.asyncio
    async def test_context_is_kept_in_step(self, tools):
        from database import _db
        order_id = next(o["order_id"] for o in _db.orders.values()
                        if o["customer_id"] == "CUST-10000")
        agent = _agent([
            {"tools": [{"name": "list_my_orders",
                        }]},
            {"tools": [{"name": "check_return_eligibility",
                        "arguments": {"order_id": order_id}}]},
            {"text": "done"},
        ], tools)
        context = verified_context()
        await agent.respond("can i return it", context,
                            [{"role": "customer", "content": "can i return it"}])
        assert context.customer_id == "CUST-10000"
        assert context.known_order_ids
        assert context.current_order_id == order_id
        # A declared field, not one invented by assignment. This assertion used
        # to name `returns_answered_for`, which no dataclass field backed --
        # Python created it on the instance and the test passed while nothing
        # in the app could ever read it.
        assert context.current_order_returnable is not None

    @pytest.mark.asyncio
    async def test_buttons_are_real_order_ids(self, tools):
        agent = _agent([
            {"tools": [{"name": "list_my_orders",
                        }]},
            {"text": "here you go"},
        ], tools)
        result = await agent.respond("orders", verified_context(),
                                     [{"role": "customer", "content": "orders"}])
        assert any(b.startswith("ORD-") for b in result["suggested_actions"])

    @pytest.mark.asyncio
    async def test_empty_reply_is_an_error_not_a_blank_message(self, tools):
        agent = _agent([{"text": "   "}], tools, route="general")
        with pytest.raises(RuntimeError):
            await agent.respond("hi", ConversationContext(),
                                [{"role": "customer", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_runaway_tool_loop_is_bounded(self, tools):
        agent = _agent([{"tools": [{"name": "list_policy_topics"}]}] * 30, tools,
                       route="policy")
        with pytest.raises(RuntimeError):
            await agent.respond("hi", ConversationContext(),
                                [{"role": "customer", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_hallucinated_tool_is_reported_not_fatal(self, tools):
        """Small open models invent tool names; that must not kill the turn."""
        agent = _agent([
            {"tools": [{"name": "issue_refund_now", "arguments": {"amount": 500}}]},
            {"text": "I can't do that, but support can."},
        ], tools, route="data")
        result = await agent.respond("refund", ConversationContext(),
                                     [{"role": "customer", "content": "refund"}])
        assert "support" in result["response"]

    @pytest.mark.asyncio
    async def test_malformed_tool_arguments_are_reported_not_fatal(self, tools):
        """Small models emit invalid JSON; the model gets told, not the user."""
        agent = _agent([
            {"tools": [{"name": "list_my_orders",
                        "arguments": {"__malformed_arguments__": "{oops"}}]},
            {"text": "Could you confirm your customer ID?"},
        ], tools)
        result = await agent.respond("x", ConversationContext(),
                                     [{"role": "customer", "content": "x"}])
        assert "customer ID" in result["response"]

    @pytest.mark.asyncio
    async def test_token_usage_is_accumulated(self, tools):
        agent = _agent([
            {"tools": [{"name": "list_policy_topics"}], "input_tokens": 100, "output_tokens": 20},
            {"text": "ok", "input_tokens": 150, "output_tokens": 30},
        ], tools, route="policy")
        await agent.respond("x", ConversationContext(),
                            [{"role": "customer", "content": "x"}])
        usage = agent.usage()
        # The triage call is scripted with the default 10/5, and counted --
        # it is a real per-turn cost and easy to forget.
        assert usage["input_tokens"] == 260
        assert usage["output_tokens"] == 55
        assert usage["turns"] == 1
        assert usage["triage_calls"] == 1

    def test_history_alternates_roles(self):
        history = [
            {"role": "customer", "content": "one"},
            {"role": "customer", "content": "two"},
            {"role": "bot", "content": "reply"},
            {"role": "customer", "content": "three"},
        ]
        turns = LLMAgent._build_turns("three", history)
        roles = [t["role"] for t in turns]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles
        assert roles[-1] == "user"

    def test_established_facts_are_passed_to_the_model(self):
        """Only what a tool returned. Everything else is in the transcript."""
        context = verified_context()
        context.known_order_ids = ["ORD-100000", "ORD-100001"]
        note = LLMAgent._context_note(context)
        assert "CUST-10000" in note
        assert "ORD-100000" in note

    def test_an_empty_context_says_it_is_not_verified(self):
        """The most important thing an agent can know about a fresh session."""
        note = LLMAgent._context_note(ConversationContext())
        assert "NOT VERIFIED" in note


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class TestBackendSelection:
    def test_no_configuration_means_no_backend(self, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "CHATBOT_BASE_URL", "CHATBOT_API_KEY",
                    "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CHATBOT_PROVIDER", "auto")
        monkeypatch.setenv("CHATBOT_ENGINE", "auto")
        assert resolve_backend() is None

    def test_base_url_selects_openai_compatible(self, monkeypatch):
        monkeypatch.setenv("CHATBOT_PROVIDER", "auto")
        monkeypatch.setenv("CHATBOT_ENGINE", "auto")
        monkeypatch.setenv("CHATBOT_BASE_URL", "http://localhost:8000/v1")
        backend = resolve_backend()
        assert isinstance(backend, OpenAIBackend)
        assert backend.base_url == "http://localhost:8000/v1"

    def test_anthropic_key_selects_anthropic(self, monkeypatch):
        monkeypatch.setenv("CHATBOT_PROVIDER", "auto")
        monkeypatch.setenv("CHATBOT_ENGINE", "auto")
        monkeypatch.delenv("CHATBOT_BASE_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert isinstance(resolve_backend(), AnthropicBackend)

    def test_engine_rules_disables_every_backend(self, monkeypatch):
        monkeypatch.setenv("CHATBOT_ENGINE", "rules")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("CHATBOT_BASE_URL", "http://localhost:8000/v1")
        assert resolve_backend() is None

    def test_local_server_needs_no_real_key(self, monkeypatch):
        monkeypatch.setenv("CHATBOT_PROVIDER", "openai")
        monkeypatch.setenv("CHATBOT_ENGINE", "auto")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHATBOT_API_KEY", raising=False)
        backend = resolve_backend()
        assert backend is not None and backend.available

    def test_explicit_provider_overrides_detection(self, monkeypatch):
        monkeypatch.setenv("CHATBOT_ENGINE", "auto")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("CHATBOT_PROVIDER", "openai")
        assert isinstance(resolve_backend(), OpenAIBackend)


# ---------------------------------------------------------------------------
# Wire format — real SDK against a faithful mock of vLLM's API
# ---------------------------------------------------------------------------

class TestOpenAIWireFormat:
    """Schema translation must match what a real vLLM server expects."""

    def test_tool_schemas_use_the_function_wrapper(self):
        backend = OpenAIBackend("m", "k", "http://x/v1")
        schemas = backend.tool_schemas(TOOL_DEFINITIONS)
        assert all(s["type"] == "function" for s in schemas)
        assert all("parameters" in s["function"] for s in schemas)
        assert {s["function"]["name"] for s in schemas} == {
            t["name"] for t in TOOL_DEFINITIONS
        }

    def test_anthropic_schemas_use_input_schema(self):
        backend = AnthropicBackend.__new__(AnthropicBackend)
        schemas = ModelBackend.tool_schemas.__get__(backend, AnthropicBackend) \
            if False else AnthropicBackend.tool_schemas(backend, TOOL_DEFINITIONS)
        assert all("input_schema" in s for s in schemas)
        assert all("type" not in s for s in schemas)

    @pytest.mark.asyncio
    async def test_an_agent_with_no_tools_omits_the_field_entirely(self):
        """`"tools": []` is invalid and vLLM 400s on it.

        Triage and the general agent both have no tools by design, so an empty
        array here failed *every* one of their turns -- which meant every turn,
        since triage runs first. The mock server accepted it and 155 tests
        passed while the real thing was completely broken.
        """
        backend = OpenAIBackend("m", "k", "http://x/v1")
        captured = {}

        class _Completions:
            @staticmethod
            async def create(**request):
                captured.update(request)
                raise _Stop()

        class _Stop(Exception):
            pass

        backend.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions()))

        with pytest.raises(_Stop):
            await backend.complete("sys", [{"role": "user", "content": "hi"}],
                                   tools=[], max_tokens=16)
        assert "tools" not in captured, "an empty tools array is rejected by vLLM"
        assert "tool_choice" not in captured

        captured.clear()
        with pytest.raises(_Stop):
            await backend.complete("sys", [{"role": "user", "content": "hi"}],
                                   tools=backend.tool_schemas(TOOL_DEFINITIONS),
                                   max_tokens=16)
        assert captured["tools"], "real tools must still be sent"
        assert captured["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_anthropic_also_omits_an_empty_tools_array(self):
        backend = AnthropicBackend.__new__(AnthropicBackend)
        backend.model = "m"
        captured = {}

        class _Stop(Exception):
            pass

        class _Messages:
            @staticmethod
            async def create(**request):
                captured.update(request)
                raise _Stop()

        backend.client = types.SimpleNamespace(messages=_Messages())
        with pytest.raises(_Stop):
            await AnthropicBackend.complete(
                backend, "sys", [{"role": "user", "content": "hi"}], [], 16)
        assert "tools" not in captured

    @pytest.mark.asyncio
    async def test_required_is_sent_and_degrades_once_if_rejected(self):
        """A server that will not take `required` must not fail every turn.

        `tool_choice: "required"` is in the OpenAI schema, but self-hosted
        servers vary. The first rejection turns it off for the process and the
        call is retried with "auto" -- the policy agent goes back to deciding
        for itself, which is worse but is still an answer.
        """
        backend = OpenAIBackend("m", "k", "http://x/v1")
        seen = []

        class _Completions:
            @staticmethod
            async def create(**request):
                seen.append(request.get("tool_choice"))
                if request.get("tool_choice") == "required":
                    raise ValueError("tool_choice 'required' is not supported")
                raise _Stop()

        class _Stop(Exception):
            pass

        backend.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions()))
        schemas = backend.tool_schemas(TOOL_DEFINITIONS)

        with pytest.raises(_Stop):
            await backend.complete("sys", [{"role": "user", "content": "hi"}],
                                   tools=schemas, max_tokens=16,
                                   tool_choice="required")
        assert seen == ["required", "auto"], "the rejection was not retried"

        # And it stays off, rather than paying for a failed call every turn.
        seen.clear()
        with pytest.raises(_Stop):
            await backend.complete("sys", [{"role": "user", "content": "hi"}],
                                   tools=schemas, max_tokens=16,
                                   tool_choice="required")
        assert seen == ["auto"]

    @pytest.mark.asyncio
    async def test_anthropic_spells_required_as_any(self):
        backend = AnthropicBackend.__new__(AnthropicBackend)
        backend.model = "m"
        captured = {}

        class _Stop(Exception):
            pass

        class _Messages:
            @staticmethod
            async def create(**request):
                captured.update(request)
                raise _Stop()

        backend.client = types.SimpleNamespace(messages=_Messages())
        with pytest.raises(_Stop):
            await AnthropicBackend.complete(
                backend, "sys", [{"role": "user", "content": "hi"}],
                AnthropicBackend.tool_schemas(backend, TOOL_DEFINITIONS), 16,
                tool_choice="required")
        assert captured["tool_choice"] == {"type": "any"}

    def test_tool_results_are_separate_tool_role_messages(self):
        """OpenAI wants role='tool' per result; Anthropic wants one user turn."""
        backend = OpenAIBackend("m", "k", "http://x/v1")

        class _Fn:
            name = "search_knowledge"
            arguments = '{"query": "returns"}'

        class _Call:
            id = "call_1"
            function = _Fn()

        class _Msg:
            content = None
            tool_calls = [_Call()]

        reply = ModelReply(text="", tool_calls=[
            ToolCall(id="call_1", name="search_knowledge", arguments={"query": "returns"})
        ], raw_message=_Msg())

        messages = []
        backend.append_tool_results(messages, reply, [
            ToolOutcome(call=reply.tool_calls[0], payload={"ok": True, "data": {}})
        ])

        assert messages[0]["role"] == "assistant"
        assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str), \
            "arguments must be a JSON string, not an object"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_1"


class TestEngineSelection:
    def test_falls_back_to_rules_when_the_llm_fails(self, tools):
        """A broken LLM must degrade the answer, not break the chatbot."""

        class Exploding:
            available = True
            status = "ready (test)"

            async def respond(self, *args, **kwargs):
                raise RuntimeError("simulated API outage")

        generator = ResponseGenerator(tools, llm_agent=Exploding())
        result = asyncio.run(generator.generate_response(
            "what is your return policy",
            ConversationContext(),
            [{"role": "customer", "content": "what is your return policy"}],
        ))
        assert result["response"].strip()
        assert result["agent"] != "llm"

    def test_reports_unavailable_when_there_is_no_model(self, tools):
        """No keyword engine hides behind this any more."""
        generator = ResponseGenerator(tools, llm_agent=None)
        assert generator.engine == "unavailable"

    def test_an_unanswerable_turn_says_so_rather_than_guessing(self, tools):
        generator = ResponseGenerator(tools, llm_agent=None)
        result = asyncio.run(generator.generate_response(
            "how much is shipping", ConversationContext(),
            [{"role": "customer", "content": "how much is shipping"}]))
        assert result["engine"] == "unavailable"
        assert result["agent"] == "none"
        assert result["actions_taken"] == []
        # It must not invent an answer, and must point somewhere real.
        assert "support@example.com" in result["response"]
        assert "$" not in result["response"]

    def test_reports_which_engine_is_live(self, tools):
        generator = ResponseGenerator(tools, llm_agent=None)
        assert generator.engine_status


# ---------------------------------------------------------------------------
# Tool router


# ---------------------------------------------------------------------------
# Triage and the specialists
# ---------------------------------------------------------------------------

class _RoutingBackend(_ScriptedBackend):
    """Answers triage with a fixed route, then never calls a tool.

    The specialists are tested against a model that refuses to act, so a
    passing test means the *structure* works, not that a good model rescued it.
    """

    def __init__(self, route, text="Here you go."):
        super().__init__([{"text": route}] + [{"text": text}] * 4)
        self.tool_lists = []

    async def complete(self, system, messages, tools, max_tokens,
                       tool_choice="auto"):
        self.tool_lists.append([t["name"] for t in (tools or [])])
        return await super().complete(system, messages, tools, max_tokens)


class TestTheTriagePromptCoversButtonLabels:
    """The bot generates its own inputs, and they look nothing like the examples.

    A tap on a suggestion sends the label as the message, so triage sees
    "Track order" -- two words, no question mark, no possessive. Every routing
    example in the prompt was a full sentence, and "general" is described as
    handling "messages you cannot make sense of", so terse commands drifted
    there. The subject came back right (`order_tracking`) while the specialist
    came back wrong, which is the signature of a model that understood the
    message and had nowhere in the instructions to put it.

    Routed to general it is worse than useless: that agent has no tools, so it
    asks for a customer ID it cannot call begin_verification with.
    """

    def test_every_button_the_ui_offers_is_routable(self):
        """Whatever `_suggest` can emit, the prompt has to have an answer for."""
        from llm_agent import TRIAGE_PROMPT, LLMAgent

        emitted = set()
        for tools_used, route in (([], "data"), ([], "general"),
                                  (["search_knowledge"], "policy"),
                                  (["get_order_details"], "data")):
            emitted.update(LLMAgent._suggest(ConversationContext(),
                                             tools_used, route))

        named = {line.split('"')[1]
                 for line in TRIAGE_PROMPT.splitlines()
                 if line.strip().startswith('"') and "->" in line}
        missing = {b for b in emitted if b not in named}
        assert not missing, (
            f"the UI offers {sorted(missing)} but the triage prompt never "
            "shows how to route them")

    def test_terse_is_not_the_same_as_unintelligible(self):
        from llm_agent import TRIAGE_PROMPT

        assert "not\nmerely short" in TRIAGE_PROMPT.replace("-- ", "")

    def test_the_worked_example_sends_a_bare_lookup_to_data(self):
        from llm_agent import TRIAGE_PROMPT

        line = next(l for l in TRIAGE_PROMPT.splitlines()
                    if '"Track order"' in l)
        assert "data order_tracking" in line


class TestTriage:
    """Triage is a model call whose whole job is naming the next agent."""

    @staticmethod
    async def _route(tools, message, route_reply, context=None):
        backend = _RoutingBackend(route_reply)
        agent = LLMAgent(tools, backend=backend)
        result = await agent.respond(
            message, context or ConversationContext(),
            [{"role": "customer", "content": message}])
        return result, backend

    @pytest.mark.asyncio
    @pytest.mark.parametrize("said,expected_agent", [
        ("data", "data"),
        ("policy", "policy"),
        ("general", "triage"),
    ])
    async def test_the_route_decides_which_agent_answers(self, tools, said,
                                                         expected_agent):
        result, _ = await self._route(tools, "anything", said)
        assert result["agent"] == expected_agent

    @pytest.mark.asyncio
    async def test_triage_is_its_own_call(self, tools):
        """One triage round trip, then the specialist's.

        Routed to `general` on purpose. A policy turn that answers with no
        lookup now earns a grounding round, which is correct behaviour and
        would make this count 3 -- but this test is about triage being its
        own call, not about the policy guard.
        """
        _, backend = await self._route(tools, "hello there", "general")
        assert len(backend.sent) == 2
        # Triage gets no tools at all -- it decides, it does not act.
        assert backend.tool_lists[0] == []

    @pytest.mark.asyncio
    async def test_each_specialist_sees_only_its_own_tools(self, tools):
        _, data = await self._route(tools, "where is my order", "data")
        _, policy = await self._route(tools, "what is your return policy", "policy")
        _, general = await self._route(tools, "hello", "general")

        assert set(data.tool_lists[1]) == {
            "begin_verification", "confirm_phone", "list_my_orders",
            "get_order_details", "check_return_eligibility"}
        assert set(policy.tool_lists[1]) == {"search_knowledge", "list_policy_topics"}
        assert general.tool_lists[1] == []

    @pytest.mark.asyncio
    async def test_each_specialist_gets_its_own_prompt(self, tools):
        _, data = await self._route(tools, "x", "data")
        _, policy = await self._route(tools, "x", "policy")
        assert "account specialist" in data.sent[1]["system"]
        assert "policy specialist" in policy.sent[1]["system"]
        # And triage's prompt is neither of those.
        assert "triage agent" in data.sent[0]["system"]
        assert "account specialist" not in data.sent[0]["system"]

    @pytest.mark.asyncio
    async def test_a_chatty_route_is_still_understood(self, tools):
        """Asked for one word, models still write sentences."""
        for said in ("data", "data.", "The answer is data.", '"data"', "DATA"):
            result, _ = await self._route(tools, "x", said)
            assert result["agent"] == "data", said

    @pytest.mark.asyncio
    async def test_an_unusable_answer_is_sent_back_to_triage(self, tools):
        """No regex overrules the agent -- it is told the format and asked again."""
        backend = _ScriptedBackend([
            {"text": "I'm not sure what you mean"},   # names no specialist
            {"text": "data order_status"},            # asked again, answers
            {"text": "Here are your orders."},
        ])
        agent = LLMAgent(tools, backend=backend)
        result = await agent.respond(
            "my id is CUST-10000", ConversationContext(),
            [{"role": "customer", "content": "my id is CUST-10000"}])
        assert result["agent"] == "data"
        assert agent.usage()["triage_calls"] == 2
        # The retry told it what was wrong rather than silently rerunning.
        correction = backend.sent[1]["messages"][-1]["content"]
        assert "two lowercase words" in correction

    @pytest.mark.asyncio
    async def test_triage_that_never_answers_falls_through_to_general(self, tools):
        """Failing here would blame the tools for a routing problem.

        The general agent has no tools, so an unroutable turn cannot leak
        anything, and saying "I did not understand, say that another way" is
        both true and useful -- unlike "our systems are down", which is not.
        """
        agent = LLMAgent(tools, backend=_ScriptedBackend(
            [{"text": "hmm"}] * 3 + [{"text": "Sorry, I did not follow that."}]))
        result = await agent.respond("x", ConversationContext(),
                                     [{"role": "customer", "content": "x"}])
        assert result["agent"] == "triage"          # the general agent's label
        assert result["actions_taken"] == []
        assert agent.usage()["triage_gave_up"] == 1

    @pytest.mark.asyncio
    async def test_giving_up_is_counted_not_hidden(self, tools):
        """A rising count is how you find out the routing is broken."""
        agent = LLMAgent(tools, backend=_ScriptedBackend(
            [{"text": "data order_status"}, {"text": "ok"}]))
        await agent.respond("x", ConversationContext(),
                            [{"role": "customer", "content": "x"}])
        assert agent.usage()["triage_gave_up"] == 0





class TestToolScoping:
    """A specialist offered a tool outside its job is a bug worth reporting."""

    @pytest.mark.asyncio
    async def test_a_specialist_cannot_reach_another_agents_tool(self, tools):
        """The policy agent asking for an order lookup gets told no."""
        backend = _ScriptedBackend([
            {"text": "policy"},
            {"tools": [{"name": "list_my_orders",
                        }]},
            {"text": "Let me answer from the policies instead."},
            # The reply above cites no lookup, so the grounding guard fetches
            # the passages and asks again. That is the behaviour under test in
            # test_security.py; here it just has to be budgeted for.
            {"text": "Returns are covered by the policy above."},
        ])
        agent = LLMAgent(tools, backend=backend)
        result = await agent.respond(
            "what is your return policy", ConversationContext(),
            [{"role": "customer", "content": "what is your return policy"}])

        refusal = json.loads(
            [m for m in backend.sent[-1]["messages"] if m.get("role") == "tool"][-1]
            ["content"])
        assert refusal["ok"] is False
        assert "not available to you" in refusal["error"]
        # And the turn still completes rather than dying.
        assert result["response"]
        # Nothing from the account database leaked into the answer. Checked
        # directly rather than via answer_source == "none": the refused turn
        # now gets grounded in the knowledge base, so the source is
        # "knowledge_base" -- correct, and not what this test is about.
        assert "list_my_orders" not in result["actions_taken"]
        assert result["answer_source"] != "account_data"

    def test_every_agent_only_names_real_tools(self):
        from llm_agent import AGENTS, TOOLS_BY_NAME

        for spec in AGENTS.values():
            for name in spec.tools:
                assert name in TOOLS_BY_NAME, f"{spec.name} names unknown tool {name}"

    def test_the_two_specialists_do_not_share_tools(self):
        from llm_agent import AGENTS

        data = set(AGENTS["data"].tools)
        policy = set(AGENTS["policy"].tools)
        assert not (data & policy), "a shared tool makes the routing pointless"
        assert AGENTS["general"].tools == ()


class TestTheAgentDoesNotRepeatItself:
    """Sending the customer the same sentence twice is its own failure.

    Three of the worst turns in the 40-turn run were this shape: "i dont have
    my cust number" answered with the sentence that had just asked for it, and
    a nonexistent CUST-99999 answered the same way again. Both are the context
    note describing an unchanged world, so the model answers an unchanged
    question identically.
    """

    def test_identical_text_is_a_repeat_whatever_the_punctuation(self):
        from llm_agent import LLMAgent as A

        first = "I can help with that -- what's your customer ID, or the email?"
        assert A._is_repeat(first, first)
        assert A._is_repeat(first.upper(), first)
        assert A._is_repeat(first.replace("--", "—") + "  ", first)

    def test_a_different_reply_is_not_a_repeat(self):
        from llm_agent import LLMAgent as A

        first = "I can help with that -- what's your customer ID, or the email?"
        second = ("No problem -- the email address on the account works just "
                  "as well. What is it?")
        assert not A._is_repeat(second, first)

    def test_short_replies_are_allowed_to_recur(self):
        """"Yes, that's right." twice is not a bug worth a retry."""
        from llm_agent import LLMAgent as A

        assert not A._is_repeat("Yes, that's right.", "Yes, that's right.")

    def test_nothing_to_repeat_on_the_first_turn(self):
        from llm_agent import LLMAgent as A

        assert not A._is_repeat("anything at all, at some length", "")

    @pytest.mark.asyncio
    async def test_the_guard_fires_and_the_second_answer_is_sent(self, tools):
        asked = ("I can help with that -- what's your customer ID, or the "
                 "email address on the account?")
        backend = _ScriptedBackend([
            {"text": "data"},
            {"text": asked},                     # the repeat
            {"text": "No problem -- the email on the account works too. "
                     "What is it?"},             # after the correction
        ])
        agent = LLMAgent(tools, backend=backend)
        result = await agent.respond(
            "i dont have my cust number", ConversationContext(),
            [{"role": "customer", "content": "can you check on my order"},
             {"role": "bot", "content": asked},
             {"role": "customer", "content": "i dont have my cust number"}])

        assert "repeat_reply" in result["telemetry"]["guards"]
        assert result["response"] != asked
        assert "email" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_the_guard_gives_up_after_one_try(self, tools):
        """One retry, then the reply ships.

        A model that repeats itself twice is not going to be argued out of it,
        and looping would spend the whole step budget to send the customer
        nothing.
        """
        asked = ("I can help with that -- what's your customer ID, or the "
                 "email address on the account?")
        backend = _ScriptedBackend([
            {"text": "data"},
            {"text": asked},
            {"text": asked},
        ])
        agent = LLMAgent(tools, backend=backend)
        result = await agent.respond(
            "i dont have my cust number", ConversationContext(),
            [{"role": "customer", "content": "can you check on my order"},
             {"role": "bot", "content": asked},
             {"role": "customer", "content": "i dont have my cust number"}])

        assert result["telemetry"]["guards"].count("repeat_reply") == 1
        assert result["response"] == asked


class TestTheContextNoteRemembersWhatWasTried:
    """An unverified turn must not look identical to the one before it."""

    def test_a_second_ask_is_marked_as_a_second_ask(self):
        from llm_agent import LLMAgent

        context = ConversationContext()
        first = LLMAgent._context_note(context, "data")
        context.identity_asks = 1
        second = LLMAgent._context_note(context, "data")

        assert first != second, \
            "two unverified turns produced the same prompt, so the model " \
            "answers the same question twice and has no reason to vary"
        assert "ALREADY asked" in second
        assert "email" in second

    def test_a_failed_lookup_is_remembered(self):
        from llm_agent import LLMAgent

        context = ConversationContext()
        context.failed_lookups = 1
        note = LLMAgent._context_note(context, "data")
        assert "matched no account" in note

    def test_a_wrong_identifier_is_never_stored(self):
        """CUST-99999 is somebody's guess, and possibly somebody's ID.

        The count is the useful part; the value is not, and an unverified
        conversation holds no identity -- including a wrong one.
        """
        from llm_agent import LLMAgent

        context = ConversationContext()
        LLMAgent._absorb(context, "begin_verification",
                         {"ok": True, "data": {"found": False,
                                               "message": "No account matches"}})
        assert context.failed_lookups == 1
        assert context.customer_id is None
        assert "99999" not in LLMAgent._context_note(context, "data")

    def test_a_successful_lookup_clears_the_count(self):
        from llm_agent import LLMAgent

        context = ConversationContext()
        context.failed_lookups = 2
        LLMAgent._absorb(context, "begin_verification",
                         {"ok": True, "data": {"found": True,
                                               "challenge_token": "tok"}})
        assert context.failed_lookups == 0
        assert context.challenge_token == "tok"


class TestThePromptKeepsUpWithTheTurn:
    @pytest.mark.asyncio
    async def test_verification_is_reflected_before_the_next_step(self, tools):
        """The stale-prompt bug.

        `system` was built once, before the tool loop. So after confirm_phone
        succeeded mid-turn, the last line the model read still said "a
        verification is in progress: call confirm_phone with it" -- which it
        had just done. Nothing told it the gate had opened, so it announced
        the lookup instead of running it, and the narration guard had to spend
        a whole extra generation undoing that.
        """
        import database

        customer_id = "CUST-10000"
        phone = database.get_customer_by_id(customer_id)["phone"]
        context = ConversationContext()

        # Turn one opens the challenge, exactly as a real conversation would.
        opening = LLMAgent(tools, backend=_ScriptedBackend([
            {"text": "data"},
            {"tools": [{"name": "begin_verification",
                        "arguments": {"identifier": customer_id}}]},
            {"text": "Can you confirm the number on the account?"},
        ]))
        await opening.respond(f"my id is {customer_id}", context,
                              [{"role": "customer",
                                "content": f"my id is {customer_id}"}])
        assert context.challenge_token

        # Turn two is the one that used to go wrong: confirm_phone succeeds
        # part way through, and the next step has to know it.
        backend = _ScriptedBackend([
            {"text": "data"},
            {"tools": [{"name": "confirm_phone",
                        "arguments": {"phone": phone}}]},
            {"tools": [{"name": "list_my_orders", "arguments": {}}]},
            {"text": "Your most recent order is on its way."},
        ])
        agent = LLMAgent(tools, backend=backend)
        await agent.respond(phone, context,
                            [{"role": "customer", "content": phone}])
        assert context.is_verified

        before, after = backend.sent[1]["system"], backend.sent[2]["system"]
        assert "verification is in progress" in before
        assert "IDENTITY VERIFIED" in after, \
            "the prompt still described verification as unfinished after it " \
            "had succeeded"
        assert "account tools are available" in after


class TestTheButtonsBeforeVerification:
    def test_no_real_customer_ids_are_offered(self, tools):
        """The interface used to hand out CUST-10000/10001/10002.

        Real IDs from the seeded dataset, offered as one-tap buttons, next to
        a prompt explaining that identity cannot be skipped. They also drove
        the repeat loop: the label became the next message, the state did not
        move, and the same question came back.
        """
        from llm_agent import LLMAgent

        buttons = LLMAgent._suggest(ConversationContext(), [], "data")
        assert not any(b.upper().startswith("CUST-") for b in buttons)
        assert buttons == ["Contact support"]

    def test_the_class_no_longer_carries_a_list_of_them(self):
        import llm_agent

        assert not hasattr(llm_agent.LLMAgent, "EXAMPLE_CUSTOMER_IDS")


class TestVerificationDoesNotEndTheTurn:
    """Confirming a phone number is plumbing, not an answer.

    The customer asked to track an order. That confirm_phone returned
    {"verified": true} is something the system needed, not something they
    wanted told -- but the model reads a tool result as an event worth
    reporting, writes "Great, you're confirmed -- let me check your orders",
    and ends the turn having looked nothing up.
    """

    @staticmethod
    def _verified_context(tools):
        """A context holding a live challenge, ready for confirm_phone."""
        import database

        customer_id = "CUST-10000"
        phone = database.get_customer_by_id(customer_id)["phone"]
        context = ConversationContext()
        agent = LLMAgent(tools, backend=_ScriptedBackend([
            {"text": "data"},
            {"tools": [{"name": "begin_verification",
                        "arguments": {"identifier": customer_id}}]},
            {"text": "Can you confirm the number on the account?"},
        ]))
        return context, phone, agent

    @pytest.mark.asyncio
    async def test_the_step_after_the_gate_opens_cannot_be_prose(self, tools):
        context, phone, opening = self._verified_context(tools)
        await opening.respond("my id is CUST-10000", context,
                              [{"role": "customer",
                                "content": "my id is CUST-10000"}])

        backend = _ScriptedBackend([
            {"text": "data"},
            {"tools": [{"name": "confirm_phone", "arguments": {"phone": phone}}]},
            {"tools": [{"name": "list_my_orders", "arguments": {}}]},
            {"text": "Thanks -- your most recent order is on its way."},
        ])
        result = await LLMAgent(tools, backend=backend).respond(
            phone, context, [{"role": "customer", "content": phone}])

        choices = [c["tool_choice"] for c in backend.sent]
        assert choices[2] == "required", \
            "the model was free to narrate instead of looking anything up"
        # The opening move is still the model's -- the data agent legitimately
        # asks for a customer ID there, and forcing a call would break that.
        assert choices[1] == "auto"
        # Once the lookup has run it must be free to write the answer.
        assert choices[3] == "auto"
        assert "list_my_orders" in result["actions_taken"]

    @pytest.mark.asyncio
    async def test_no_narration_guard_is_needed_any_more(self, tools):
        """The guard stays as a backstop, but should stop firing here.

        This is the whole point: the narration was costing a full extra
        generation on every verification turn.
        """
        context, phone, opening = self._verified_context(tools)
        await opening.respond("my id is CUST-10000", context,
                              [{"role": "customer",
                                "content": "my id is CUST-10000"}])

        backend = _ScriptedBackend([
            {"text": "data"},
            {"tools": [{"name": "confirm_phone", "arguments": {"phone": phone}}]},
            {"tools": [{"name": "list_my_orders", "arguments": {}}]},
            {"text": "Thanks -- your most recent order is on its way."},
        ])
        result = await LLMAgent(tools, backend=backend).respond(
            phone, context, [{"role": "customer", "content": phone}])

        assert result["telemetry"]["guards"] == []
        assert result["telemetry"]["model_calls"] == 4

    @pytest.mark.asyncio
    async def test_a_wrong_number_still_gets_a_sentence(self, tools):
        """A failed confirm_phone is exactly when prose IS the right move.

        Forcing a tool call here would answer "that number doesn't match"
        with a lookup the gate is still refusing.
        """
        context, _phone, opening = self._verified_context(tools)
        await opening.respond("my id is CUST-10000", context,
                              [{"role": "customer",
                                "content": "my id is CUST-10000"}])

        backend = _ScriptedBackend([
            {"text": "data"},
            {"tools": [{"name": "confirm_phone",
                        "arguments": {"phone": "+19999999999"}}]},
            {"text": "That number doesn't match what we have. Want to try "
                     "another one?"},
        ])
        result = await LLMAgent(tools, backend=backend).respond(
            "+19999999999", context,
            [{"role": "customer", "content": "+19999999999"}])

        assert not context.is_verified
        assert [c["tool_choice"] for c in backend.sent] == ["auto"] * 3
        assert "match" in result["response"]


class TestForcedFirstSearch:
    """The policy agent's opening call can be forced to be a tool call.

    Prompting it to search first was measured and found wanting: over a
    40-turn run it self-directed 8 of 33 searches and the grounding guard
    fetched the other 25 after the model had already answered from memory.
    `tool_choice="required"` moves that from a request to a constraint.
    """

    @staticmethod
    def _choices(backend):
        """tool_choice sent on each call, triage first."""
        return [call["tool_choice"] for call in backend.sent]

    @pytest.mark.asyncio
    async def test_policy_first_call_is_forced_when_enabled(
            self, tools, monkeypatch):
        monkeypatch.setenv("CHATBOT_FORCE_POLICY_SEARCH", "1")
        backend = _ScriptedBackend([
            {"text": "policy"},
            {"tools": [{"name": "search_knowledge",
                        "arguments": {"query": "returns"}}]},
            {"text": "You have 30 days to return an unopened item."},
        ])
        agent = LLMAgent(tools, backend=backend)
        await agent.respond("what is your return policy", ConversationContext(),
                            [{"role": "customer",
                              "content": "what is your return policy"}])

        triage, first, second = self._choices(backend)
        assert first == "required", "the policy agent's opening call was optional"
        # Triage has no tools and the follow-up has to be free to write prose;
        # forcing either would leave the turn with no way to produce an answer.
        assert triage == "auto"
        assert second == "auto"

    @pytest.mark.asyncio
    async def test_nothing_is_forced_by_default(self, tools):
        backend = _ScriptedBackend([
            {"text": "policy"},
            {"tools": [{"name": "search_knowledge",
                        "arguments": {"query": "returns"}}]},
            {"text": "You have 30 days to return an unopened item."},
        ])
        agent = LLMAgent(tools, backend=backend)
        await agent.respond("what is your return policy", ConversationContext(),
                            [{"role": "customer",
                              "content": "what is your return policy"}])

        assert set(self._choices(backend)) == {"auto"}

    @pytest.mark.asyncio
    async def test_the_data_agent_is_left_alone(self, tools, monkeypatch):
        """Only policy is forced.

        The data agent's first move is often to ask for a phone number, and a
        forced tool call would turn that question into a lookup it cannot yet
        run.
        """
        monkeypatch.setenv("CHATBOT_FORCE_POLICY_SEARCH", "1")
        backend = _ScriptedBackend([
            {"text": "data"},
            {"text": "What phone number is on the account?"},
        ])
        agent = LLMAgent(tools, backend=backend)
        await agent.respond("where is my order", ConversationContext(),
                            [{"role": "customer", "content": "where is my order"}])

        assert set(self._choices(backend)) == {"auto"}


class TestRoutedTurnsAreLabelled:
    @pytest.mark.asyncio
    async def test_the_reply_says_which_agent_and_which_engine(self, tools):
        backend = _RoutingBackend("policy")
        agent = LLMAgent(tools, backend=backend)
        result = await agent.respond(
            "how much is shipping", ConversationContext(),
            [{"role": "customer", "content": "how much is shipping"}])
        assert result["agent"] == "policy"
        assert result["engine"] == "llm"
        assert result["routed_to"] == "policy"

    @pytest.mark.asyncio
    async def test_routing_counts_are_tracked(self, tools):
        # The policy reply cites no lookup, so it earns one grounding round.
        agent = LLMAgent(tools, backend=_ScriptedBackend(
            [{"text": "policy"}, {"text": "a"}, {"text": "a, grounded"},
             {"text": "data"}, {"text": "b"}]))
        for message in ("how much is shipping", "where is my order"):
            await agent.respond(message, ConversationContext(),
                                [{"role": "customer", "content": message}])
        assert agent.usage()["routed_to"] == {"data": 1, "policy": 1, "general": 0}
        assert agent.usage()["triage_calls"] == 2



class TestAgainstAServerThatValidates:
    """End to end over HTTP, against a mock that enforces the OpenAI schema.

    The unit tests all passed while every real turn was failing, because the
    mock accepted a request vLLM rejects. These run the whole turn through the
    real SDK against a fixture that is as strict as the thing it stands in for.
    """

    @pytest.fixture
    def server(self):
        import subprocess
        import sys
        import time
        import urllib.error
        import urllib.request

        port = 8399
        process = subprocess.Popen(
            [sys.executable, "mock_vllm_server.py", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{base}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:                                          # pragma: no cover
            process.terminate()
            pytest.skip("mock server did not start")
        yield base
        process.terminate()
        process.wait(timeout=10)

    @staticmethod
    def _script(base, turns):
        import urllib.request

        urllib.request.urlopen(urllib.request.Request(
            f"{base}/__script",
            data=json.dumps({"script": turns}).encode(),
            headers={"Content-Type": "application/json"}))

    @staticmethod
    def _sent(base):
        import urllib.request

        return json.load(urllib.request.urlopen(f"{base}/__received"))["requests"]

    @pytest.mark.asyncio
    async def test_a_tool_free_turn_completes(self, tools, server):
        """Triage and the general agent both send no tools. Neither may 400."""
        from llm_providers import OpenAIBackend

        self._script(server, [{"text": "general greeting"},
                              {"text": "Hello! How can I help?"}])
        agent = LLMAgent(tools, backend=OpenAIBackend(
            "mock/Qwen3-8B", "k", f"{server}/v1"))
        result = await agent.respond("hi", ConversationContext(),
                                     [{"role": "customer", "content": "hi"}])
        assert result["response"] == "Hello! How can I help?"
        assert result["agent"] == "triage"
        assert result["intent"] == "greeting"

        sent = self._sent(server)
        assert "tools" not in sent[0], "triage must not send an empty tools array"
        assert "tools" not in sent[1], "the general agent has no tools"

    @pytest.mark.asyncio
    async def test_a_specialist_turn_sends_only_its_own_tools(self, tools, server):
        from llm_providers import OpenAIBackend

        self._script(server, [{"text": "policy shipping"},
                              {"text": "Standard shipping is free over $50."}])
        agent = LLMAgent(tools, backend=OpenAIBackend(
            "mock/Qwen3-8B", "k", f"{server}/v1"))
        await agent.respond("how much is shipping", ConversationContext(),
                            [{"role": "customer", "content": "how much is shipping"}])

        sent = self._sent(server)
        names = {t["function"]["name"] for t in sent[1]["tools"]}
        assert names == {"search_knowledge", "list_policy_topics"}

    def test_the_mock_rejects_what_vllm_rejects(self, server):
        """If the fixture is more forgiving than vLLM it will mislead us again."""
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"{server}/v1/chat/completions",
            data=json.dumps({"model": "m", "messages": [], "tools": []}).encode(),
            headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        assert caught.value.code == 400


class TestTheJudge:
    """The judge grades judgement calls. It must never silently pass one."""

    def test_it_reads_the_verdict_however_it_is_dressed(self):
        from judge import Judge

        for text, expected in [
            ("VERDICT: PASS\nREASON: asks for the ID.", True),
            ("VERDICT: FAIL\nREASON: never asked.", False),
            ("**VERDICT:** FAIL\nREASON: dodged it.", False),
            ("verdict - pass\nreason - fine", True),
            ("PASS", True),
            ("Fail", False),
        ]:
            passed, _ = Judge._read_verdict(text)
            assert passed is expected, text

    def test_an_unreadable_verdict_is_not_a_pass(self):
        """A judge that defaults to PASS turns the suite green over nothing."""
        from judge import Judge

        for text in ("I think it's probably fine?", "", "   ", "maybe"):
            passed, reason = Judge._read_verdict(text)
            assert passed is None, text
            assert "unreadable" in reason

    def test_the_reason_survives(self):
        from judge import Judge

        _, reason = Judge._read_verdict(
            "VERDICT: FAIL\nREASON: it asked for a phone number it cannot check.")
        assert "phone number" in reason

    @pytest.mark.asyncio
    async def test_no_judge_model_means_not_asked_rather_than_passed(self):
        from judge import Criterion, Judge

        judge = Judge(backend=None)
        passed, reason = await judge.assess([], "hi", "hello",
                                            Criterion("says something useful"))
        assert passed is None
        assert "no judge model" in reason

    @pytest.mark.asyncio
    async def test_a_judge_that_crashes_does_not_fail_the_scenario(self):
        from judge import Criterion, Judge

        class Exploding:
            available = True
            model = "boom"
            description = "boom"

            def build_messages(self, turns):
                return list(turns)

            async def complete(self, **kwargs):
                raise RuntimeError("judge server down")

        judge = Judge(backend=Exploding())
        passed, reason = await judge.assess([], "hi", "hello",
                                            Criterion("says something useful"))
        assert passed is None
        assert "judge unavailable" in reason

    def test_it_says_when_it_is_marking_its_own_work(self, monkeypatch):
        from judge import Judge

        class Same:
            available = True
            model = "Qwen/Qwen3-8B"
            description = "openai-compatible · Qwen/Qwen3-8B · local"

        monkeypatch.setenv("CHATBOT_MODEL", "Qwen/Qwen3-8B")
        assert "grading its own work" in Judge(backend=Same()).description

        class Bigger(Same):
            model = "claude-sonnet-4-5"
            description = "anthropic · claude-sonnet-4-5 · hosted API"

        assert "grading its own work" not in Judge(backend=Bigger()).description

    def test_facts_are_not_delegated_to_a_judge(self):
        """The line between the two is the whole design; keep it visible."""
        import judge as judge_module

        prose = judge_module.__doc__
        assert "did a tool actually run?" in prose
        assert "did verification get enforced?" in prose
