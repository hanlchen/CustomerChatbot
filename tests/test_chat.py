"""
Sessions, history and context: the state a conversation carries between turns.

The agents are tested in test_retrieval.py and the identity gate in
test_security.py. What is checked here is the layer underneath both -- that a
session keeps its own history, that context set on one turn is still there on
the next, and that the session cap actually caps.
"""

import pytest
from conversation_manager import (
    ConversationContext,
    ConversationManager,
    ResponseGenerator,
    get_conversation_manager,
    reset_conversation_manager,
)
from mcp_server import CustomerChatbotTools


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def manager():
    """A manager with no sessions carried over from the last test."""
    reset_conversation_manager()
    return get_conversation_manager()


@pytest.fixture(scope="session")
def mcp_tools():
    """The tool surface, built once -- it holds no per-test state."""
    return CustomerChatbotTools()


# ============================================================================
# Conversation Manager Tests
# ============================================================================

class TestConversationManager:
    """Sessions, history and context."""

    def test_create_session(self, manager):
        """A new session is addressable by the id it hands back."""
        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) > 0

        conversation = manager.get_session(session_id)
        assert conversation is not None
        assert conversation.session_id == session_id

    def test_create_session_with_customer_id(self, manager):
        """A customer id supplied up front lands in the context."""
        customer_id = "CUST-10000"
        session_id = manager.create_session(customer_id=customer_id)

        conversation = manager.get_session(session_id)
        assert conversation.context.customer_id == customer_id

    def test_get_session_by_customer(self, manager):
        """A returning customer resolves to their existing session."""
        customer_id = "CUST-10001"
        session_id = manager.create_session(customer_id=customer_id)

        conversation = manager.get_session_by_customer(customer_id)
        assert conversation is not None
        assert conversation.session_id == session_id

    def test_add_message(self, manager):
        """A stored message keeps its role, text and timestamp."""
        session_id = manager.create_session()

        message = manager.add_message(
            session_id,
            role="customer",
            content="Where is my order?"
        )

        assert message is not None
        assert message.role == "customer"
        assert message.content == "Where is my order?"
        assert message.timestamp is not None

    def test_message_with_intent(self, manager):
        """The intent label travels with the message it describes."""
        session_id = manager.create_session()

        message = manager.add_message(
            session_id,
            role="bot",
            content="I can help with that.",
            intent="order_tracking"
        )

        assert message.intent == "order_tracking"

    def test_get_message_history(self, manager):
        """History comes back in the order it was written."""
        session_id = manager.create_session()

        # Add multiple messages
        manager.add_message(session_id, "customer", "Hello")
        manager.add_message(session_id, "bot", "Hi there!")
        manager.add_message(session_id, "customer", "How are you?")

        history = manager.get_message_history(session_id)
        assert len(history) == 3
        assert history[0]["role"] == "customer"
        assert history[1]["role"] == "bot"

    def test_message_history_with_limit(self, manager):
        """A limit returns the most recent messages, not the first."""
        session_id = manager.create_session()

        # Add 5 messages
        for i in range(5):
            manager.add_message(session_id, "customer", f"Message {i}")

        history = manager.get_message_history(session_id, limit=2)
        assert len(history) == 2
        assert "4" in history[-1]["content"]  # Last message


    def test_get_recent_context(self, manager):
        """The context string carries both speakers and the known ids."""
        session_id = manager.create_session()

        manager.add_message(session_id, "customer", "Where is my order?")
        manager.add_message(session_id, "bot", "I can help you track that.")
        manager.update_context(
            session_id,
            customer_id="CUST-10000",
            current_order_id="ORD-100000"
        )

        context_str = manager.get_recent_context(session_id)
        assert "CUSTOMER:" in context_str or "Customer:" in context_str.upper()
        assert "BOT:" in context_str or "Bot:" in context_str.upper()
        assert "CUST-10000" in context_str

    def test_end_session(self, manager):
        """Ending a session makes it unreachable."""
        session_id = manager.create_session()

        result = manager.end_session(session_id)
        assert result is True

        # Session should no longer exist
        conversation = manager.get_session(session_id)
        assert conversation is None

    def test_get_all_sessions(self, manager):
        """Every live session is listed."""
        manager.create_session()
        manager.create_session()

        all_sessions = manager.get_all_sessions()
        assert len(all_sessions) >= 2

    def test_the_session_cap_actually_caps(self):
        """max_sessions was ignored for any value below 100.

        `_cleanup_old_sessions` defaulted `keep_recent` to a hardcoded 100 and
        returned early while the count was under it, so a manager capped at 10
        grew without limit. The test that used to live here created 15
        sessions, asserted the list was non-empty and passed -- it would have
        passed with cleanup deleted entirely.
        """
        manager = ConversationManager(max_sessions=10)
        for _ in range(15):
            manager.create_session()

        assert len(manager.get_all_sessions()) == 10

    def test_the_cap_keeps_the_newest_sessions(self):
        """Evicting the conversation someone is still typing into is the bad case."""
        manager = ConversationManager(max_sessions=5)
        ids = [manager.create_session() for _ in range(8)]

        surviving = {s["session_id"] for s in manager.get_all_sessions()}
        assert surviving == set(ids[-5:])
        assert not surviving & set(ids[:3]), "an older session outlived a newer one"


# ============================================================================
# Response Generator Tests
# ============================================================================

class TestResponseGenerator:
    """What every reply carries, whatever answered it."""

    @pytest.mark.asyncio
    async def test_generate_response(self, mcp_tools):
        """A turn always produces text, even with no model configured."""
        generator = ResponseGenerator(mcp_tools)
        context = ConversationContext()
        history = []

        response = await generator.generate_response(
            "What is your return policy?",
            context,
            history
        )

        assert response is not None
        assert "response" in response
        assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_response_intent_classification(self, mcp_tools):
        """Every reply is labelled with a subject."""
        generator = ResponseGenerator(mcp_tools)
        context = ConversationContext()

        response = await generator.generate_response(
            "Where is my order?",
            context,
            []
        )

        assert "intent" in response
        assert response["intent"] is not None

    @pytest.mark.asyncio
    async def test_response_confidence_score(self, mcp_tools):
        """Confidence is present and in range."""
        generator = ResponseGenerator(mcp_tools)
        context = ConversationContext()

        response = await generator.generate_response(
            "Hello",
            context,
            []
        )

        assert "confidence" in response
        assert 0 <= response["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_response_with_actions(self, mcp_tools):
        """Buttons are always a list, never None."""
        generator = ResponseGenerator(mcp_tools)
        context = ConversationContext()

        response = await generator.generate_response(
            "I want to return something",
            context,
            []
        )

        assert "suggested_actions" in response
        assert isinstance(response["suggested_actions"], list)


# ============================================================================
# Multi-turn Conversation Tests
# ============================================================================

class TestMultiTurnConversation:
    """What survives from one turn to the next."""


    def test_context_persistence(self, manager):
        """Context set on one turn is still there after another message."""
        session_id = manager.create_session()

        # Set initial context
        manager.update_context(
            session_id,
            customer_id="CUST-10000",
            current_order_id="ORD-100000"
        )

        # Add message
        manager.add_message(session_id, "customer", "Hello")

        # Context should still be there
        context = manager.get_context(session_id)
        assert context.customer_id == "CUST-10000"
        assert context.current_order_id == "ORD-100000"

    def test_multiple_sessions_isolation(self, manager):
        """Two conversations never see the other one's history."""
        session1 = manager.create_session(customer_id="CUST-10000")
        session2 = manager.create_session(customer_id="CUST-10001")

        # Add messages to each
        manager.add_message(session1, "customer", "Message 1")
        manager.add_message(session2, "customer", "Message 2")

        # Verify isolation
        history1 = manager.get_message_history(session1)
        history2 = manager.get_message_history(session2)

        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["content"] != history2[0]["content"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestChatIntegration:
    """The conversation object as the API serialises it."""

    def test_conversation_dataclass_to_dict(self, manager):
        """to_dict carries the fields the API promises."""
        session_id = manager.create_session()
        manager.add_message(session_id, "customer", "Hello")
        manager.add_message(session_id, "bot", "Hi!")

        conversation = manager.get_session(session_id)
        conv_dict = conversation.to_dict()

        assert "session_id" in conv_dict
        assert "messages" in conv_dict
        assert "context" in conv_dict
        assert conv_dict["message_count"] == 2

    def test_message_timestamps(self, manager):
        """Timestamps are ISO strings, which the transcript endpoint returns as-is."""
        session_id = manager.create_session()

        message = manager.add_message(session_id, "customer", "Test")

        assert message.timestamp is not None
        # Should be ISO format
        assert "T" in message.timestamp or "-" in message.timestamp

    def test_conversation_with_entities(self, manager):
        """Entities attached to a message survive the round trip."""
        session_id = manager.create_session()

        entities = [
            {"type": "ORDER_ID", "value": "ORD-100000"},
            {"type": "PRODUCT", "value": "Widget"}
        ]

        message = manager.add_message(
            session_id,
            "customer",
            "I want to return my order",
            entities=entities
        )

        assert message.entities == entities

    def test_conversation_with_metadata(self, manager):
        """Arbitrary metadata rides along without being reshaped."""
        session_id = manager.create_session()

        metadata = {
            "user_ip": "192.168.1.1",
            "device": "mobile"
        }

        message = manager.add_message(
            session_id,
            "customer",
            "Hello",
            metadata=metadata
        )

        assert message.metadata == metadata


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
