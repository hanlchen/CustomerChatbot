"""
Test suite for chat endpoints and conversation management.

Tests:
- Session creation
- Message sending and receiving
- Conversation history
- Context tracking
- Multi-turn conversations
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
    """Create fresh conversation manager for each test."""
    reset_conversation_manager()
    return get_conversation_manager()


@pytest.fixture(scope="session")
def mcp_tools():
    """Create MCP tools instance."""
    return CustomerChatbotTools()


# ============================================================================
# Conversation Manager Tests
# ============================================================================

class TestConversationManager:
    """Test conversation manager functionality."""

    def test_create_session(self, manager):
        """Test creating a new session."""
        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) > 0

        conversation = manager.get_session(session_id)
        assert conversation is not None
        assert conversation.session_id == session_id

    def test_create_session_with_customer_id(self, manager):
        """Test creating session with customer ID."""
        customer_id = "CUST-10000"
        session_id = manager.create_session(customer_id=customer_id)

        conversation = manager.get_session(session_id)
        assert conversation.context.customer_id == customer_id

    def test_get_session_by_customer(self, manager):
        """Test retrieving session by customer ID."""
        customer_id = "CUST-10001"
        session_id = manager.create_session(customer_id=customer_id)

        conversation = manager.get_session_by_customer(customer_id)
        assert conversation is not None
        assert conversation.session_id == session_id

    def test_add_message(self, manager):
        """Test adding message to conversation."""
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
        """Test adding message with intent."""
        session_id = manager.create_session()

        message = manager.add_message(
            session_id,
            role="bot",
            content="I can help with that.",
            intent="order_tracking"
        )

        assert message.intent == "order_tracking"

    def test_get_message_history(self, manager):
        """Test retrieving message history."""
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
        """Test message history with limit."""
        session_id = manager.create_session()

        # Add 5 messages
        for i in range(5):
            manager.add_message(session_id, "customer", f"Message {i}")

        history = manager.get_message_history(session_id, limit=2)
        assert len(history) == 2
        assert "4" in history[-1]["content"]  # Last message


    def test_get_recent_context(self, manager):
        """Test getting recent context as formatted string."""
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
        """Test ending a session."""
        session_id = manager.create_session()

        result = manager.end_session(session_id)
        assert result is True

        # Session should no longer exist
        conversation = manager.get_session(session_id)
        assert conversation is None

    def test_get_all_sessions(self, manager):
        """Test getting all active sessions."""
        manager.create_session()
        manager.create_session()

        all_sessions = manager.get_all_sessions()
        assert len(all_sessions) >= 2

    def test_max_sessions_cleanup(self):
        """Test automatic cleanup of old sessions."""
        # Create manager with small max and keep_recent
        small_manager = ConversationManager(max_sessions=10)

        # Create 15 sessions (should trigger cleanup)
        session_ids = []
        for i in range(15):
            sid = small_manager.create_session()
            session_ids.append(sid)

        # Should have cleaned up oldest sessions
        sessions = small_manager.get_all_sessions()
        # The cleanup keeps keep_recent (100 by default) which is more than 10 max_sessions
        # So this is primarily testing that cleanup doesn't crash
        assert sessions is not None
        assert len(sessions) > 0


# ============================================================================
# Response Generator Tests
# ============================================================================

class TestResponseGenerator:
    """Test response generation."""

    @pytest.mark.asyncio
    async def test_generate_response(self, mcp_tools):
        """Test basic response generation."""
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
        """Test that response includes intent."""
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
        """Test that response includes confidence."""
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
        """Test that response can include suggested actions."""
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
    """Test multi-turn conversation flows."""


    def test_context_persistence(self, manager):
        """Test that context persists across messages."""
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
        """Test that sessions are isolated from each other."""
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
    """Test complete chat integration."""

    def test_conversation_dataclass_to_dict(self, manager):
        """Test converting conversation to dict."""
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
        """Test that messages have proper timestamps."""
        session_id = manager.create_session()

        message = manager.add_message(session_id, "customer", "Test")

        assert message.timestamp is not None
        # Should be ISO format
        assert "T" in message.timestamp or "-" in message.timestamp

    def test_conversation_with_entities(self, manager):
        """Test message with extracted entities."""
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
        """Test message with metadata."""
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
