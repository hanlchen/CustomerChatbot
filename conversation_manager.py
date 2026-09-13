"""
Sessions, history and context for CustomerChatbot.

Infrastructure only. This module used to also contain a keyword-and-if-ladder
rules engine that answered turns itself; the agents in `llm_agent.py` do that
now, so what is left here is the plumbing they run on: who is talking, what
has been said, and what the tools have established so far.

    ConversationManager   sessions, history, expiry
    ConversationContext   facts the tools returned, carried across turns
    ResponseGenerator     hands a turn to the agents and stamps the reply
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "customer" or "bot"
    content: str
    timestamp: str
    intent: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationContext:
    """Facts the tools established, carried across turns.

    Only what a tool actually returned lives here. The agents read the
    conversation itself for everything else -- what the customer meant, what
    they already said, what they are waiting on -- because they have the whole
    transcript and a flag would only be a worse copy of it.
    """
    customer_id: Optional[str] = None
    current_order_id: Optional[str] = None
    # Order IDs from the last lookup, newest first. Used for the buttons and
    # to remind an agent what it already knows.
    known_order_ids: List[str] = field(default_factory=list)
    # Verdict from the last eligibility check, so the buttons offer a return
    # or a damage claim rather than both.
    current_order_returnable: Optional[bool] = None

    # -- identity ---------------------------------------------------------
    # Set only by a successful phone confirmation. Every account tool is gated
    # on it, so it is the single thing standing between a conversation and
    # somebody's order history. Never set from anything the customer typed.
    verified_customer_id: Optional[str] = None
    verified_at: Optional[float] = None
    # The verification attempt currently in flight. The token is opaque and
    # server-side; the customer id behind it is not exposed until they pass.
    challenge_token: Optional[str] = None
    verification_failed: bool = False
    # How many times we have asked this conversation who it is, and how many
    # identifiers came back unrecognised. Neither stores what was typed --
    # an unverified conversation holds no identity, and a wrong one is still
    # somebody's data. They exist because without them every unverified turn
    # rebuilds an identical context note, and the agent asks the identical
    # question: "i dont have my cust number" got the opening sentence back
    # word for word, because as far as the prompt was concerned it had never
    # been asked.
    identity_asks: int = 0
    failed_lookups: int = 0

    @property
    def is_verified(self) -> bool:
        """Whether this conversation has proved who it is talking to.

        Time-limited: a session left open on a shared machine should not still
        be able to read an account an hour later.
        """
        if not self.verified_customer_id or self.verified_at is None:
            return False
        import security

        return (time.time() - self.verified_at) <= security.VERIFICATION_TTL_SECONDS


@dataclass
class Conversation:
    """Represents a complete conversation session."""
    session_id: str
    created_at: str
    messages: List[Message] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)
    last_message_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_message_time": self.last_message_time,
            "message_count": len(self.messages),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "intent": m.intent
                } for m in self.messages
            ],
            "context": {
                "customer_id": self.context.customer_id,
                "current_order_id": self.context.current_order_id,
                "known_order_ids": list(self.context.known_order_ids),
            }
        }


class ConversationManager:
    """Manages conversations and session state."""

    def __init__(self, max_sessions: int = 1000):
        """Initialize conversation manager."""
        self.conversations: Dict[str, Conversation] = {}
        self.max_sessions = max_sessions
        self.customer_sessions: Dict[str, str] = {}  # Map customer_id to session_id

    def create_session(self, customer_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.

        Args:
            customer_id: Optional customer ID to associate with session

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())[:8]

        conversation = Conversation(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        if customer_id:
            conversation.context.customer_id = customer_id
            self.customer_sessions[customer_id] = session_id

        self.conversations[session_id] = conversation

        # Cleanup old sessions if needed
        if len(self.conversations) > self.max_sessions:
            self._cleanup_old_sessions()

        return session_id

    def get_session(self, session_id: str) -> Optional[Conversation]:
        """Get a conversation by session ID."""
        return self.conversations.get(session_id)

    def get_session_by_customer(self, customer_id: str) -> Optional[Conversation]:
        """Get customer's active session."""
        session_id = self.customer_sessions.get(customer_id)
        if session_id:
            return self.conversations.get(session_id)
        return None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            session_id: Session to add message to
            role: "customer" or "bot"
            content: Message content
            intent: Detected intent (for bot messages)
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            The added message
        """
        conversation = self.conversations.get(session_id)
        if not conversation:
            raise ValueError(f"Session {session_id} not found")

        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            intent=intent,
            entities=entities,
            metadata=metadata
        )

        conversation.messages.append(message)
        conversation.last_message_time = datetime.now(timezone.utc).isoformat()

        return message

    def update_context(
        self,
        session_id: str,
        customer_id: Optional[str] = None,
        current_order_id: Optional[str] = None,
        current_product_id: Optional[str] = None,
        last_intent: Optional[str] = None,
        customer_data: Optional[Dict[str, Any]] = None,
        order_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update conversation context."""
        conversation = self.conversations.get(session_id)
        if not conversation:
            raise ValueError(f"Session {session_id} not found")

        if customer_id:
            conversation.context.customer_id = customer_id
        if current_order_id:
            conversation.context.current_order_id = current_order_id
        if current_product_id:
            conversation.context.current_product_id = current_product_id
        if last_intent:
            conversation.context.last_intent = last_intent
        if customer_data:
            conversation.context.customer_data = customer_data
        if order_data:
            conversation.context.order_data = order_data

    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context."""
        conversation = self.conversations.get(session_id)
        if conversation:
            return conversation.context
        return None

    def get_message_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get message history for a session.

        Args:
            session_id: Session ID
            limit: Max messages to return (None = all)

        Returns:
            List of messages
        """
        conversation = self.conversations.get(session_id)
        if not conversation:
            return []

        messages = conversation.messages
        if limit:
            messages = messages[-limit:]

        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "intent": m.intent
            } for m in messages
        ]

    def get_recent_context(self, session_id: str, max_messages: int = 10) -> str:
        """
        Get recent conversation context as formatted string.
        Useful for feeding to response generation.
        """
        conversation = self.conversations.get(session_id)
        if not conversation:
            return ""

        messages = conversation.messages[-max_messages:]
        context_str = "Recent conversation:\n"

        for msg in messages:
            role = msg.role.upper()
            context_str += f"{role}: {msg.content}\n"

        # Add current context
        ctx = conversation.context
        if ctx.customer_id:
            context_str += f"\nCustomer ID: {ctx.customer_id}\n"
        if ctx.current_order_id:
            context_str += f"Current Order: {ctx.current_order_id}\n"
        if ctx.known_order_ids:
            context_str += f"Known Orders: {', '.join(ctx.known_order_ids)}\n"

        return context_str

    def end_session(self, session_id: str) -> bool:
        """End a conversation session."""
        if session_id in self.conversations:
            conversation = self.conversations[session_id]
            if conversation.context.customer_id:
                del self.customer_sessions[conversation.context.customer_id]
            del self.conversations[session_id]
            return True
        return False

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all active sessions."""
        return [conv.to_dict() for conv in self.conversations.values()]

    def _cleanup_old_sessions(self, keep_recent: int = 100) -> None:
        """Remove oldest sessions when max is reached."""
        if len(self.conversations) <= keep_recent:
            return

        # Sort by last message time and remove oldest
        sorted_sessions = sorted(
            self.conversations.items(),
            key=lambda x: x[1].last_message_time
        )

        for session_id, _ in sorted_sessions[:-keep_recent]:
            self.end_session(session_id)


class ResponseGenerator:
    """Hands a turn to the agents, and stamps the fields every reply carries.

    There is no fallback engine any more. When the model is unreachable the
    customer is told that plainly, because a support bot that quietly answers
    from a keyword table is worse than one that says it is having trouble --
    the first kind is wrong without telling you.
    """

    UNAVAILABLE = (
        "I'm sorry — I can't reach our systems right now, so I don't want to "
        "guess at an answer. Please try again in a moment. If it's urgent, "
        "support@example.com or 1-800-555-0100 will get you to a person."
    )

    def __init__(self, tools, llm_agent=None):
        self.tools = tools

        if llm_agent is not None:
            self.llm_agent = llm_agent
        else:
            try:
                from llm_agent import LLMAgent
                self.llm_agent = LLMAgent(tools)
            except Exception as exc:  # pragma: no cover - import guard
                logging.getLogger(__name__).warning(
                    "LLM agent unavailable: %s", exc
                )
                self.llm_agent = None

    @property
    def engine(self) -> str:
        """Whether the next turn can be answered at all."""
        if self.llm_agent is not None and self.llm_agent.available:
            return "llm"
        return "unavailable"

    @property
    def engine_status(self) -> str:
        if self.llm_agent is None:
            return "unavailable (llm_agent could not be imported)"
        if self.llm_agent.available:
            return f"llm — {self.llm_agent.status}"
        return f"unavailable — {self.llm_agent.status}"

    async def generate_response(
        self,
        customer_message: str,
        context: ConversationContext,
        message_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Produce a reply, or say honestly that we cannot."""
        if self.llm_agent is not None and self.llm_agent.available:
            try:
                return self._finish(await self.llm_agent.respond(
                    customer_message, context, message_history
                ))
            except Exception as exc:
                # Full traceback, not just the message: "every turn fails" is
                # only diagnosable if the first one left a trail.
                logging.getLogger(__name__).error(
                    "turn failed: %s: %s", type(exc).__name__, exc, exc_info=True)
                return self._unavailable(f"{type(exc).__name__}: {exc}")

        detail = self.llm_agent.status if self.llm_agent else "no agent"
        return self._unavailable(detail)

    @classmethod
    def _unavailable(cls, detail: str) -> Dict[str, Any]:
        """The honest failure. `detail` is for the operator, not the customer."""
        return {
            "response": cls.UNAVAILABLE,
            "intent": "general_inquiry",
            "confidence": 0.0,
            "suggested_actions": ["Contact support"],
            "actions_taken": [],
            "answer_source": "none",
            "agent": "none",
            "engine": "unavailable",
            "error": detail,
        }

    @staticmethod
    def _finish(reply: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in fields every reply must carry."""
        if "answer_source" not in reply:
            from llm_agent import LLMAgent

            reply["answer_source"] = LLMAgent._answer_source(
                reply.get("actions_taken") or []
            )
        # `agent` names the specialist (triage / data / policy); `engine` says
        # whether a model produced it. The UI badge keys off `engine`, so the
        # two cannot share a field.
        reply.setdefault("engine", "llm")
        return reply


# Global conversation manager instance
_manager = None

def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance."""
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager

def reset_conversation_manager() -> None:
    """Reset the conversation manager (for testing)."""
    global _manager
    _manager = ConversationManager()
