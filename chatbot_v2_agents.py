"""Policy-Aware SQL + RAG Chatbot v2 - Using OpenAI Agents SDK with Tracing."""

import os
import re
import sqlite3
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import streamlit as st
import openai

from agents import Agent, Runner, function_tool, trace, handoff

load_dotenv(override=True)


# ============= Metrics Tracking =============

class MetricsTracker:
    """Track API usage metrics for the agents."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.intent_classifications = 0
        self.policy_searches = 0
        self.data_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.avg_latency_ms = 0.0
        self._latencies = []

    def record_request(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Record an API request with token counts."""
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens

    def record_latency(self, latency_ms: float):
        """Record request latency."""
        self._latencies.append(latency_ms)
        if self._latencies:
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def record_intent(self):
        """Record an intent classification."""
        self.intent_classifications += 1

    def record_policy_search(self):
        """Record a policy search."""
        self.policy_searches += 1

    def record_data_query(self):
        """Record a data query."""
        self.data_queries += 1

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    def record_error(self):
        """Record an error."""
        self.errors += 1

    def get_summary(self) -> dict:
        """Get metrics summary."""
        cache_total = self.cache_hits + self.cache_misses
        cache_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "intent_classifications": self.intent_classifications,
            "policy_searches": self.policy_searches,
            "data_queries": self.data_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "errors": self.errors,
            "avg_latency_ms": f"{self.avg_latency_ms:.0f}"
        }


def get_metrics() -> MetricsTracker:
    """Get or create metrics tracker in session state."""
    if "metrics_tracker" not in st.session_state:
        st.session_state.metrics_tracker = MetricsTracker()
    return st.session_state.metrics_tracker


# ============= Response Cache (Policy Questions Only) =============

# Keywords that indicate a policy question (cacheable)
POLICY_KEYWORDS = [
    "policy", "policies", "return", "refund", "cancel", "cancellation",
    "warranty", "shipping", "delivery", "exchange", "rules", "allowed",
    "can i", "am i able", "is it possible", "how do i", "what is your",
    "what's your", "do you", "are you able"
]


def is_policy_question(message: str) -> bool:
    """Check if message is a policy question (cacheable) vs data query (not cacheable)."""
    msg_lower = message.lower()

    # Data queries - NOT cacheable (order status, my orders, etc.)
    data_keywords = ["my order", "order status", "track", "where is", "when will",
                     "what did i", "my purchase", "order number", "order #"]
    if any(kw in msg_lower for kw in data_keywords):
        return False

    # Policy questions - cacheable
    if any(kw in msg_lower for kw in POLICY_KEYWORDS):
        return True

    return False


class ResponseCache:
    """Cache for POLICY responses only (5 min TTL). Data queries are not cached."""

    def __init__(self, ttl_minutes: int = 5):
        self.cache = {}  # key -> {"response": str, "expires": datetime}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.hits = 0
        self.misses = 0

    def _make_key(self, message: str) -> str:
        """Create cache key from normalized message (not customer-specific for policies)."""
        normalized = message.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, customer_id: int, message: str) -> Optional[str]:
        """Get cached response if exists, not expired, and is a policy question."""
        # Only cache policy questions
        if not is_policy_question(message):
            return None

        key = self._make_key(message)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires"]:
                self.hits += 1
                return entry["response"]
            else:
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, customer_id: int, message: str, response: str):
        """Cache a policy response only."""
        # Only cache policy questions
        if not is_policy_question(message):
            return

        key = self._make_key(message)
        self.cache[key] = {
            "response": response,
            "expires": datetime.now() + self.ttl
        }

    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        now = datetime.now()
        valid = sum(1 for v in self.cache.values() if now < v["expires"])
        return {"total": len(self.cache), "valid": valid, "hits": self.hits, "misses": self.misses}


# Cache will be stored in session state to persist across reruns
def get_cache() -> ResponseCache:
    """Get or create cache in session state."""
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = ResponseCache(ttl_minutes=5)
    return st.session_state.response_cache


# ============= Rate Limiter =============

class RateLimiter:
    """Simple rate limiter to prevent API abuse."""

    def __init__(self, max_requests: int = 20, window_minutes: int = 1):
        """
        Args:
            max_requests: Maximum requests allowed per window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window = timedelta(minutes=window_minutes)
        self.requests = {}  # customer_id -> list of timestamps

    def is_allowed(self, customer_id: int) -> tuple[bool, str]:
        """
        Check if request is allowed for this customer.

        Returns:
            (allowed: bool, message: str)
        """
        now = datetime.now()
        window_start = now - self.window

        # Get or create request list for customer
        if customer_id not in self.requests:
            self.requests[customer_id] = []

        # Clean old requests outside window
        self.requests[customer_id] = [
            ts for ts in self.requests[customer_id] if ts > window_start
        ]

        # Check limit
        current_count = len(self.requests[customer_id])
        if current_count >= self.max_requests:
            remaining_time = self.requests[customer_id][0] + self.window - now
            seconds = int(remaining_time.total_seconds())
            return False, f"Rate limit exceeded. Please wait {seconds} seconds before trying again."

        # Record this request
        self.requests[customer_id].append(now)
        return True, ""

    def get_remaining(self, customer_id: int) -> int:
        """Get remaining requests for this customer in current window."""
        now = datetime.now()
        window_start = now - self.window

        if customer_id not in self.requests:
            return self.max_requests

        valid_requests = [ts for ts in self.requests[customer_id] if ts > window_start]
        return max(0, self.max_requests - len(valid_requests))


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter in session state."""
    if "rate_limiter" not in st.session_state:
        # 20 requests per minute per customer
        st.session_state.rate_limiter = RateLimiter(max_requests=20, window_minutes=1)
    return st.session_state.rate_limiter


# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "policy-chatbot")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database", "chatbot.db")
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ============= Pinecone Functions =============

def get_pinecone_index():
    """Get Pinecone index for policy search."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Pinecone error: {e}")
        return None


def get_embedding(text: str) -> list[float]:
    """Get embedding for text using OpenAI."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# ============= Pre-loaded Policy Cache (Fast!) =============

def get_policy_cache() -> dict:
    """Get or create policy cache in session state - loads once at startup."""
    if "policy_cache" not in st.session_state:
        from policies.documents import POLICY_DOCUMENTS
        st.session_state.policy_cache = {
            "public": [p for p in POLICY_DOCUMENTS if p["metadata"].get("visibility") == "public"],
            "internal": [p for p in POLICY_DOCUMENTS if p["metadata"].get("visibility") == "internal"],
            "all": POLICY_DOCUMENTS
        }
    return st.session_state.policy_cache


def search_policies_fast(query: str, top_k: int = 3, visibility: str = "public") -> list[dict]:
    """Fast keyword-based policy search using pre-loaded cache. No API calls!"""
    cache = get_policy_cache()
    policies = cache.get(visibility, cache["public"])

    query_lower = query.lower()
    keywords = query_lower.split()

    # Score policies by keyword matches
    scored = []
    for policy in policies:
        content_lower = policy["content"].lower()
        topic = policy["metadata"].get("topic", "").lower()

        # Count keyword matches
        score = 0
        for kw in keywords:
            if kw in content_lower:
                score += content_lower.count(kw)
            if kw in topic:
                score += 5  # Topic match = higher weight

        # Boost for common policy-related keywords
        policy_keywords = {
            "cancel": ["cancel", "cancellation"],
            "return": ["return", "refund"],
            "ship": ["ship", "shipping", "delivery"],
            "warranty": ["warranty", "guarantee"],
        }
        for category, terms in policy_keywords.items():
            if any(t in query_lower for t in terms) and category in topic:
                score += 10

        if score > 0:
            scored.append({
                "content": policy["content"],
                "topic": policy["metadata"].get("topic", ""),
                "scope": policy["metadata"].get("scope", ""),
                "visibility": policy["metadata"].get("visibility", "public"),
                "score": score
            })

    # Sort by score and return top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def search_pinecone_policies(query: str, top_k: int = 3, visibility: str = "public") -> list[dict]:
    """Search Pinecone for relevant policies (slower, used as fallback).

    Args:
        query: Search query
        top_k: Number of results
        visibility: 'public' for customer-facing, 'internal' for internal only, 'all' for both
    """
    index = get_pinecone_index()
    if not index:
        return []

    try:
        query_embedding = get_embedding(query)

        # Build filter based on visibility
        filter_dict = None
        if visibility != "all":
            filter_dict = {"visibility": {"$eq": visibility}}

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        policies = []
        for match in results.matches:
            policies.append({
                "content": match.metadata.get("content", ""),
                "topic": match.metadata.get("topic", ""),
                "scope": match.metadata.get("scope", ""),
                "visibility": match.metadata.get("visibility", "public"),
                "score": match.score
            })
        return policies
    except Exception as e:
        print(f"Pinecone search error: {e}")
        return []


# ============= Database Functions =============

def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(query: str) -> list[dict]:
    """Execute SQL query and return results."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        if cursor.description:
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        return []
    finally:
        conn.close()


def validate_customer_email(email: str) -> Optional[dict]:
    """Validate customer by email. Returns customer info including phone for 2FA."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT customer_id, full_name, phone FROM customers WHERE email = ?",
            (email.lower().strip(),)
        )
        row = cursor.fetchone()
        if row:
            return {"customer_id": row[0], "full_name": row[1], "phone": row[2]}
        return None
    finally:
        conn.close()


def validate_customer_phone(customer_id: int, phone: str) -> bool:
    """Validate phone number matches customer record (2FA step)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Normalize phone: remove spaces, dashes, etc.
        phone_normalized = ''.join(c for c in phone if c.isdigit())
        cursor.execute(
            "SELECT phone FROM customers WHERE customer_id = ?",
            (customer_id,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            db_phone_normalized = ''.join(c for c in row[0] if c.isdigit())
            return phone_normalized == db_phone_normalized
        return False
    finally:
        conn.close()


# ============= Intent Classification Tool =============

@function_tool
def classify_intent(message: str) -> str:
    """
    Classify the user's message into an intent category.

    Args:
        message: The user's message to classify

    Returns:
        One of: GREETING, POLICY_QUESTION, DATA_QUESTION, ACTION_REQUEST, OTHER
    """
    classification_prompt = """Classify this message into ONE category:
- GREETING: Hello, hi, thanks, bye, etc.
- POLICY_QUESTION: Questions about policies, rules, what's allowed, return policy, cancellation policy
- DATA_QUESTION: Questions about orders, products, status, what did I buy, my orders
- ACTION_REQUEST: Requests to cancel, change, update, or do something
- OTHER: Anything else

Message: "{message}"

Respond with ONLY the category name."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": classification_prompt.format(message=message)}],
        temperature=0,
        max_tokens=20
    )

    # Track metrics
    metrics = get_metrics()
    metrics.record_intent()
    if response.usage:
        metrics.record_request(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

    intent = response.choices[0].message.content.strip().upper()
    valid_intents = ["GREETING", "POLICY_QUESTION", "DATA_QUESTION", "ACTION_REQUEST", "OTHER"]

    return intent if intent in valid_intents else "OTHER"


# ============= Policy Search Tool (Pinecone RAG) =============

@function_tool
def search_policies(query: str) -> str:
    """
    Search the policy database for relevant PUBLIC policies.
    Use this for any policy-related questions from customers.
    Only returns customer-facing policies, not internal policies.

    Args:
        query: The policy question or topic to search for

    Returns:
        Relevant PUBLIC policy information from the knowledge base
    """
    # Track metrics
    metrics = get_metrics()
    metrics.record_policy_search()

    # Use FAST local search - no API calls!
    policies = search_policies_fast(query, top_k=3, visibility="public")

    if not policies:
        return "No specific policy found. Please contact customer support for more information."

    result = "Relevant Policies Found:\n\n"
    for i, policy in enumerate(policies, 1):
        result += f"--- Policy {i} (Topic: {policy['topic']}) ---\n"
        result += f"{policy['content']}\n\n"

    return result


def search_internal_policies(query: str) -> str:
    """
    Search for INTERNAL policies (not exposed as a tool to agents).
    Used internally by the system for guidance, not shared with customers.
    """
    # Use FAST local search
    policies = search_policies_fast(query, top_k=2, visibility="internal")

    if not policies:
        return ""

    result = "Internal Guidelines:\n"
    for policy in policies:
        result += f"{policy['content']}\n"

    return result


# ============= Data Agent Tools =============

@function_tool
def search_customer_orders(customer_id: int, query_type: str) -> str:
    """
    Search for customer orders in the database.

    Args:
        customer_id: The customer's ID
        query_type: Type of query - 'all', 'recent', 'last_month'
    """
    # Track metrics
    metrics = get_metrics()
    metrics.record_data_query()

    queries = {
        "all": f"""
            SELECT o.order_id, o.order_date, o.status, o.total_amount,
                   GROUP_CONCAT(p.name, ', ') as products
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
            WHERE o.customer_id = {customer_id}
            GROUP BY o.order_id
            ORDER BY o.order_date DESC
            LIMIT 10
        """,
        "recent": f"""
            SELECT o.order_id, o.order_date, o.status, o.total_amount,
                   GROUP_CONCAT(p.name, ', ') as products
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
            WHERE o.customer_id = {customer_id}
            GROUP BY o.order_id
            ORDER BY o.order_date DESC
            LIMIT 5
        """,
        "last_month": f"""
            SELECT o.order_id, o.order_date, o.status, o.total_amount,
                   GROUP_CONCAT(p.name, ', ') as products
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
            WHERE o.customer_id = {customer_id}
              AND o.order_date >= date('now', 'start of month', '-1 month')
              AND o.order_date < date('now', 'start of month')
            GROUP BY o.order_id
            LIMIT 10
        """
    }

    query = queries.get(query_type, queries["all"])
    results = execute_query(query)

    if not results:
        return "No orders found."

    return json.dumps(results, indent=2, default=str)


@function_tool
def get_order_details(customer_id: int, order_id: int) -> str:
    """
    Get detailed information about a specific order.

    Args:
        customer_id: The customer's ID (for verification)
        order_id: The order ID to look up
    """
    query = f"""
        SELECT o.order_id, o.order_date, o.status, o.total_amount, o.shipping_address,
               p.name as product, oi.quantity, oi.unit_price
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.order_id = {order_id} AND o.customer_id = {customer_id}
    """
    results = execute_query(query)

    if not results:
        return f"Order {order_id} not found or does not belong to this customer."

    return json.dumps(results, indent=2, default=str)


def is_safe_sql_query(sql_query: str) -> tuple[bool, str]:
    """Validate that a SQL query is a safe read-only SELECT.

    Returns:
        (is_safe, error_message). error_message is empty when safe.
    """
    sql_upper = sql_query.upper().strip()

    if not sql_upper.startswith("SELECT"):
        return False, "Error: Only SELECT queries are allowed."

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "--", ";--"]
    for word in forbidden:
        if word in sql_upper:
            return False, f"Error: Query contains forbidden keyword: {word}"

    if "EMAIL" in sql_upper and "SELECT" in sql_upper:
        return False, "Error: Cannot query email addresses for privacy."

    return True, ""


@function_tool
def execute_custom_sql(customer_id: int, sql_query: str) -> str:
    """
    Execute a custom SQL query for the customer.
    IMPORTANT: Query must filter by customer_id for security.

    Args:
        customer_id: The customer's ID
        sql_query: The SQL SELECT query to execute
    """
    safe, error = is_safe_sql_query(sql_query)
    if not safe:
        return error

    try:
        results = execute_query(sql_query)
        results = [{k: v for k, v in row.items() if k.lower() != 'email'} for row in results]

        if not results:
            return "No results found."
        return json.dumps(results[:50], indent=2, default=str)
    except Exception as e:
        return f"Error executing query: {str(e)}"


# ============= Policy Agent Tools =============

@function_tool
def check_cancellation_policy(order_id: int, customer_id: int) -> str:
    """
    Check if an order can be cancelled based on its status and policy.

    Args:
        order_id: The order ID to check
        customer_id: The customer's ID (for verification)
    """
    # First get order status from DB
    query = f"""
        SELECT order_id, status, order_date
        FROM orders
        WHERE order_id = {order_id} AND customer_id = {customer_id}
    """
    results = execute_query(query)

    if not results:
        return f"Order {order_id} not found or does not belong to you."

    order = results[0]
    status = order["status"]

    # Use FAST local search for cancellation policy
    policy_results = search_policies_fast("order cancellation policy", top_k=1)

    policy_text = ""
    if policy_results:
        policy_text = policy_results[0]["content"]

    # Also include status-specific guidance
    status_guidance = {
        "PLACED": "Based on status PLACED: This order CAN be cancelled. Contact customer support.",
        "PROCESSING": "Based on status PROCESSING: Cancellation may be possible but not guaranteed. Contact support immediately.",
        "SHIPPED": "Based on status SHIPPED: This order CANNOT be cancelled. Wait for delivery and initiate return if needed.",
        "DELIVERED": "Based on status DELIVERED: Cannot be cancelled. May return within 30 days."
    }

    guidance = status_guidance.get(status, "Unknown status. Contact customer support.")

    return f"""Order {order_id} Status: {status}

{guidance}

--- Full Cancellation Policy ---
{policy_text}"""


# ============= Agent Definitions =============

# Data Agent - handles order queries
data_agent_instructions = """
You are a Data Assistant for customer support. You help customers find information about their orders.

Customer ID: {customer_id}
Customer Name: {customer_name}

Your job:
1. Use search_customer_orders to find orders (use query_type: 'all', 'recent', or 'last_month')
2. Use get_order_details for specific order info
3. Use execute_custom_sql for complex queries (always include customer_id filter!)

Present data in a friendly, readable format. Never expose email addresses.
Address the customer by name.
"""

# Policy Agent - handles policy questions and action requests
policy_agent_instructions = """
You are a Policy Expert for customer support. You help customers understand store policies.

Customer ID: {customer_id}
Customer Name: {customer_name}

Your job:
1. ALWAYS use the search_policies tool to find relevant policies from our knowledge base
2. For cancellation requests: Use check_cancellation_policy to check order status and policy
3. Explain policies clearly based on what you find in the knowledge base
4. Direct customers to customer support for any actions

IMPORTANT: You CANNOT actually cancel orders or process refunds. Only explain policies and direct to support.
Always search the policy database before answering policy questions.
"""

# Main Support Agent with intent classification
support_agent_instructions = """
You are a friendly Customer Support Assistant.

Customer ID: {customer_id}
Customer Name: {customer_name}

WORKFLOW - Follow these steps for EVERY message:

1. FIRST: Use the classify_intent tool to determine the type of request
2. THEN: Based on the intent:
   - GREETING: Respond warmly yourself (no handoff needed)
   - DATA_QUESTION: Hand off to Data Agent
   - POLICY_QUESTION: Hand off to Policy Agent
   - ACTION_REQUEST: Hand off to Policy Agent
   - OTHER: Ask for clarification

IMPORTANT: Always classify intent first before deciding what to do!

Be helpful, professional, and never expose email addresses.
Greet the customer by name when appropriate.
"""


def create_agents(customer_id: int, customer_name: str):
    """Create agents configured for a specific customer."""

    # Data Agent
    data_agent = Agent(
        name="Data Agent",
        instructions=data_agent_instructions.format(
            customer_id=customer_id,
            customer_name=customer_name
        ),
        tools=[search_customer_orders, get_order_details, execute_custom_sql],
        model=MODEL
    )

    # Policy Agent with Pinecone RAG
    policy_agent = Agent(
        name="Policy Agent",
        instructions=policy_agent_instructions.format(
            customer_id=customer_id,
            customer_name=customer_name
        ),
        tools=[search_policies, check_cancellation_policy],
        model=MODEL
    )

    # Main Support Agent with intent classifier and handoffs
    support_agent = Agent(
        name="Support Agent",
        instructions=support_agent_instructions.format(
            customer_id=customer_id,
            customer_name=customer_name
        ),
        tools=[classify_intent],
        handoffs=[
            handoff(agent=data_agent, tool_description_override="Hand off to Data Agent for order queries and data questions"),
            handoff(agent=policy_agent, tool_description_override="Hand off to Policy Agent for policy questions and action requests")
        ],
        model=MODEL
    )

    return support_agent


# ============= Streamlit App =============

def init_session():
    """Initialize session state."""
    if "customer_id" not in st.session_state:
        st.session_state.customer_id = None
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = None
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_step" not in st.session_state:
        st.session_state.auth_step = "email"  # "email" -> "phone" -> "done"
    if "pending_customer" not in st.session_state:
        st.session_state.pending_customer = None  # Temp storage during 2FA
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []


async def process_message(message: str, customer_id: int, customer_name: str) -> tuple[str, list]:
    """Process a message using the agent system with caching. Fast policy search enabled."""
    import time

    debug_logs = []
    metrics = get_metrics()
    start_time = time.time()

    # Check cache first
    cache = get_cache()
    cached_response = cache.get(customer_id, message)
    if cached_response:
        debug_logs.append("⚡ CACHE HIT - returning cached response")
        metrics.record_cache_hit()
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_latency(latency_ms)
        return cached_response, debug_logs

    debug_logs.append("🔍 Cache miss - processing...")
    metrics.record_cache_miss()

    support_agent = create_agents(customer_id, customer_name)

    # Run without tracing for speed (set ENABLE_TRACING=1 to enable)
    enable_tracing = os.getenv("ENABLE_TRACING", "0") == "1"

    try:
        if enable_tracing:
            with trace("Customer Support Chat"):
                debug_logs.append("Agent processing (with tracing)...")
                result = await Runner.run(support_agent, message)
        else:
            debug_logs.append("Agent processing...")
            result = await Runner.run(support_agent, message)

        debug_logs.append("✓ Agent completed")
        response = result.final_output

        # Cache the response
        cache.set(customer_id, message, response)
        debug_logs.append("💾 Response cached")

    except Exception as e:
        metrics.record_error()
        raise e

    # Record latency
    latency_ms = (time.time() - start_time) * 1000
    metrics.record_latency(latency_ms)
    debug_logs.append(f"⏱️ Latency: {latency_ms:.0f}ms")

    return response, debug_logs


async def process_message_streaming(message: str, customer_id: int, customer_name: str, placeholder):
    """Process a message with streaming output for faster perceived response."""

    debug_logs = []

    # Check cache first
    cache = get_cache()
    cached_response = cache.get(customer_id, message)
    if cached_response:
        debug_logs.append("⚡ CACHE HIT - returning cached response")
        placeholder.markdown(cached_response)
        return cached_response, debug_logs

    debug_logs.append("🔍 Cache miss - processing...")

    support_agent = create_agents(customer_id, customer_name)
    full_response = ""
    current_text = ""

    # Run with streaming
    with trace("Customer Support Chat"):
        debug_logs.append("Starting streaming agent...")
        streamed_result = Runner.run_streamed(support_agent, message)

        async for event in streamed_result.stream_events():
            # Capture raw streaming deltas for live display
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                # Handle OpenAI chat completion chunks
                if hasattr(data, 'choices') and data.choices:
                    for choice in data.choices:
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                            if hasattr(delta, 'content') and delta.content:
                                current_text += delta.content
                                placeholder.markdown(current_text + "▌")

            # Capture final message output
            elif isinstance(event, RunItemStreamEvent):
                if event.name == "message_output_created":
                    item = event.item
                    if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
                        for content in item.raw_item.content:
                            if hasattr(content, 'text'):
                                full_response = content.text

        # Wait for completion
        while not streamed_result.is_complete:
            await asyncio.sleep(0.1)

        # Get final response - prefer message output, fallback to streamed text
        if not full_response and current_text:
            full_response = current_text

        placeholder.markdown(full_response)
        debug_logs.append("✓ Complete")

    # Cache the response
    cache.set(customer_id, message, full_response)
    debug_logs.append("💾 Response cached")

    return full_response, debug_logs


def main():
    st.set_page_config(page_title="Customer Support v2 (Agents)", page_icon="🤖")
    st.title("🤖 Customer Support Chatbot v2")
    st.caption("Powered by OpenAI Agents SDK with Pinecone RAG & Tracing")
    st.markdown("---")

    init_session()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Initial prompt based on auth step
    if not st.session_state.authenticated:
        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                welcome = "Hello! Welcome to Customer Support. Please enter your email address to get started."
                st.write(welcome)
                st.session_state.chat_history.append({"role": "assistant", "content": welcome})

    # Chat input
    if user_input := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            if not st.session_state.authenticated:
                # Two-step authentication: email then phone
                if st.session_state.auth_step == "email":
                    with st.spinner("Checking email..."):
                        customer = validate_customer_email(user_input.strip())
                        if customer:
                            # Store customer info temporarily, ask for phone
                            st.session_state.pending_customer = customer
                            st.session_state.auth_step = "phone"
                            # Mask phone number for hint (show last 4 digits)
                            phone = customer.get("phone", "")
                            masked = f"***-{phone[-4:]}" if len(phone) >= 4 else "***"
                            response = f"Thanks! For security, please enter your phone number (hint: {masked}):"
                            st.session_state.debug_logs.append(f"📧 Email verified, awaiting phone")
                        else:
                            response = "I couldn't find an account with that email. Please try again."
                            st.session_state.debug_logs.append("✗ Email not found")
                    st.write(response)

                elif st.session_state.auth_step == "phone":
                    with st.spinner("Verifying phone..."):
                        pending = st.session_state.pending_customer
                        if pending and validate_customer_phone(pending["customer_id"], user_input.strip()):
                            # Phone verified - complete authentication
                            st.session_state.customer_id = pending["customer_id"]
                            st.session_state.customer_name = pending["full_name"]
                            st.session_state.authenticated = True
                            st.session_state.auth_step = "done"
                            st.session_state.pending_customer = None
                            response = f"✓ Verified! Welcome, {pending['full_name']}! How can I help you today?"
                            st.session_state.debug_logs.append(f"✓ Authenticated: {pending['full_name']}")
                        else:
                            response = "Phone number doesn't match. Please try again."
                            st.session_state.debug_logs.append("✗ Phone verification failed")
                    st.write(response)
            else:
                # Check rate limit first
                rate_limiter = get_rate_limiter()
                allowed, rate_msg = rate_limiter.is_allowed(st.session_state.customer_id)

                if not allowed:
                    response = f"⚠️ {rate_msg}"
                    st.session_state.debug_logs.append("⚠️ Rate limited")
                    st.warning(response)
                else:
                    # Process with agents (fast policy search enabled)
                    st.session_state.debug_logs.append(f"→ Input: {user_input[:50]}...")
                    with st.spinner("Processing..."):
                        try:
                            response, logs = asyncio.run(process_message(
                                user_input,
                                st.session_state.customer_id,
                                st.session_state.customer_name
                            ))
                            st.session_state.debug_logs.extend(logs)
                            st.session_state.debug_logs.append("✓ Response received")
                        except Exception as e:
                            response = f"I apologize, but I encountered an error. Please try again or contact support."
                            st.session_state.debug_logs.append(f"✗ Error: {str(e)}")
                    st.write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Sidebar
    with st.sidebar:
        st.header("Session Info")
        if st.session_state.authenticated:
            st.success(f"✓ {st.session_state.customer_name}")
            st.caption(f"ID: {st.session_state.customer_id}")
        else:
            st.info("Not logged in")

        st.markdown("---")
        st.markdown("### Test Accounts")
        st.code("alice@example.com | 555-0101\nbob@example.com | 555-0102\ncarol@example.com | 555-0103")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.debug_logs = []
                st.session_state.authenticated = False
                st.session_state.auth_step = "email"
                st.session_state.pending_customer = None
                st.session_state.customer_id = None
                st.session_state.customer_name = None
                st.rerun()
        with col2:
            if st.button("🧹 Clear Cache"):
                get_cache().clear()
                st.session_state.debug_logs.append("Cache cleared!")
                st.rerun()

        # Cache & Rate limit stats
        cache_stats = get_cache().stats()
        st.caption(f"📦 Cache: {cache_stats['valid']} entries | Hits: {cache_stats['hits']}")

        if st.session_state.authenticated:
            remaining = get_rate_limiter().get_remaining(st.session_state.customer_id)
            st.caption(f"⏱️ Rate limit: {remaining}/20 requests remaining")

        # Metrics display
        st.markdown("---")
        st.markdown("### 📊 Metrics")
        metrics_summary = get_metrics().get_summary()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens", metrics_summary["total_tokens"])
            st.metric("Requests", metrics_summary["total_requests"])
        with col2:
            st.metric("Avg Latency", f"{metrics_summary['avg_latency_ms']}ms")
            st.metric("Cache Rate", metrics_summary["cache_hit_rate"])

        with st.expander("Detailed Metrics"):
            st.json(metrics_summary)

        if st.button("🔄 Reset Metrics"):
            get_metrics().reset()
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔍 Traces")
        st.markdown("[View in OpenAI Dashboard →](https://platform.openai.com/traces)")

        st.markdown("---")
        st.markdown("### Debug Logs")
        if st.session_state.debug_logs:
            for log in st.session_state.debug_logs[-15:]:
                st.code(log, language=None)
        else:
            st.caption("No logs yet")

        st.markdown("---")
        st.markdown("### Sample Questions")
        st.markdown("""
**Data:**
- What are my orders?
- What did I order last month?

**Policy (uses Pinecone):**
- What's your return policy?
- Can I cancel an order?

**Action:**
- Cancel order 101
        """)


if __name__ == "__main__":
    main()
