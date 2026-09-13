"""
The HTTP layer: routes, middleware, and the chat and metrics endpoints.

Everything customer-facing goes through `/api/chat/message`. There is no
endpoint that returns an account's orders directly -- `/api/lookup-orders` and
`/api/order-details` used to exist and were removed, because they predated the
verification gate and served any account's data to anyone who asked. A chat
turn is the only path that can prove who it is talking to first.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

# Before anything reads os.environ. production_constants and the model
# backends both resolve their settings at import time, so a .env loaded after
# them would be ignored.
from env_file import load as _load_env_file

_ENV_FROM_FILE = _load_env_file()

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import security
import telemetry
from mcp_server import CustomerChatbotTools
from database import (
    get_all_customers, get_all_orders, get_all_products,
    get_all_policies, get_all_faqs, get_all_support_tickets
)
from conversation_manager import (
    get_conversation_manager, ResponseGenerator
)
from production_constants import (
    API_VERSION, ENVIRONMENT, DEBUG,
    API_HOST, API_PORT, LOG_LEVEL,
)


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchKnowledgeRequest(BaseModel):
    """Search policies and FAQs."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class ChatMessageRequest(BaseModel):
    """Send a message in chat."""
    session_id: str = Field(..., min_length=1, max_length=50, description="Chat session ID")
    message: str = Field(..., min_length=1, max_length=2000, description="Customer message")


class ChatSessionResponse(BaseModel):
    """Chat session response."""
    session_id: str = Field(..., description="Unique session ID")
    created_at: str = Field(..., description="Session creation timestamp")


class ChatMessageResponse(BaseModel):
    """Chat message response."""
    session_id: str = Field(..., description="Session ID")
    response: str = Field(..., description="Bot response")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: float = Field(default=0.0, description="Confidence score")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested follow-up actions")
    actions_taken: List[str] = Field(default_factory=list, description="Actions performed")
    # Subject and source are separate axes: "how much is shipping" is a
    # shipping question answered from the knowledge base. Folding the second
    # into `intent` made shipping, payment and warranty indistinguishable.
    answer_source: Optional[str] = Field(
        None, description="knowledge_base | account_data | none")
    timestamp: str = Field(..., description="Response timestamp")
    # Which specialist answered: triage, data or policy. Same names in both
    # engines, so a mid-conversation fallback is comparable.
    agent: Optional[str] = Field(None, description="triage | data | policy")
    # Which engine that specialist belonged to. The UI badge keys off this, so
    # it has to stay separate from `agent`.
    engine: Optional[str] = Field(None, description="llm | unavailable")
    # Why a turn could not be answered. Operator-facing: the customer sees the
    # apology in `response`, but a turn that fails silently is a turn nobody
    # can debug, and "every reply is the apology" told us nothing last time.
    error: Optional[str] = Field(None, description="Why the turn failed, if it did")
    model: Optional[str] = Field(None, description="Model that produced this reply")
    provider: Optional[str] = Field(None, description="anthropic | openai-compatible")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Timestamp")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")


class MetricsResponse(BaseModel):
    """System metrics response."""
    total_requests: int = Field(..., description="Total requests processed")
    avg_response_time_ms: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate percentage")
    uptime_seconds: int = Field(..., description="Uptime in seconds")
    timestamp: str = Field(..., description="Metrics timestamp")


# ============================================================================
# Metrics Tracking
# ============================================================================

class MetricsCollector:
    """What the app is actually doing, in numbers you can act on.

    The old version kept one mean over every request, which mixed a 20-second
    model turn in with a 1ms /health poll and reported 111ms while customers
    waited half a minute. A mean over a mixed population is not a slow number,
    it is a meaningless one -- so chat turns are counted separately, and
    percentiles are reported instead of an average.
    """

    KEEP = 1000        # recent samples per series

    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.rate_limited = 0
        self.response_times: List[float] = []
        # Chat turns are the number that matters: one customer message in, one
        # answer out, however many model calls that took.
        self.turn_times: List[float] = []
        self.turns_by_agent: dict = {}
        self.turns_unavailable = 0
        self.start_time = time.time()

    def record_request(self, response_time_ms: float, is_error: bool = False):
        self.total_requests += 1
        self.response_times.append(response_time_ms)
        if is_error:
            self.total_errors += 1
        if len(self.response_times) > self.KEEP:
            self.response_times = self.response_times[-self.KEEP:]

    def record_turn(self, elapsed_ms: float, agent: Optional[str],
                    engine: Optional[str]) -> None:
        self.turn_times.append(elapsed_ms)
        if len(self.turn_times) > self.KEEP:
            self.turn_times = self.turn_times[-self.KEEP:]
        key = agent or "unknown"
        self.turns_by_agent[key] = self.turns_by_agent.get(key, 0) + 1
        if engine == "unavailable":
            self.turns_unavailable += 1

    def record_rate_limited(self) -> None:
        self.rate_limited += 1

    @staticmethod
    def percentile(samples: List[float], fraction: float) -> float:
        """Nearest-rank percentile. No numpy for four numbers."""
        if not samples:
            return 0.0
        ordered = sorted(samples)
        index = min(len(ordered) - 1,
                    max(0, int(round(fraction * len(ordered) + 0.5)) - 1))
        return round(ordered[index], 2)

    def latency(self, samples: List[float]) -> dict:
        return {
            "count": len(samples),
            "p50_ms": self.percentile(samples, 0.50),
            "p95_ms": self.percentile(samples, 0.95),
            "p99_ms": self.percentile(samples, 0.99),
            "max_ms": round(max(samples), 2) if samples else 0.0,
        }

    def get_metrics(self) -> MetricsResponse:
        """The narrow shape the old endpoint promised, kept for compatibility."""
        avg = (sum(self.response_times) / len(self.response_times)
               if self.response_times else 0)
        error_rate = ((self.total_errors / self.total_requests * 100)
                      if self.total_requests else 0)
        return MetricsResponse(
            total_requests=self.total_requests,
            avg_response_time_ms=round(avg, 2),
            error_rate=round(error_rate, 2),
            uptime_seconds=int(time.time() - self.start_time),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def detail(self) -> dict:
        """Everything, including what the agents and the cache have been doing."""
        from response_cache import get_cache

        report = {
            "uptime_seconds": int(time.time() - self.start_time),
            "requests": {
                "total": self.total_requests,
                "errors": self.total_errors,
                "error_rate_pct": round(
                    (self.total_errors / self.total_requests * 100)
                    if self.total_requests else 0, 2),
                "rate_limited": self.rate_limited,
                "latency": self.latency(self.response_times),
            },
            "chat_turns": {
                "total": len(self.turn_times),
                "unavailable": self.turns_unavailable,
                "by_agent": dict(self.turns_by_agent),
                "latency": self.latency(self.turn_times),
            },
            "cache": get_cache().stats(),
            "rate_limit": {
                "limit_per_minute": limiter.limit,
                "tracked_callers": limiter.tracked_callers,
            },
        }
        try:
            generator = get_response_generator()
            agent = getattr(generator, "llm_agent", None)
            if agent is not None and agent.available:
                usage = agent.usage()
                report["model"] = {
                    "model": agent.model,
                    "provider": agent.provider,
                    "turns": usage["turns"],
                    "triage_calls": usage["triage_calls"],
                    "triage_gave_up": usage["triage_gave_up"],
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "routed_to": usage["routed_to"],
                }
        except Exception as exc:                    # pragma: no cover
            report["model"] = {"error": str(exc)}
        return report


metrics = MetricsCollector()

# 20 requests a minute per caller. A person talking to a model that answers in
# tens of seconds cannot approach this; a runaway client or someone walking the
# order-id space reaches it immediately, which is the point.
limiter = security.RateLimiter(
    limit=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20")),
    window_seconds=60.0,
)


# ============================================================================
# Application Initialization
# ============================================================================

# Initialize MCP tools
mcp_tools = CustomerChatbotTools()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info(f"Starting CustomerChatbot {API_VERSION} ({ENVIRONMENT})")
    if _ENV_FROM_FILE:
        # Say so explicitly. A setting that came from a file you forgot you
        # wrote is harder to debug than one you typed.
        logger.info("Loaded from .env: %s", ", ".join(sorted(_ENV_FROM_FILE)))

    # Build the engine and index the knowledge base at boot rather than on the
    # first customer message. Without this the first person to say anything
    # waits for the retriever to index and the model client to construct.
    try:
        generator = get_response_generator()
        logger.info("Response engine: %s", generator.engine_status)
    except Exception as exc:
        logger.warning("Could not initialise the response engine: %s", exc)

    try:
        from retrieval import get_retriever
        retriever = get_retriever()
        logger.info("Knowledge base: %d passages (%s)",
                    len(retriever.chunks), retriever.backend)
    except Exception as exc:
        logger.warning("Could not build the knowledge index: %s", exc)

    yield
    logger.info("Shutting down CustomerChatbot")


app = FastAPI(
    title="CustomerChatbot MCP Server",
    description="Production-grade customer support system with MCP protocol integration",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Security Headers Middleware
# ============================================================================

# Endpoints cheap enough that limiting them would only get in the way of
# monitoring, and that expose nothing.
RATE_LIMIT_EXEMPT = {"/health", "/metrics", "/metrics/detail", "/status", "/ui"}


def _caller(request: Request) -> str:
    """Who to count this against.

    Session first: several people behind one office NAT should not share an
    allowance, and a single session looping is exactly what we want to catch.
    """
    session = request.headers.get("x-session-id")
    if session:
        return f"session:{security.clean_text(session, 64)}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Cap how fast one caller can drive the app."""
    if request.url.path in RATE_LIMIT_EXEMPT:
        return await call_next(request)

    allowed, remaining, retry_after = limiter.check(_caller(request))
    if not allowed:
        metrics.record_rate_limited()
        logger.warning("rate limited %s on %s", _caller(request), request.url.path)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Too many requests. Please slow down.",
                "retry_after_seconds": round(retry_after, 1),
                "limit_per_minute": limiter.limit,
            },
            headers={"Retry-After": str(int(retry_after) + 1),
                     "X-RateLimit-Limit": str(limiter.limit),
                     "X-RateLimit-Remaining": "0"},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limiter.limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        response_time = (time.time() - start_time) * 1000
        metrics.record_request(response_time, is_error=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # The two HTML pages are single self-contained files: their CSS and JS are
    # inline, which a bare "default-src 'self'" policy blocks outright. Relax
    # the policy for those first-party assets and keep it strict everywhere
    # else (notably the JSON API, which should never execute anything).
    #
    # /metrics/dashboard was missing from this list, so the browser refused
    # every rule and every line of its script: the page rendered as unstyled
    # HTML with no data in it. Nothing failed loudly -- the server returned
    # 200, the file was intact, and only the console said why. It took a
    # screenshot to notice.
    if request.url.path in ("/ui", "/metrics/dashboard"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "base-uri 'none'; "
            "form-action 'none'; "
            "frame-ancestors 'none'"
        )
    else:
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    # Record metrics
    response_time = (time.time() - start_time) * 1000
    is_error = response.status_code >= 400
    metrics.record_request(response_time, is_error=is_error)

    logger.debug(
        f"{request.method} {request.url.path} - {response.status_code} - {response_time:.2f}ms"
    )

    return response


# ============================================================================
# Health & Status Endpoints
# ============================================================================

_response_generator = None


def get_response_generator() -> ResponseGenerator:
    """Build the response generator once per process."""
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator(mcp_tools)
        logger.info("Response engine: %s", _response_generator.engine_status)
    return _response_generator


def _engine_info() -> dict:
    """Which reasoning engine and retrieval backend are actually live."""
    generator = get_response_generator()
    try:
        from retrieval import get_retriever
        retriever = get_retriever()
        retrieval_backend = retriever.backend
        indexed = len(retriever.chunks)
    except Exception as exc:
        retrieval_backend = f"unavailable ({exc})"
        indexed = 0
    info = {
        "reasoning": generator.engine,
        "reasoning_detail": generator.engine_status,
        "retrieval": retrieval_backend,
        "indexed_passages": indexed,
    }

    agent = getattr(generator, "llm_agent", None)
    if agent is not None and getattr(agent, "available", False):
        info["provider"] = agent.provider
        info["model"] = agent.model
        info["max_tool_steps"] = agent.max_steps
        # Cumulative since process start -- the input to any cost calculation.
        info["token_usage"] = agent.usage()

    return info


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=API_VERSION,
        environment=ENVIRONMENT
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics() -> MetricsResponse:
    """Get system metrics."""
    return metrics.get_metrics()


@app.get("/metrics/detail", tags=["Metrics"])
async def get_metrics_detail():
    """Latency percentiles, chat turns, token usage, cache hit rate."""
    return metrics.detail()


@app.get("/metrics/history", tags=["Metrics"])
async def get_metrics_history(hours: int = 24, bucket: str = "hour"):
    """Model health over a window, from the persistent turn log.

    This is the endpoint that survives a restart. `/metrics/detail` reports
    since-boot counters; this reports the last N hours whether or not the
    process was up for them.
    """
    store = telemetry.get_store()
    if store is None or not store.available:
        return {
            "enabled": False,
            "reason": ("telemetry is switched off (CHATBOT_TELEMETRY) or the "
                       "database could not be opened"),
        }
    hours = max(1, min(int(hours), 24 * 365))
    if bucket not in ("hour", "day"):
        bucket = "hour"
    report = store.health(hours=hours, buckets=bucket)
    report["enabled"] = True
    report["database"] = store.path
    report["total_turns_all_time"] = store.total_turns()
    report["write_failures"] = store.write_failures
    return report


@app.get("/metrics/turns", tags=["Metrics"])
async def get_recent_turns(limit: int = 50):
    """The last N turns as recorded, newest first.

    The per-run view: what was asked, which agent took it, which tools ran,
    which guards fired, what it cost.
    """
    store = telemetry.get_store()
    if store is None or not store.available:
        return {"enabled": False, "turns": []}
    return {
        "enabled": True,
        "turns": store.recent(limit=max(1, min(int(limit), 500))),
    }


@app.get("/metrics/dashboard", tags=["Metrics"], response_class=HTMLResponse)
async def metrics_dashboard():
    """The same numbers, readable without a JSON formatter."""
    page = os.path.join(os.path.dirname(__file__), "static", "metrics_dashboard.html")
    try:
        with open(page, encoding="utf-8") as handle:
            return HTMLResponse(content=handle.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="dashboard page not found")


def _observability_info() -> dict:
    """Whether turns are being recorded, and where.

    Reported rather than assumed: a telemetry layer that silently stopped
    writing looks exactly like a quiet week.
    """
    import tracing

    store = telemetry.get_store()
    if store is None:
        persistence = {"enabled": False,
                       "reason": "CHATBOT_TELEMETRY is off"}
    elif not store.available:
        persistence = {"enabled": False,
                       "reason": f"could not open {store.path}"}
    else:
        persistence = {
            "enabled": True,
            "database": store.path,
            "turns_recorded": store.total_turns(),
            "write_failures": store.write_failures,
        }
    return {
        "turn_log": persistence,
        "langsmith": {"enabled": tracing.enabled(), "status": tracing.status()},
    }


@app.get("/status", tags=["Health"])
async def get_status():
    """Get detailed status information."""
    return {
        "status": "running",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(time.time() - metrics.start_time),
        "engine": _engine_info(),
        "observability": _observability_info(),
        "data": {
            "customers": len(get_all_customers()),
            "orders": len(get_all_orders()),
            "products": len(get_all_products()),
            "policies": len(get_all_policies()),
            "faqs": len(get_all_faqs()),
            "support_tickets": len(get_all_support_tickets()),
        }
    }


@app.get("/ui", tags=["Info"], include_in_schema=False)
async def chat_ui():
    """Serve the chat interface from the same origin as the API.

    Opening chat_interface.html straight off disk works too, but serving it
    here means the page and the API always agree on host and port.
    """
    from fastapi.responses import FileResponse

    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "static", "chat_interface.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404,
                            detail="static/chat_interface.html not found")
    return FileResponse(ui_path, media_type="text/html")


@app.get("/", tags=["Info"])
async def root():
    """API root with information."""
    return {
        "service": "CustomerChatbot MCP Server",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "metrics": "/metrics",
            "search_knowledge": "/api/search-knowledge",
            "chat": "/api/chat/message",
            "lookup_orders": "/api/lookup-orders",
            "order_details": "/api/order-details",
            "chat_session": "/api/chat/session",
            "chat_message": "/api/chat/message",
            "chat_history": "/api/chat/history/{session_id}",
            "chat_session_details": "/api/chat/session/{session_id}"
        }
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/search-knowledge", tags=["Tools"])
async def search_knowledge(request: SearchKnowledgeRequest):
    """Search policies and FAQs at passage level.

    The same retrieval the policy agent uses: BM25 with concept expansion,
    fused with dense embeddings when they are installed.
    """
    try:
        result = await mcp_tools.search_knowledge(request.query, request.top_k)
        return {
            "success": True,
            "query": request.query,
            "results": result.data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Policy search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Policy search failed")


# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/api/chat/session", response_model=ChatSessionResponse, tags=["Chat"])
async def create_chat_session():
    """Create a new chat session.

    Returns a unique session ID for managing multi-turn conversations.
    """
    try:
        manager = get_conversation_manager()
        session_id = manager.create_session()
        conversation = manager.get_session(session_id)

        return ChatSessionResponse(
            session_id=session_id,
            created_at=conversation.created_at
        )
    except Exception as e:
        logger.error(f"Chat session creation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create chat session")


@app.post("/api/chat/message", response_model=ChatMessageResponse, tags=["Chat"])
async def send_chat_message(request: ChatMessageRequest):
    """Send a message in an existing chat session.

    Processes customer message, maintains conversation context, and generates
    contextual bot response using conversation history and MCP tools.
    """
    try:
        manager = get_conversation_manager()

        # Get or validate session
        conversation = manager.get_session(request.session_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found"
            )

        # Strip control characters before anything stores or renders this.
        # Not SQL escaping -- there is no SQL here -- but the same instinct:
        # do not let the field decide what it contains.
        message = security.clean_text(request.message, security.MAX_MESSAGE_LENGTH)
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Add customer message to history
        manager.add_message(
            request.session_id,
            role="customer",
            content=message
        )

        # Get conversation context and history
        context = manager.get_context(request.session_id)
        message_history = manager.get_message_history(request.session_id)

        # Shared generator: building it per-request would re-create the LLM
        # client and re-index the knowledge base on every message.
        response_generator = get_response_generator()
        turn_started = time.time()
        response_data = await response_generator.generate_response(
            message,
            context,
            message_history
        )
        # Timed separately from the HTTP request: a chat turn and a /health
        # poll averaged together produce a number that describes neither.
        turn_ms = (time.time() - turn_started) * 1000
        metrics.record_turn(
            turn_ms,
            response_data.get("agent"),
            response_data.get("engine"),
        )

        # The same turn, to disk. In-memory counters answer "how is it right
        # now"; this is what makes "how has it been this week" answerable at
        # all. A store that is off or broken must not cost the customer their
        # reply, so record() never raises.
        store = telemetry.get_store()
        if store is not None:
            store.record(telemetry.TurnRecord.from_response(
                response_data,
                session_id=request.session_id,
                turn_index=len(message_history),
                latency_ms=turn_ms,
                message=message,
                verified=bool(getattr(context, "is_verified", False)),
            ))

        # Add bot response to history
        manager.add_message(
            request.session_id,
            role="bot",
            content=response_data["response"],
            intent=response_data.get("intent"),
            metadata={
                "confidence": response_data.get("confidence", 0.0),
                "suggested_actions": response_data.get("suggested_actions", []),
                "actions_taken": response_data.get("actions_taken", [])
            }
        )

        return ChatMessageResponse(
            session_id=request.session_id,
            response=response_data["response"],
            intent=response_data.get("intent"),
            confidence=response_data.get("confidence", 0.0),
            suggested_actions=response_data.get("suggested_actions", []),
            actions_taken=response_data.get("actions_taken", []),
            answer_source=response_data.get("answer_source"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=response_data.get("agent"),
            engine=response_data.get("engine", "unavailable"),
            error=response_data.get("error"),
            model=response_data.get("model"),
            provider=response_data.get("provider"),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat message processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process chat message")


@app.get("/api/chat/history/{session_id}", tags=["Chat"])
async def get_chat_history(session_id: str, limit: Optional[int] = None):
    """Get conversation history for a session.

    Returns all messages in a chat session, optionally limited to most recent N messages.
    """
    try:
        manager = get_conversation_manager()

        # Validate session
        conversation = manager.get_session(session_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        # Get message history
        history = manager.get_message_history(session_id, limit=limit)

        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@app.get("/api/chat/session/{session_id}", tags=["Chat"])
async def get_chat_session(session_id: str):
    """Get chat session details.

    Returns session metadata and current conversation context.
    """
    try:
        manager = get_conversation_manager()

        # Get session
        conversation = manager.get_session(session_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        # Get context
        context = manager.get_context(session_id)

        return {
            "session_id": session_id,
            "created_at": conversation.created_at,
            "last_message_time": conversation.last_message_time,
            "message_count": len(conversation.messages),
            "context": {
                "customer_id": context.customer_id,
                "current_order_id": context.current_order_id,
                "known_order_ids": list(context.known_order_ids),
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat session")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail or "Request failed",
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Documentation: http://{API_HOST}:{API_PORT}/docs")

    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
