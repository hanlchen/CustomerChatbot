"""
The tools, and the only path from an agent to the data.

Every fact that reaches a customer comes back as a tool result from here.
Agents never touch `database.py` directly, which is what stops a generative
model stating an order number nothing looked up.

Exposed over the Model Context Protocol as well as in-process, so the same
tools can be driven by an MCP client. Account tools refuse to run until the
conversation has confirmed the phone number; that check lives in the agent's
`_run_tool`, in front of the call, not in a prompt.
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import sys

from database import (
    get_customer_by_id as db_get_customer_by_id,
    get_orders_by_customer as db_get_orders_by_customer,
    get_order_details as db_get_order_details,
    get_system_metrics as db_get_metrics,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions and Data Classes
# ============================================================================

class TransportType(Enum):
    """Supported transport types for MCP communication."""
    STDIO = "stdio"
    HTTP = "http"


@dataclass
class ToolSchema:
    """Schema definition for an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ToolResult:
    """Result returned from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


# ============================================================================
# Tool Implementations
# ============================================================================

class CustomerChatbotTools:
    """Implements all tools exposed by the MCP server."""

    @staticmethod
    async def search_knowledge(query: str, top_k: int = 4) -> ToolResult:
        """
        Search policies AND FAQs at passage level for a natural-language question.

        Returns the specific section that answers the question rather than
        the top of a whole document, and matches paraphrases ("can I send
        this back") as well as exact terms.

        Args:
            query: The customer's question, in their own words
            top_k: Maximum passages to return (default 4)

        Returns:
            ToolResult with ranked passages, each carrying its source title
        """
        try:
            start_time = datetime.now()
            from retrieval import search_knowledge as _search

            passages = _search(query, top_k=top_k)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"search_knowledge: {len(passages)} passages for '{query}'")
            return ToolResult(
                success=True,
                data={"query": query, "passages": passages},
                execution_time_ms=execution_time,
            )
        except Exception as e:
            logger.error(f"search_knowledge error: {str(e)}", exc_info=True)
            return ToolResult(
                success=False, data=None,
                error=f"Failed to search knowledge base: {str(e)}",
            )

    @staticmethod
    async def begin_verification(identifier: str) -> ToolResult:
        """
        Start identity verification for a customer ID or email address.

        Returns a challenge token and a *masked* hint at the phone number on
        file. Deliberately does not return the customer ID: an unverified
        conversation should never hold one, because anything the model holds
        it can be talked into repeating.

        Args:
            identifier: A customer ID (CUST-10000) or the account email

        Returns:
            ToolResult with {found, challenge_token, phone_hint}
        """
        try:
            start_time = datetime.now()
            import security
            from database import get_customer_by_email

            value = security.clean_text(identifier, 200)
            customer = None
            if security.looks_like_customer_id(value):
                customer = db_get_customer_by_id(value)
            elif security.looks_like_email(value):
                customer = get_customer_by_email(value)
            else:
                return ToolResult(
                    success=False, data=None,
                    error=("That is not a customer ID or an email address. A "
                           "customer ID looks like CUST-10000."),
                )

            if not customer:
                # Same shape whether or not the account exists, so this cannot
                # be used to find out which emails are registered.
                return ToolResult(
                    success=True,
                    data={"found": False,
                          "message": ("No account matches that. Check the "
                                      "spelling, or try the other one -- "
                                      "customer ID or email.")},
                )

            phone = customer.get("phone") or ""
            if not phone:
                return ToolResult(
                    success=False, data=None,
                    error=("There is no phone number on this account, so it "
                           "cannot be verified here. Send them to "
                           "support@example.com or 1-800-555-0100."),
                )

            challenge = security.store().open(customer["customer_id"], phone)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info("begin_verification: challenge opened for %s",
                        customer["customer_id"])
            return ToolResult(
                success=True,
                data={
                    "found": True,
                    "challenge_token": challenge.token,
                    "phone_hint": security.mask_phone(phone),
                    "attempts_allowed": security.MAX_PHONE_ATTEMPTS,
                },
                execution_time_ms=execution_time,
            )
        except Exception as e:
            logger.error(f"begin_verification error: {str(e)}", exc_info=True)
            return ToolResult(success=False, data=None,
                              error=f"Failed to start verification: {str(e)}")

    @staticmethod
    async def confirm_phone(challenge_token: str, phone: str) -> ToolResult:
        """
        Check a phone number against a live verification challenge.

        On success the session is cleared to see that customer's data and
        nobody else's. Attempts are counted and the challenge is destroyed
        once they run out.

        Args:
            challenge_token: From begin_verification
            phone: The number the customer gave, in any format

        Returns:
            ToolResult with {verified, customer_id?, attempts_left}
        """
        try:
            import security

            challenge = security.store().get(challenge_token)
            if challenge is None:
                return ToolResult(
                    success=True,
                    data={"verified": False, "expired": True,
                          "message": ("That verification has expired. Start "
                                      "again with the customer ID or email.")},
                )

            # A customer mid-verification says all sorts of things, and the
            # agent hands whatever they said to this tool. "Return an item" is
            # not a wrong phone number, it is not a phone number -- counting it
            # as an attempt burns the allowance of somebody who has not tried
            # to guess anything yet.
            if len(security.phone_digits(phone)) < security.SIGNIFICANT_DIGITS:
                return ToolResult(
                    success=True,
                    data={"verified": False,
                          "not_a_phone_number": True,
                          "attempts_left": challenge.attempts_left,
                          "message": ("That does not look like a phone number, "
                                      "so it was not counted as an attempt. Ask "
                                      "them for the number on the account.")},
                )

            challenge.attempts += 1
            if security.phone_matches(phone, challenge.phone_on_file):
                customer_id = challenge.customer_id
                security.store().close(challenge.token)
                logger.info("confirm_phone: verified %s", customer_id)
                return ToolResult(
                    success=True,
                    data={"verified": True, "customer_id": customer_id},
                )

            if challenge.attempts_left <= 0:
                security.store().close(challenge.token)
                logger.warning("confirm_phone: attempts exhausted for %s",
                               challenge.customer_id)
                return ToolResult(
                    success=True,
                    data={"verified": False, "attempts_left": 0,
                          "locked_out": True,
                          "message": ("Too many attempts. For security this "
                                      "has to continue with a person: "
                                      "support@example.com or "
                                      "1-800-555-0100.")},
                )

            logger.info("confirm_phone: mismatch, %d attempts left",
                        challenge.attempts_left)
            return ToolResult(
                success=True,
                data={"verified": False,
                      "attempts_left": challenge.attempts_left,
                      "message": "That does not match the number on file."},
            )
        except Exception as e:
            logger.error(f"confirm_phone error: {str(e)}", exc_info=True)
            return ToolResult(success=False, data=None,
                              error=f"Failed to check the number: {str(e)}")

    @staticmethod
    async def check_return_eligibility(order_id: str,
                                       owner_customer_id: Optional[str] = None) -> ToolResult:
        """
        Decide whether a specific order can be returned, and why.

        The verdict is computed from the order record so it never has to be
        inferred from dates in prose.

        Args:
            order_id: The order to evaluate (e.g. "ORD-100000")

        Returns:
            ToolResult with the verdict, the reason, and the returnable items
        """
        try:
            start_time = datetime.now()
            from database import get_order_details as _details

            order = _details(order_id)
            if not order:
                return ToolResult(
                    success=False, data=None,
                    error=f"Order {order_id} not found",
                )

            # Data isolation. A verified session may read its own orders and
            # nothing else -- otherwise anyone who got through verification
            # once could walk the whole order-id space. Answering "not found"
            # rather than "not yours" means the response cannot be used to
            # discover which order numbers exist.
            if owner_customer_id and order.get("customer_id") != owner_customer_id:
                logger.warning(
                    "blocked cross-customer read: %s tried to read %s",
                    owner_customer_id, order_id)
                return ToolResult(
                    success=False, data=None,
                    error=f"Order {order_id} not found on this account",
                )

            status = (order.get("status") or "unknown").lower()
            deadline = str(order.get("return_deadline") or "")[:10]

            if status in ("pending", "processing"):
                verdict, reason = "not_yet", (
                    "The order has not shipped, so there is nothing to return. "
                    "It can still be cancelled. The 30-day return window opens "
                    "on delivery."
                )
            elif status == "shipped":
                verdict, reason = "not_yet", (
                    "The order is still in transit. The 30-day return window "
                    "starts the day it is delivered."
                )
            elif status == "cancelled":
                verdict, reason = "not_applicable", "This order was cancelled."
            elif status == "returned":
                verdict, reason = "not_applicable", "This order was already returned."
            elif order.get("return_eligible"):
                verdict, reason = "yes", (
                    f"Within the return window{f' until {deadline}' if deadline else ''}."
                )
            else:
                verdict, reason = "no", (
                    f"The return window closed{f' on {deadline}' if deadline else ''}."
                )

            items = order.get("items", [])
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"check_return_eligibility: {order_id} -> {verdict}")
            return ToolResult(
                success=True,
                data={
                    "order_id": order.get("order_id"),
                    "status": status,
                    "verdict": verdict,
                    "reason": reason,
                    "return_deadline": deadline or None,
                    "estimated_delivery": str(order.get("estimated_delivery") or "")[:10] or None,
                    "returnable_items": [
                        i.get("product_name") for i in items if i.get("return_eligible")
                    ],
                    "non_returnable_items": [
                        i.get("product_name") for i in items if not i.get("return_eligible")
                    ],
                },
                execution_time_ms=execution_time,
            )
        except Exception as e:
            logger.error(f"check_return_eligibility error: {str(e)}", exc_info=True)
            return ToolResult(
                success=False, data=None,
                error=f"Failed to check return eligibility: {str(e)}",
            )

    @staticmethod
    async def list_policy_topics() -> ToolResult:
        """
        List the policy topics a customer can ask about.

        Used when someone asks to "see policies" generally rather than for a
        specific one -- a keyword search has nothing to match on there.

        Returns:
            ToolResult with categories and the policy titles under each

        Example:
            >>> result = await list_policy_topics()
            >>> result.data["categories"]
            [{"category": "Returns", "titles": ["30-Day Return Policy", ...]}, ...]
        """
        try:
            start_time = datetime.now()
            from database import _db

            grouped: Dict[str, List[str]] = {}
            for policy in _db.policies:
                grouped.setdefault(policy.get("category", "General"), []).append(
                    policy.get("title", "")
                )

            categories = [
                {"category": name, "titles": titles}
                for name, titles in sorted(grouped.items())
            ]
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"list_policy_topics: {len(categories)} categories")

            return ToolResult(
                success=True,
                data={"categories": categories, "total_policies": len(_db.policies)},
                execution_time_ms=execution_time,
            )
        except Exception as e:
            logger.error(f"list_policy_topics error: {str(e)}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to list policy topics: {str(e)}",
            )

    @staticmethod
    async def lookup_customer_orders(customer_id: str) -> ToolResult:
        """
        Look up all orders for a specific customer.

        Args:
            customer_id: Unique customer identifier

        Returns:
            ToolResult with list of orders for the customer

        Example:
            >>> result = await lookup_customer_orders("CUST-12345")
            >>> result.data
            {
                "customer_id": "CUST-12345",
                "customer_name": "John Smith",
                "total_orders": 5,
                "orders": [
                    {
                        "order_id": "ORD-001",
                        "order_date": "2025-06-15T10:30:00",
                        "status": "delivered",
                        "total_amount": 149.99,
                        "item_count": 3
                    },
                    ...
                ]
            }
        """
        try:
            start_time = datetime.now()

            # Get customer info
            customer = db_get_customer_by_id(customer_id)
            if not customer:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Customer {customer_id} not found"
                )

            # Get customer orders, newest first. "My most recent order" is the
            # single most common way customers refer to one, so the order the
            # list arrives in is part of the answer, not a detail.
            orders = sorted(
                db_get_orders_by_customer(customer_id),
                key=lambda o: str(o.get("order_date") or ""),
                reverse=True,
            )

            order_list = [
                {
                    "order_id": order.get("order_id"),
                    "order_date": order.get("order_date"),
                    "status": order.get("status"),
                    "total_amount": order.get("total_amount"),
                    "item_count": len(order.get("items", [])),
                    # Product names travel with the summary so a customer who
                    # can't recall an order number can still identify it.
                    "item_names": [
                        item.get("product_name")
                        for item in order.get("items", [])
                        if item.get("product_name")
                    ],
                    "shipping_status": order.get("shipping_status"),
                    "tracking_number": order.get("tracking_number"),
                }
                for order in orders
            ]

            result_data = {
                "customer_id": customer_id,
                "customer_name": f"{customer.get('first_name')} {customer.get('last_name')}",
                "email": customer.get("email"),
                "total_orders": len(order_list),
                "orders": order_list,
                "account_created": customer.get("created_at"),
            }

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"lookup_customer_orders: Found {len(order_list)} orders for {customer_id}")

            return ToolResult(
                success=True,
                data=result_data,
                execution_time_ms=execution_time
            )
        except Exception as e:
            logger.error(f"lookup_customer_orders error: {str(e)}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to look up customer orders: {str(e)}"
            )

    @staticmethod
    async def get_order_details(order_id: str,
                                owner_customer_id: Optional[str] = None) -> ToolResult:
        """
        Get detailed information about a specific order.

        Args:
            order_id: Unique order identifier

        Returns:
            ToolResult with the order, its items and its shipping state

        Example:
            >>> result = await get_order_details("ORD-001")
            >>> result.data
            {
                "order_id": "ORD-001",
                "customer_id": "CUST-12345",
                "order_date": "2025-06-15T10:30:00",
                "status": "delivered",
                "items": [
                    {
                        "product_id": "PROD-100",
                        "product_name": "Wireless Headphones",
                        "quantity": 1,
                        "unit_price": 79.99,
                        "total_price": 79.99
                    },
                    ...
                ],
                "subtotal": 159.98,
                "tax": 12.80,
                "shipping_cost": 9.99,
                "total_amount": 182.77,
                "shipping_address": {...},
                "tracking_number": "1Z999AA10123456784",
                "estimated_delivery": "2025-06-20T18:00:00",
                "return_eligible": true,
                "return_deadline": "2025-07-15"
            }
        """
        try:
            start_time = datetime.now()

            order = db_get_order_details(order_id)
            if not order:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Order {order_id} not found"
                )

            # Data isolation. A verified session may read its own orders and
            # nothing else -- otherwise anyone who verified once could walk
            # the whole order-id space. "Not found" rather than "not yours",
            # so the answer cannot be used to discover which orders exist.
            if owner_customer_id and order.get("customer_id") != owner_customer_id:
                logger.warning("blocked cross-customer read: %s tried to read %s",
                               owner_customer_id, order_id)
                return ToolResult(
                    success=False, data=None,
                    error=f"Order {order_id} not found on this account",
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"get_order_details: Retrieved details for {order_id}")

            return ToolResult(
                success=True,
                data=order,
                execution_time_ms=execution_time
            )
        except Exception as e:
            logger.error(f"get_order_details error: {str(e)}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to get order details: {str(e)}"
            )

    @staticmethod
    async def get_system_metrics() -> ToolResult:
        """
        Get current system metrics and health status.

        Args:
            None

        Returns:
            ToolResult with system metrics

        Example:
            >>> result = await get_system_metrics()
            >>> result.data
            {
                "timestamp": "2025-08-17T14:30:45Z",
                "system_status": "healthy",
                "database_connection": "connected",
                "active_sessions": 42,
                "api_response_time_ms": 125,
                "cache_hit_rate": 0.87,
                "pending_orders": 23,
                "pending_tickets": 8,
                "customer_satisfaction_score": 4.6,
                "uptime_hours": 720,
                "error_rate_percent": 0.02
            }
        """
        try:
            start_time = datetime.now()

            metrics = db_get_metrics()

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"get_system_metrics: Retrieved system metrics")

            return ToolResult(
                success=True,
                data=metrics,
                execution_time_ms=execution_time
            )
        except Exception as e:
            logger.error(f"get_system_metrics error: {str(e)}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to get system metrics: {str(e)}"
            )


# ============================================================================
# Tool Schema Definitions
# ============================================================================

# What this server exposes over MCP, and why it is not everything.
#
# The account tools -- orders, order details, return eligibility -- are gated
# on a *verified session*, and MCP has no session: a client connects, calls a
# function and disconnects. Exposing them here would hand out a route around
# the verification gate and the data-isolation check, which is the whole point
# of both. So MCP gets what is safe without an identity: the knowledge base,
# which is the same for every customer, and the health metrics.
#
# Anything account-related goes through the chat API, where a conversation can
# actually prove who it is talking to.
from llm_agent import TOOLS_BY_NAME as _AGENT_TOOLS

MCP_EXPOSED = ("search_knowledge", "list_policy_topics")


def _mcp_handlers(tools: "CustomerChatbotTools"):
    """Dispatch table. Generated from the same names as the schemas above, so
    the two cannot drift -- the hand-written copy advertised three tools it
    could not run and could run two it never advertised."""
    return {
        "search_knowledge": lambda a: tools.search_knowledge(
            a.get("query"), a.get("top_k", 4)),
        "list_policy_topics": lambda a: tools.list_policy_topics(),
        "get_system_metrics": lambda a: tools.get_system_metrics(),
    }


TOOL_SCHEMAS: Dict[str, ToolSchema] = {
    **{
        name: ToolSchema(
            name=name,
            description=_AGENT_TOOLS[name]["description"],
            input_schema=_AGENT_TOOLS[name]["parameters"],
        )
        for name in MCP_EXPOSED
    },
    "get_system_metrics": ToolSchema(
        name="get_system_metrics",
        description="Retrieve current system health metrics including database status, response times, error rates, and pending support tickets. Useful for monitoring system health or providing uptime information.",
        input_schema={"type": "object", "properties": {}},
    ),
}


# ============================================================================
# MCP Server Protocol Implementation
# ============================================================================

class MCPServer:
    """
    Model Context Protocol Server for CustomerChatbot.

    This server implements the MCP specification to expose chatbot tools to Claude
    and other AI models. It handles initialization, tool calls, and resource queries.
    """

    def __init__(self, transport_type: TransportType = TransportType.STDIO):
        self.transport_type = transport_type
        self.tools = CustomerChatbotTools()
        self.request_id = 0
        logger.info(f"MCP Server initialized with {transport_type.value} transport")

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the MCP server and return protocol information.

        Returns:
            Dictionary with server capabilities and version info
        """
        logger.info("MCP Server initialization requested")

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": list(TOOL_SCHEMAS.keys()),
                "resources": True,
                "sampling": False,
                "roots": False
            },
            "serverInfo": {
                "name": "CustomerChatbot MCP Server",
                "version": "1.0.0",
                "description": "MCP server for e-commerce customer support chatbot"
            }
        }

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route tool calls to appropriate implementations.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments passed to the tool

        Returns:
            Tool execution result
        """
        logger.info(f"Handling tool call: {tool_name} with args: {json.dumps(arguments)}")

        handlers = _mcp_handlers(self.tools)

        handler = handlers.get(tool_name)
        if handler is None:
            return {"type": "error", "error": f"Unknown tool: {tool_name}"}

        try:
            result = await handler(arguments)

            return {
                "type": "tool_result",
                "name": tool_name,
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "executionTimeMs": result.execution_time_ms
            }
        except Exception as e:
            logger.error(f"Error handling tool call {tool_name}: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "error": f"Failed to execute tool {tool_name}: {str(e)}"
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools.

        Returns:
            List of tool definitions with schemas
        """
        tools = []
        for schema in TOOL_SCHEMAS.values():
            tools.append({
                "name": schema.name,
                "description": schema.description,
                "inputSchema": schema.input_schema
            })
        return tools

    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming MCP protocol messages.

        Args:
            message: MCP protocol message

        Returns:
            Response message
        """
        try:
            message_type = message.get("type")
            logger.info(f"Handling MCP message type: {message_type}")

            if message_type == "initialize":
                return {
                    "type": "server_response",
                    "content": await self.initialize()
                }

            elif message_type == "list_tools":
                return {
                    "type": "tools_response",
                    "tools": await self.list_tools()
                }

            elif message_type == "call_tool":
                tool_name = message.get("name")
                arguments = message.get("arguments", {})
                return await self.handle_tool_call(tool_name, arguments)

            else:
                return {
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                }
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "error": f"Server error: {str(e)}"
            }


# ============================================================================
# Transport Implementations
# ============================================================================

async def run_stdio_server():
    """Run the MCP server using stdio transport."""
    server = MCPServer(TransportType.STDIO)
    logger.info("Starting stdio transport MCP server")

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                message = json.loads(line)
                response = await server.handle_message(message)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {str(e)}")
                print(json.dumps({"type": "error", "error": "Invalid JSON"}))
                sys.stdout.flush()
    except KeyboardInterrupt:
        logger.info("Stdio server interrupted")
    except Exception as e:
        logger.error(f"Stdio server error: {str(e)}", exc_info=True)


async def run_http_server(host: str = "localhost", port: int = 8000):
    """
    Run the MCP server using HTTP bridge transport.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    try:
        from aiohttp import web
    except ImportError:
        logger.error("aiohttp not installed. Install with: pip install aiohttp")
        return

    server = MCPServer(TransportType.HTTP)

    async def handle_mcp_message(request):
        try:
            message = await request.json()
            response = await server.handle_message(message)
            return web.json_response(response)
        except Exception as e:
            logger.error(f"HTTP handler error: {str(e)}", exc_info=True)
            return web.json_response(
                {"type": "error", "error": str(e)},
                status=500
            )

    app = web.Application()
    app.router.add_post("/mcp", handle_mcp_message)
    app.router.add_post("/tools", handle_mcp_message)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"HTTP MCP server running at http://{host}:{port}")

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("HTTP server interrupted")
    finally:
        await runner.cleanup()


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """
    Main entry point for the MCP server.

    Supports both stdio and HTTP transport via command line arguments.

    Usage:
        python mcp_server.py                    # Run with stdio transport
        python mcp_server.py --http             # Run with HTTP transport
        python mcp_server.py --http --port 8080 # Run HTTP on custom port
    """
    import sys

    if "--http" in sys.argv:
        port = 8000
        if "--port" in sys.argv:
            port_idx = sys.argv.index("--port")
            if port_idx + 1 < len(sys.argv):
                port = int(sys.argv[port_idx + 1])
        await run_http_server(port=port)
    else:
        await run_stdio_server()


if __name__ == "__main__":
    asyncio.run(main())
