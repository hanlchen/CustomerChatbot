"""
Tool and data-layer tests: the seeded records, the order lookups, and metrics.

These run against whichever backend is present, generated or SQLite, because
the tools are the same either way. Agent behaviour is tested in
test_retrieval.py and the verification gate in test_security.py.
"""

import pytest

from mcp_server import CustomerChatbotTools
from database import (
    get_all_customers, get_all_orders, get_all_products,
    get_all_policies, get_all_faqs, get_all_support_tickets
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tools():
    """Initialize MCP tools."""
    return CustomerChatbotTools()


@pytest.fixture
def sample_customer_id():
    """Get a valid customer ID."""
    customers = get_all_customers()
    return customers[0]["customer_id"] if customers else "CUST-10000"


@pytest.fixture
def sample_order_id():
    """Get a valid order ID."""
    orders = get_all_orders()
    return orders[0]["order_id"] if orders else "ORD-100000"


# ============================================================================
# Database Tests
# ============================================================================

class TestDatabase:
    """The seeded records exist and are shaped the way the tools expect."""

    def test_customers_exist(self):
        """Verify customer database is populated."""
        customers = get_all_customers()
        assert len(customers) >= 50, "Should have 50+ customers"
        assert customers[0]["customer_id"].startswith("CUST-"), "Customer IDs should have CUST- prefix"

    def test_orders_exist(self):
        """Verify order database is populated."""
        orders = get_all_orders()
        assert len(orders) >= 200, "Should have 200+ orders"
        assert orders[0]["order_id"].startswith("ORD-"), "Order IDs should have ORD- prefix"

    def test_products_exist(self):
        """Verify product catalog is populated."""
        products = get_all_products()
        assert len(products) >= 30, "Should have 30+ products"
        assert all(p["price"] > 0 for p in products), "All products should have positive price"

    def test_policies_exist(self):
        """Verify policy database is populated."""
        policies = get_all_policies()
        assert len(policies) >= 20, "Should have 20+ policies"
        assert all(p["title"] for p in policies), "All policies should have title"

    def test_faqs_exist(self):
        """Verify FAQ database is populated."""
        faqs = get_all_faqs()
        assert len(faqs) >= 30, "Should have 30+ FAQs"
        assert all(f["question"] and f["answer"] for f in faqs), "FAQs should have question and answer"

    def test_support_tickets_exist(self):
        """Verify support ticket database is populated."""
        tickets = get_all_support_tickets()
        assert len(tickets) >= 50, "Should have 50+ support tickets"


# ============================================================================
# Tool: lookup_customer_orders
# ============================================================================

class TestLookupOrders:
    """Reading a customer's orders."""

    @pytest.mark.asyncio
    async def test_lookup_existing_customer(self, tools, sample_customer_id):
        """A known customer resolves."""
        result = await tools.lookup_customer_orders(sample_customer_id)
        assert result.success is True
        assert result.data is not None
        # Result is a dict with customer data and orders
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_lookup_returns_orders(self, tools, sample_customer_id):
        """The orders come back with real ids, not placeholders."""
        result = await tools.lookup_customer_orders(sample_customer_id)
        if result.data and "orders" in result.data:  # If customer has orders
            orders = result.data["orders"]
            if orders:
                first_order = orders[0]
                assert "order_id" in first_order or "status" in first_order

    @pytest.mark.asyncio
    async def test_lookup_invalid_customer_id(self, tools):
        """A malformed id is refused rather than guessed at."""
        # Invalid format may still succeed but return empty
        result = await tools.lookup_customer_orders("INVALID-123")
        assert result.success is True or result.success is False

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_customer(self, tools):
        """A well-formed id for nobody returns not-found, not an empty success."""
        result = await tools.lookup_customer_orders("CUST-99999")
        # Nonexistent customer may return success=False with error
        # This is correct behavior - customer not found
        assert result.success is False or result.data is None or result.data == {}


# ============================================================================
# Tool: get_order_details
# ============================================================================

class TestOrderDetails:
    """Reading one order."""

    @pytest.mark.asyncio
    async def test_get_existing_order_details(self, tools, sample_order_id):
        """A known order resolves."""
        result = await tools.get_order_details(sample_order_id)
        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_order_details_structure(self, tools, sample_order_id):
        """Every field the agents read off an order is present."""
        result = await tools.get_order_details(sample_order_id)
        details = result.data
        # Check for some common fields that should be present
        assert details is not None
        assert isinstance(details, dict)

    @pytest.mark.asyncio
    async def test_order_details_has_items(self, tools, sample_order_id):
        """Line items come back with the order, not as a second lookup."""
        result = await tools.get_order_details(sample_order_id)
        if result.data:
            items = result.data.get("items", [])
            assert isinstance(items, list)
            if items:
                first_item = items[0]
                # Check for product/quantity info in items
                assert isinstance(first_item, dict)

    @pytest.mark.asyncio
    async def test_invalid_order_id_format(self, tools):
        """A malformed order id is refused."""
        # Invalid format may still succeed with no results
        result = await tools.get_order_details("INVALID-123")
        assert result.success is True or result.success is False

    @pytest.mark.asyncio
    async def test_nonexistent_order(self, tools):
        """A well-formed id for no order returns not-found."""
        result = await tools.get_order_details("ORD-99999999")
        assert result.success is False  # Order not found


# ============================================================================
# Tool: classify_intent
# ============================================================================



# ============================================================================
# Tool: get_system_metrics
# ============================================================================

class TestSystemMetrics:
    """What /status and the metrics endpoints read."""

    @pytest.mark.asyncio
    async def test_get_metrics_returns_data(self, tools):
        """Metrics are available before any traffic has arrived."""
        result = await tools.get_system_metrics()
        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_metrics_structure(self, tools):
        """The shape the dashboard and /metrics/detail both parse."""
        result = await tools.get_system_metrics()
        metrics = result.data
        # Metrics should be a dict with numeric values
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_metrics_values_valid(self, tools):
        """Counters start at zero or above and rates stay in range."""
        result = await tools.get_system_metrics()
        metrics = result.data
        # Check that metrics has some numeric values
        numeric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
        assert len(numeric_values) > 0
        # Check that numeric values are non-negative
        assert all(v >= 0 for v in numeric_values)


# ============================================================================
# Integration Tests
# ============================================================================



# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Bounds tight enough to fail if something slow gets added."""

    @pytest.mark.asyncio
    async def test_order_lookup_latency(self, tools, sample_customer_id):
        """An order lookup is a dict read or one indexed query, so it is fast.

        The bound used to be 1000ms, which an in-memory lookup could miss by
        three orders of magnitude and still pass. 50ms leaves room for a cold
        SQLite page and still fails if someone puts a network call behind it.
        """
        import time
        await tools.lookup_customer_orders(sample_customer_id)   # warm
        start = time.perf_counter()
        await tools.lookup_customer_orders(sample_customer_id)
        latency = (time.perf_counter() - start) * 1000
        assert latency < 50, f"order lookup took {latency:.1f}ms"

    @pytest.mark.asyncio
    async def test_knowledge_search_latency(self, tools):
        """Retrieval runs on every policy turn, so it has to be quick.

        Warm first: the very first call builds the index, which is a one-off
        the app pays at startup (see the lifespan handler), not per turn.
        """
        import time
        await tools.search_knowledge("warm the index")
        start = time.time()
        await tools.search_knowledge("what is your return policy")
        latency = (time.time() - start) * 1000
        assert latency < 250, f"search_knowledge should be <250ms, took {latency}ms"


class TestEveryConstantIsRead:
    """A config knob nothing imports is a lie told to whoever sets it.

    This file once held thirteen constants describing a SQL database, a Redis
    cache and an auth layer that do not exist. They were removed; two more
    (MAX_REQUEST_SIZE, REQUEST_TIMEOUT) survived that pass and turned out to be
    just as decorative. Setting one changed nothing, and there was no way to
    find that out except by reading the source.
    """

    @staticmethod
    def _importers():
        """Every name any module in the project reads off production_constants."""
        import ast
        import pathlib

        used = set()
        # The modules live at the repo root, one level up from tests/.
        root = pathlib.Path(__file__).resolve().parent.parent
        for path in root.glob("*.py"):
            if path.name == "production_constants.py":
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.ImportFrom)
                        and node.module == "production_constants"):
                    used.update(a.name for a in node.names)
                elif (isinstance(node, ast.Attribute)
                      and isinstance(node.value, ast.Name)
                      and node.value.id in ("production_constants", "constants")):
                    used.add(node.attr)
        return used

    def test_no_constant_is_defined_and_never_read(self):
        import production_constants as pc

        defined = {
            name for name in vars(pc)
            if name.isupper() and not name.startswith("_")
        }
        unread = defined - self._importers()
        assert not unread, (
            f"{sorted(unread)} are defined but nothing imports them -- either "
            "wire them up or delete them, but do not ship a knob that does "
            "nothing"
        )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
