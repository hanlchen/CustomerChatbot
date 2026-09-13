"""
The customer records, in SQLite instead of in memory.

`database.py` generates 55 customers and 250 orders from a fixed seed at
import. That is deterministic and fast, which is exactly right for tests --
and useless as a product: nothing survives a restart, nothing can change, and
there is no way to look at the data except by running Python.

This puts the same records in a real file. It is deliberately a *drop-in*: the
eleven functions in `database.py` keep their signatures and keep returning the
same dicts, so `mcp_server.py`, the agents, the API and the whole test suite
are untouched. Only where the bytes live changes.

    python seed_db.py                 # build customer.db from the seeded data
    sqlite3 customer.db "SELECT customer_id, phone FROM customers LIMIT 5;"

If `customer.db` is absent, `database.py` falls back to generating in memory
exactly as before -- so a fresh clone, and CI, work with no extra step.

    CHATBOT_CUSTOMER_DB    path to the file (default: customer.db)
    CHATBOT_CUSTOMER_DB=   empty disables it and forces the in-memory path

**Every query here is parameterised.** That is not decoration: until now this
project had no SQL at all, and the README said so rather than claiming an
injection defence it had not earned. Introducing SQL introduces the risk, so
the rule is absolute -- no f-string, no `%`, no `.format()` ever reaches a
cursor. `test_customer_db.py` asserts it against hostile input.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = "customer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id    TEXT PRIMARY KEY,
    first_name     TEXT,
    last_name      TEXT,
    email          TEXT,
    phone          TEXT,
    tier           TEXT,
    created_at     TEXT,
    last_order_date TEXT,
    total_orders   INTEGER,
    lifetime_value REAL,
    preferred_shipping_method TEXT,
    -- Addresses are nested objects with no query of their own; JSON keeps the
    -- round trip exact rather than flattening and rebuilding six columns.
    shipping_address TEXT,
    billing_address  TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    order_date      TEXT,
    status          TEXT,
    shipping_status TEXT,
    subtotal        REAL,
    tax             REAL,
    shipping_cost   REAL,
    total_amount    REAL,
    tracking_number TEXT,
    estimated_delivery TEXT,
    actual_delivery    TEXT,
    return_deadline    TEXT,
    return_eligible    INTEGER,
    notes           TEXT,
    shipping_address TEXT
);

-- A separate table rather than a JSON blob: line items are the one nested
-- thing worth querying on its own ("which orders contain a Cooling Pad").
CREATE TABLE IF NOT EXISTS order_items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     TEXT NOT NULL,
    position     INTEGER NOT NULL,
    product_id   TEXT,
    product_name TEXT,
    category     TEXT,
    quantity     INTEGER,
    unit_price   REAL,
    total_price  REAL,
    return_eligible INTEGER
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_items_order     ON order_items(order_id);
-- Lookups are case-insensitive at the call site (ids are normalised before
-- they get here), but email search is a plain equality on lowercase.
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
"""

# Columns stored as JSON text and rehydrated on read, so a record coming out
# is indistinguishable from one the generator made.
_JSON_COLUMNS = {"shipping_address", "billing_address"}


def _path() -> Optional[str]:
    """Where the file lives, or None when the DB path is explicitly empty."""
    value = os.environ.get("CHATBOT_CUSTOMER_DB", DEFAULT_PATH)
    return value or None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class CustomerStore:
    """Read-only access to the customer records.

    Read-only on purpose. Every write path in this app is a tool the model can
    reach, and adding one is a separate decision with its own gate -- not
    something to inherit by accident from swapping the storage layer.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA query_only = ON")

    # -- plumbing ---------------------------------------------------------

    def _rows(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    @staticmethod
    def _hydrate(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        record = dict(row)
        for column in _JSON_COLUMNS:
            if column in record and isinstance(record[column], str):
                try:
                    record[column] = json.loads(record[column])
                except (ValueError, TypeError):
                    pass
        if "return_eligible" in record and record["return_eligible"] is not None:
            record["return_eligible"] = bool(record["return_eligible"])
        return record

    def _order_with_items(self, row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        order = self._hydrate(row)
        if order is None:
            return None
        items = self._rows(
            "SELECT product_id, product_name, category, quantity, unit_price, "
            "total_price, return_eligible FROM order_items "
            "WHERE order_id = ? ORDER BY position", (order["order_id"],))
        order["items"] = [self._hydrate(item) for item in items]
        return order

    # -- the queries ------------------------------------------------------
    # Every one is parameterised. No exceptions, no "just this once".

    def customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        return self._hydrate(rows[0]) if rows else None

    def customer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM customers WHERE LOWER(email) = LOWER(?)", (email,))
        return self._hydrate(rows[0]) if rows else None

    def orders_for(self, customer_id: str) -> List[Dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM orders WHERE customer_id = ? "
            "ORDER BY order_date DESC", (customer_id,))
        return [self._order_with_items(row) for row in rows]

    def order(self, order_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        return self._order_with_items(rows[0]) if rows else None

    def all_customers(self) -> List[Dict[str, Any]]:
        return [self._hydrate(r) for r in
                self._rows("SELECT * FROM customers ORDER BY customer_id")]

    def all_orders(self) -> List[Dict[str, Any]]:
        return [self._order_with_items(r) for r in
                self._rows("SELECT * FROM orders ORDER BY order_id")]

    def counts(self) -> Dict[str, int]:
        return {
            "customers": self._rows("SELECT COUNT(*) AS n FROM customers")[0]["n"],
            "orders": self._rows("SELECT COUNT(*) AS n FROM orders")[0]["n"],
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:                              # pragma: no cover
            pass


_store: Optional[CustomerStore] = None
_resolved = False


def active() -> Optional[CustomerStore]:
    """The store, or None when there is no database to read.

    None is the normal, supported state: a fresh clone has no `customer.db`
    and `database.py` generates the records instead. A missing file is not an
    error and must never be treated as one.
    """
    global _store, _resolved
    if _resolved:
        return _store
    _resolved = True

    path = _path()
    if not path or not os.path.isfile(path):
        return None
    try:
        store = CustomerStore(path)
        counts = store.counts()
        if not counts["customers"]:
            logger.warning("%s has no customers; using generated data", path)
            store.close()
            return None
        logger.info("customer store: %s (%d customers, %d orders)",
                    path, counts["customers"], counts["orders"])
        _store = store
    except Exception as exc:                           # pragma: no cover
        # A corrupt or unreadable file must not take the app down -- it falls
        # back to the generated records, which always work.
        logger.error("could not open %s (%s); using generated data", path, exc)
        _store = None
    return _store


def reset() -> None:
    """Forget the resolved store. For tests, and after re-seeding."""
    global _store, _resolved
    if _store is not None:
        _store.close()
    _store, _resolved = None, False


# ---------------------------------------------------------------------------
# Writing (seeding only)
# ---------------------------------------------------------------------------

def seed(path: Optional[str] = None, *, overwrite: bool = False) -> Dict[str, int]:
    """Build the database from the seeded in-memory records.

    Imports `database` lazily: that module reads *this* one on its own read
    path, and a module-level import here would be circular.
    """
    target = path or _path() or DEFAULT_PATH
    if os.path.exists(target) and not overwrite:
        raise FileExistsError(
            f"{target} already exists. Pass overwrite=True (or --force) to rebuild.")
    if os.path.exists(target):
        os.remove(target)

    # Force the generated records even if a database is already active, so
    # re-seeding never copies a database onto itself.
    reset()
    previous = os.environ.get("CHATBOT_CUSTOMER_DB")
    os.environ["CHATBOT_CUSTOMER_DB"] = ""
    try:
        import database

        customers = list(database._db.customers.values())
        orders = list(database._db.orders.values())
    finally:
        if previous is None:
            os.environ.pop("CHATBOT_CUSTOMER_DB", None)
        else:
            os.environ["CHATBOT_CUSTOMER_DB"] = previous
        reset()

    conn = sqlite3.connect(target)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO customers (customer_id, first_name, last_name, email, "
            "phone, tier, created_at, last_order_date, total_orders, "
            "lifetime_value, preferred_shipping_method, shipping_address, "
            "billing_address) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(
                c.get("customer_id"), c.get("first_name"), c.get("last_name"),
                c.get("email"), c.get("phone"), c.get("tier"),
                c.get("created_at"), c.get("last_order_date"),
                c.get("total_orders"), c.get("lifetime_value"),
                c.get("preferred_shipping_method"),
                json.dumps(c.get("shipping_address")),
                json.dumps(c.get("billing_address")),
            ) for c in customers])

        conn.executemany(
            "INSERT INTO orders (order_id, customer_id, order_date, status, "
            "shipping_status, subtotal, tax, shipping_cost, total_amount, "
            "tracking_number, estimated_delivery, actual_delivery, "
            "return_deadline, return_eligible, notes, shipping_address) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(
                o.get("order_id"), o.get("customer_id"), o.get("order_date"),
                o.get("status"), o.get("shipping_status"), o.get("subtotal"),
                o.get("tax"), o.get("shipping_cost"), o.get("total_amount"),
                o.get("tracking_number"), o.get("estimated_delivery"),
                o.get("actual_delivery"), o.get("return_deadline"),
                int(bool(o.get("return_eligible"))), o.get("notes"),
                json.dumps(o.get("shipping_address")),
            ) for o in orders])

        conn.executemany(
            "INSERT INTO order_items (order_id, position, product_id, "
            "product_name, category, quantity, unit_price, total_price, "
            "return_eligible) VALUES (?,?,?,?,?,?,?,?,?)",
            [(
                o.get("order_id"), index, item.get("product_id"),
                item.get("product_name"), item.get("category"),
                item.get("quantity"), item.get("unit_price"),
                item.get("total_price"), int(bool(item.get("return_eligible"))),
            ) for o in orders for index, item in enumerate(o.get("items") or [])])

        conn.commit()
        counts = {
            "customers": conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
            "orders": conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
            "order_items": conn.execute("SELECT COUNT(*) FROM order_items").fetchone()[0],
        }
    finally:
        conn.close()

    reset()
    return {"path": target, **counts}
