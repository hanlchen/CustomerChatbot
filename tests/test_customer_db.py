"""
Tests for the SQLite customer store.

Two things have to be true for this to be a safe swap. The records that come
out must be indistinguishable from the generated ones -- otherwise every
caller above `database.py` silently changes behaviour. And the queries must be
parameterised, because this project had no SQL at all until now and the README
said so rather than claiming an injection defence it had not earned.
"""

import json
import os
import re
import sqlite3

import pytest

import customer_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A seeded database on its own file, with the store reset around it."""
    path = str(tmp_path / "customer.db")
    monkeypatch.setenv("CHATBOT_CUSTOMER_DB", path)
    customer_db.reset()
    customer_db.seed(path, overwrite=True)
    monkeypatch.setenv("CHATBOT_CUSTOMER_DB", path)
    customer_db.reset()
    store = customer_db.active()
    yield store
    customer_db.reset()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

class TestSeeding:

    def test_it_writes_every_record(self, tmp_path):
        result = customer_db.seed(str(tmp_path / "x.db"), overwrite=True)
        assert result["customers"] == 55
        assert result["orders"] == 250
        assert result["order_items"] > 0
        customer_db.reset()

    def test_it_refuses_to_clobber_without_being_told(self, tmp_path):
        path = str(tmp_path / "x.db")
        customer_db.seed(path, overwrite=True)
        with pytest.raises(FileExistsError):
            customer_db.seed(path)
        customer_db.reset()

    def test_line_items_keep_their_order(self, db):
        """ORD-100078 has the same product twice; position must survive."""
        order = db.order("ORD-100078")
        names = [i["product_name"] for i in order["items"]]
        assert names == ["Cooling Pad", "Pen Set Premium", "Cooling Pad",
                         "Desk Organizer"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class TestTheRecordsComeBackWhole:
    """A dict from SQLite must be usable anywhere a generated one was."""

    def test_a_customer_has_the_fields_callers_use(self, db):
        c = db.customer_by_id("CUST-10000")
        for field in ("customer_id", "first_name", "last_name", "email",
                      "phone", "shipping_address"):
            assert c.get(field) is not None, f"{field} missing"

    def test_nested_addresses_are_objects_not_json_strings(self, db):
        """They round-trip through a TEXT column; they must come back typed."""
        c = db.customer_by_id("CUST-10000")
        assert isinstance(c["shipping_address"], dict)
        assert c["shipping_address"].get("city")

    def test_booleans_are_booleans(self, db):
        """SQLite has no bool. `return_eligible` is read as a truth value."""
        order = db.order("ORD-100078")
        assert isinstance(order["return_eligible"], bool)
        assert all(isinstance(i["return_eligible"], bool) for i in order["items"])

    def test_an_order_carries_its_items(self, db):
        order = db.order("ORD-100078")
        assert order["items"], "items were not attached"
        assert order["items"][0]["product_name"]

    def test_orders_come_back_newest_first(self, db):
        """`list_my_orders` and 'my most recent order' both depend on this."""
        orders = db.orders_for("CUST-10000")
        dates = [o["order_date"] for o in orders]
        assert dates == sorted(dates, reverse=True)

    def test_email_lookup_is_case_insensitive(self, db):
        lower = db.customer_by_email("laura.evans0@email.com")
        upper = db.customer_by_email("LAURA.EVANS0@EMAIL.COM")
        assert lower and upper
        assert lower["customer_id"] == upper["customer_id"]

    def test_a_missing_record_is_none_not_an_error(self, db):
        assert db.customer_by_id("CUST-99999") is None
        assert db.order("ORD-999999") is None
        assert db.orders_for("CUST-99999") == []


class TestItMatchesTheGeneratedRecords:
    """The swap is only safe if callers cannot tell the difference."""

    TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    @classmethod
    def _strip_times(cls, value):
        # Dates are computed relative to now at generation time, so a seeded
        # file is a snapshot and its timestamps are frozen. Everything else
        # must be identical.
        if isinstance(value, dict):
            return {k: cls._strip_times(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._strip_times(v) for v in value]
        if isinstance(value, str) and cls.TIMESTAMP.match(value):
            return "<timestamp>"
        return value

    def _normalise(self, value):
        return self._strip_times(
            json.loads(json.dumps(value, sort_keys=True, default=str)))

    def test_a_customer_is_field_for_field_identical(self, db):
        import database

        generated = database._db.customers["CUST-10000"]
        assert self._normalise(db.customer_by_id("CUST-10000")) == \
            self._normalise(generated)

    def test_an_order_is_field_for_field_identical(self, db):
        import database

        generated = database._db.orders["ORD-100078"]
        assert self._normalise(db.order("ORD-100078")) == \
            self._normalise(generated)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

class TestEveryQueryIsParameterised:
    """Adding SQL adds the risk. This is where that gets paid for.

    The README used to say plainly that there was no SQL, and therefore no
    SQL-injection story to tell. Now there is SQL, so the claim has to be
    earned rather than asserted.
    """

    HOSTILE = [
        "CUST-10000'; DROP TABLE customers; --",
        "' OR '1'='1",
        "' OR 1=1 --",
        "CUST-10000' UNION SELECT * FROM customers --",
        '"; DELETE FROM orders; --',
        "\\'; DROP TABLE orders; --",
    ]

    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_hostile_ids_find_nothing_and_break_nothing(self, db, hostile):
        assert db.customer_by_id(hostile) is None
        assert db.order(hostile) is None
        assert db.orders_for(hostile) == []
        # And the tables are still there afterwards.
        assert db.counts()["customers"] == 55

    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_hostile_emails_find_nothing(self, db, hostile):
        assert db.customer_by_email(hostile) is None

    def test_or_1_equals_1_does_not_return_the_first_row(self, db):
        """The classic. A concatenated query would hand back a customer."""
        assert db.customer_by_id("' OR '1'='1") is None
        assert db.customer_by_email("' OR '1'='1") is None

    def test_no_query_in_this_module_is_built_by_string_formatting(self):
        """Read the source. A parameterised codebase has no f-string SQL.

        Cheap, blunt, and it catches the one mistake that matters: someone
        adding a query later and interpolating a value into it.
        """
        import inspect

        source = inspect.getsource(customer_db)
        for line in source.splitlines():
            stripped = line.strip()
            if not any(verb in stripped.upper() for verb in
                       ("SELECT ", "INSERT ", "UPDATE ", "DELETE ")):
                continue
            assert not stripped.startswith('f"'), f"f-string SQL: {stripped}"
            assert ".format(" not in stripped, f"formatted SQL: {stripped}"
            assert "%" not in stripped, f"%-interpolated SQL: {stripped}"

    def test_the_store_cannot_write(self, db):
        """Opened query_only, so a write fails even if one is ever attempted."""
        with pytest.raises(sqlite3.OperationalError):
            db._conn.execute("DELETE FROM customers")


# ---------------------------------------------------------------------------
# Falling back
# ---------------------------------------------------------------------------

class TestNoDatabaseIsNormal:
    """A fresh clone and CI have no file. That must not be an error."""

    def test_a_missing_file_yields_no_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATBOT_CUSTOMER_DB", str(tmp_path / "absent.db"))
        customer_db.reset()
        assert customer_db.active() is None
        customer_db.reset()

    def test_an_empty_setting_disables_it(self, monkeypatch):
        monkeypatch.setenv("CHATBOT_CUSTOMER_DB", "")
        customer_db.reset()
        assert customer_db.active() is None
        customer_db.reset()

    def test_a_corrupt_file_falls_back_rather_than_raising(self, tmp_path, monkeypatch):
        path = tmp_path / "broken.db"
        path.write_text("this is not a database")
        monkeypatch.setenv("CHATBOT_CUSTOMER_DB", str(path))
        customer_db.reset()
        assert customer_db.active() is None      # must not raise
        customer_db.reset()

    def test_database_functions_still_work_with_no_store(self, monkeypatch):
        import database

        monkeypatch.setenv("CHATBOT_CUSTOMER_DB", "")
        customer_db.reset()
        assert database.get_customer_by_id("CUST-10000") is not None
        assert database.get_order_details("ORD-100078") is not None
        customer_db.reset()


class TestTheToolsReadThroughIt:
    """The point of the exercise: the MCP tools search the database."""

    @pytest.mark.asyncio
    async def test_order_details_comes_from_sqlite(self, db):
        from mcp_server import CustomerChatbotTools

        result = await CustomerChatbotTools().get_order_details("ORD-100078")
        assert result.success
        assert result.data["order_id"] == "ORD-100078"

    @pytest.mark.asyncio
    async def test_customer_orders_come_from_sqlite(self, db):
        from mcp_server import CustomerChatbotTools

        result = await CustomerChatbotTools().lookup_customer_orders("CUST-10000")
        assert result.success
        assert result.data["orders"]
        assert all(o["order_id"].startswith("ORD-") for o in result.data["orders"])

    @pytest.mark.asyncio
    async def test_verification_reads_the_phone_from_sqlite(self, db):
        """The gate depends on this: no phone, no verification."""
        from mcp_server import CustomerChatbotTools

        result = await CustomerChatbotTools().begin_verification("CUST-10000")
        assert result.success
        assert result.data["found"] is True
        assert result.data["phone_hint"]
