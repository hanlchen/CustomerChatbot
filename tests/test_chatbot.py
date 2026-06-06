"""Unit tests for the policy-aware SQL + RAG chatbot.

These cover the pure-logic and database units that do not require live API
calls: metrics tracking, policy-question detection, response caching, rate
limiting, the SQL-injection guard, and the SQLite validation helpers.
"""

import time
from datetime import datetime, timedelta

import pytest

import chatbot_v2_agents as bot


# ============= is_policy_question =============

class TestIsPolicyQuestion:
    @pytest.mark.parametrize("message", [
        "What is your return policy?",
        "Can I cancel my subscription?",
        "How do I get a refund?",
        "Do you offer free shipping?",
    ])
    def test_policy_questions_are_cacheable(self, message):
        assert bot.is_policy_question(message) is True

    @pytest.mark.parametrize("message", [
        "Where is my order?",
        "What did I buy last month?",
        "Track order number 12345",
        "When will my purchase arrive?",
    ])
    def test_data_queries_are_not_cacheable(self, message):
        assert bot.is_policy_question(message) is False

    def test_data_keyword_overrides_policy_keyword(self):
        # Contains "policy" (policy kw) AND "my order" (data kw); data wins.
        assert bot.is_policy_question("What is your policy on my order?") is False

    def test_unrelated_message_is_not_cacheable(self):
        assert bot.is_policy_question("The weather is nice today") is False

    def test_is_case_insensitive(self):
        assert bot.is_policy_question("WHAT IS YOUR RETURN POLICY") is True


# ============= MetricsTracker =============

class TestMetricsTracker:
    def test_starts_zeroed(self):
        m = bot.MetricsTracker()
        summary = m.get_summary()
        assert summary["total_requests"] == 0
        assert summary["total_tokens"] == 0
        assert summary["cache_hit_rate"] == "0.0%"

    def test_record_request_accumulates_tokens(self):
        m = bot.MetricsTracker()
        m.record_request(prompt_tokens=10, completion_tokens=5)
        m.record_request(prompt_tokens=20, completion_tokens=10)
        summary = m.get_summary()
        assert summary["total_requests"] == 2
        assert summary["prompt_tokens"] == 30
        assert summary["completion_tokens"] == 15
        assert summary["total_tokens"] == 45

    def test_cache_hit_rate(self):
        m = bot.MetricsTracker()
        m.record_cache_hit()
        m.record_cache_hit()
        m.record_cache_hit()
        m.record_cache_miss()
        assert m.get_summary()["cache_hit_rate"] == "75.0%"

    def test_avg_latency(self):
        m = bot.MetricsTracker()
        m.record_latency(100)
        m.record_latency(200)
        assert m.get_summary()["avg_latency_ms"] == "150"

    def test_reset_clears_state(self):
        m = bot.MetricsTracker()
        m.record_request(prompt_tokens=10, completion_tokens=5)
        m.record_error()
        m.reset()
        summary = m.get_summary()
        assert summary["total_requests"] == 0
        assert summary["total_tokens"] == 0
        assert summary["errors"] == 0


# ============= ResponseCache =============

class TestResponseCache:
    def test_caches_policy_question(self):
        cache = bot.ResponseCache(ttl_minutes=5)
        cache.set(1, "What is your return policy?", "30 days")
        assert cache.get(1, "What is your return policy?") == "30 days"

    def test_does_not_cache_data_query(self):
        cache = bot.ResponseCache(ttl_minutes=5)
        cache.set(1, "Where is my order?", "shipped")
        assert cache.get(1, "Where is my order?") is None

    def test_key_is_customer_agnostic_for_policies(self):
        cache = bot.ResponseCache(ttl_minutes=5)
        cache.set(1, "What is your return policy?", "30 days")
        # A different customer asking the same policy question hits cache.
        assert cache.get(999, "What is your return policy?") == "30 days"

    def test_normalizes_case_and_whitespace(self):
        cache = bot.ResponseCache(ttl_minutes=5)
        cache.set(1, "What is your return policy?", "30 days")
        assert cache.get(1, "  WHAT IS YOUR RETURN POLICY?  ") == "30 days"

    def test_expired_entry_is_evicted(self):
        cache = bot.ResponseCache(ttl_minutes=5)
        cache.set(1, "What is your return policy?", "30 days")
        # Force expiry by rewriting the entry's expiration into the past.
        key = cache._make_key("What is your return policy?")
        cache.cache[key]["expires"] = datetime.now() - timedelta(seconds=1)
        assert cache.get(1, "What is your return policy?") is None
        assert key not in cache.cache

    def test_hits_and_misses_counted(self):
        cache = bot.ResponseCache(ttl_minutes=5)
        cache.set(1, "What is your return policy?", "30 days")
        cache.get(1, "What is your return policy?")          # hit
        cache.get(1, "What is your cancellation policy?")    # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_clear_empties_cache(self):
        cache = bot.ResponseCache(ttl_minutes=5)
        cache.set(1, "What is your return policy?", "30 days")
        cache.clear()
        assert cache.stats()["total"] == 0


# ============= RateLimiter =============

class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = bot.RateLimiter(max_requests=3, window_minutes=1)
        for _ in range(3):
            allowed, msg = limiter.is_allowed(customer_id=1)
            assert allowed is True
            assert msg == ""

    def test_blocks_over_limit(self):
        limiter = bot.RateLimiter(max_requests=2, window_minutes=1)
        limiter.is_allowed(1)
        limiter.is_allowed(1)
        allowed, msg = limiter.is_allowed(1)
        assert allowed is False
        assert "Rate limit exceeded" in msg

    def test_customers_are_isolated(self):
        limiter = bot.RateLimiter(max_requests=1, window_minutes=1)
        assert limiter.is_allowed(1)[0] is True
        assert limiter.is_allowed(1)[0] is False
        # Different customer still has full budget.
        assert limiter.is_allowed(2)[0] is True

    def test_get_remaining(self):
        limiter = bot.RateLimiter(max_requests=5, window_minutes=1)
        assert limiter.get_remaining(1) == 5
        limiter.is_allowed(1)
        limiter.is_allowed(1)
        assert limiter.get_remaining(1) == 3

    def test_old_requests_expire_from_window(self):
        limiter = bot.RateLimiter(max_requests=2, window_minutes=1)
        # Inject two timestamps that are already outside the 1-minute window.
        stale = datetime.now() - timedelta(minutes=2)
        limiter.requests[1] = [stale, stale]
        allowed, _ = limiter.is_allowed(1)
        assert allowed is True
        assert limiter.get_remaining(1) == 1


# ============= is_safe_sql_query (injection guard) =============

class TestIsSafeSqlQuery:
    def test_allows_plain_select(self):
        safe, err = bot.is_safe_sql_query("SELECT order_id FROM orders WHERE customer_id = 1")
        assert safe is True
        assert err == ""

    @pytest.mark.parametrize("query", [
        "DROP TABLE customers",
        "DELETE FROM orders",
        "UPDATE customers SET phone = '0'",
        "INSERT INTO orders VALUES (1)",
        "ALTER TABLE orders ADD COLUMN x",
        "CREATE TABLE evil (id int)",
    ])
    def test_blocks_non_select_statements(self, query):
        safe, err = bot.is_safe_sql_query(query)
        assert safe is False
        assert err

    def test_blocks_select_with_forbidden_keyword(self):
        safe, err = bot.is_safe_sql_query("SELECT * FROM orders; DROP TABLE orders")
        assert safe is False
        assert "DROP" in err

    def test_blocks_sql_comment_injection(self):
        safe, err = bot.is_safe_sql_query("SELECT * FROM orders -- malicious")
        assert safe is False

    def test_blocks_email_queries(self):
        safe, err = bot.is_safe_sql_query("SELECT email FROM customers WHERE customer_id = 1")
        assert safe is False
        assert "email" in err.lower()


# ============= Database helpers (against bundled chatbot.db) =============

class TestDatabaseHelpers:
    def test_validate_known_email(self):
        result = bot.validate_customer_email("alice@example.com")
        assert result is not None
        assert result["customer_id"] == 1
        assert "phone" in result

    def test_validate_email_is_case_insensitive(self):
        assert bot.validate_customer_email("ALICE@EXAMPLE.COM") is not None

    def test_validate_unknown_email_returns_none(self):
        assert bot.validate_customer_email("nobody@nowhere.com") is None

    def test_validate_phone_matches_normalized(self):
        customer = bot.validate_customer_email("alice@example.com")
        cid = customer["customer_id"]
        # Same digits, different formatting, must still validate.
        formatted = customer["phone"].replace("-", " ")
        assert bot.validate_customer_phone(cid, formatted) is True

    def test_validate_phone_rejects_wrong_number(self):
        customer = bot.validate_customer_email("alice@example.com")
        assert bot.validate_customer_phone(customer["customer_id"], "000-0000") is False

    def test_execute_query_returns_dicts(self):
        rows = bot.execute_query("SELECT customer_id FROM customers LIMIT 1")
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)
        assert "customer_id" in rows[0]
