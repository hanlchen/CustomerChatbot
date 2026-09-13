"""
Regression tests for CustomerChatbot.

Each test here pins a bug that reached the running application while the
existing unit tests were green -- data-layer defects, ID handling, dataset
determinism, and configuration that was documented but never read.
"""

import os
import pathlib

import pytest

from conversation_manager import ConversationManager
from database import (
    get_customer_by_id,
    get_order_details,
    get_orders_by_customer,
)
from mcp_server import CustomerChatbotTools


# The repo root, one level up from tests/. Several tests here read project
# files off disk (pytest.ini, every module that imports production_constants),
# and after the move to tests/ `Path(__file__).parent` is no longer it.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def tools():
    return CustomerChatbotTools()


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class TestOrderIdMapping:
    """lookup_customer_orders returned order_id=None (read order["id"])."""

    @pytest.mark.asyncio
    async def test_lookup_returns_real_order_ids(self, tools):
        result = await tools.lookup_customer_orders("CUST-10000")
        assert result.success
        assert result.data["orders"], "customer should have orders"
        for order in result.data["orders"]:
            assert order["order_id"] is not None
            assert order["order_id"].startswith("ORD-")

    @pytest.mark.asyncio
    async def test_ids_from_lookup_resolve_to_details(self, tools):
        listing = await tools.lookup_customer_orders("CUST-10000")
        order_id = listing.data["orders"][0]["order_id"]
        details = await tools.get_order_details(order_id)
        assert details.success, f"{order_id} from lookup must resolve"
        assert details.data["order_id"] == order_id

    def test_customer_name_is_populated(self):
        customer = get_customer_by_id("CUST-10000")
        assert customer["first_name"] and customer["last_name"]


class TestIdNormalization:
    """Lookups were case-sensitive while the classifier lowercased IDs."""

    @pytest.mark.parametrize(
        "raw", ["CUST-10000", "cust-10000", "Cust-10000", "CUST 10000", "cust10000"]
    )
    def test_customer_lookup_accepts_any_casing(self, raw):
        assert get_customer_by_id(raw) is not None
        assert get_orders_by_customer(raw)

    def test_order_lookup_accepts_lowercase(self):
        order_id = get_orders_by_customer("CUST-10000")[0]["order_id"]
        assert get_order_details(order_id.lower()) is not None

    def test_unknown_customer_returns_none(self):
        assert get_customer_by_id("CUST-99999") is None


class TestDeterministicData:
    """CUST-10001 was a different person on every restart."""

    def test_dataset_is_stable_across_processes(self):
        import subprocess, sys

        code = (
            "from database import _db;"
            "c=_db.customers['CUST-10001'];"
            "print(c['first_name'], c['last_name'], c['email'])"
        )
        runs = {
            subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True).stdout.strip()
            for _ in range(2)
        }
        assert len(runs) == 1, f"dataset differs between processes: {runs}"

    def test_seeding_does_not_fix_global_randomness(self):
        import subprocess, sys

        code = "import database, random; print(random.random())"
        runs = {
            subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True).stdout.strip()
            for _ in range(2)
        }
        assert len(runs) == 2, "importing the database made random() predictable"


class TestSessionIsolation:
    def test_sessions_do_not_share_context(self):
        manager = ConversationManager()
        a = manager.create_session()
        b = manager.create_session()
        manager.update_context(a, customer_id="CUST-10000")
        assert manager.get_context(a).customer_id == "CUST-10000"
        assert manager.get_context(b).customer_id is None

    def test_history_is_per_session(self):
        manager = ConversationManager()
        a = manager.create_session()
        b = manager.create_session()
        manager.add_message(a, "customer", "hello")
        assert len(manager.get_message_history(a)) == 1
        assert len(manager.get_message_history(b)) == 0


class TestTestsAreActuallyCollected:
    """A test that does not run is worse than no test.

    It looks like coverage on the file listing and catches nothing. Both
    shapes of that bug have happened here: `testpaths` pointing at a stale
    directory so collection aborted, and an explicit file list that silently
    omitted test_telemetry.py and its 26 passing tests for weeks.
    """

    def test_testpaths_resolves_to_the_real_tests(self):
        """Every path pytest.ini names must exist and hold tests.

        The original bug was `testpaths = tests` pointing at a directory that
        held a retired prototype. Collection aborted, no real test ran, and
        nothing said so. A path that exists but is empty is the same failure
        wearing a green tick, so the count is checked too.
        """
        import configparser

        config = configparser.ConfigParser()
        config.read(PROJECT_ROOT / "pytest.ini")
        listed = config["pytest"]["testpaths"].split()
        assert listed, "pytest.ini names no testpaths at all"

        found = []
        for name in listed:
            path = PROJECT_ROOT / name
            assert path.exists(), f"pytest.ini names {name}, which does not exist"
            found += list(path.rglob("test_*.py")) if path.is_dir() else [path]

        assert len(found) >= 5, (
            f"testpaths {listed} resolves to only {len(found)} test files; "
            "bare `pytest` is running almost nothing"
        )

    def test_no_test_file_sits_outside_the_test_directory(self):
        """A test file at the repo root is not collected and looks like coverage."""
        strays = sorted(p.name for p in PROJECT_ROOT.glob("test_*.py"))
        assert not strays, (
            f"{strays} are at the repo root, where testpaths does not reach "
            "them. Move them into tests/."
        )


class TestContextHasNoGhostFields:
    """`_absorb` wrote four attributes ConversationContext does not declare.

    Python creates them on the instance, so nothing failed: the writes looked
    like state, a test asserted on one of them and passed, and no code could
    ever read them back -- they were not in `to_dict()`, not persisted, not
    seen by the next turn. Leftovers from the deleted rules engine.
    """

    def test_absorb_only_writes_declared_fields(self):
        import ast
        import inspect
        from dataclasses import fields

        from conversation_manager import ConversationContext
        from llm_agent import LLMAgent

        declared = {f.name for f in fields(ConversationContext)}
        source = inspect.getsource(LLMAgent._absorb)
        tree = ast.parse(inspect.cleandoc(source))

        written = {
            target.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "context"
        }
        ghosts = written - declared
        assert not ghosts, (
            f"_absorb writes {sorted(ghosts)}, which ConversationContext does "
            "not declare -- the value is set and can never be read back"
        )


class TestDotEnvIsActuallyRead:
    """The README told you to write a .env; nothing ever read it.

    Every setting had to be exported by hand in every terminal, and a fresh
    shell silently fell back to the rules engine with no explanation. Twice
    now a documented configuration file has turned out to be decorative --
    these tests are here so it stays wired up.
    """

    @pytest.fixture(autouse=True)
    def _sandboxed_environment(self, monkeypatch):
        """Swap os.environ for a copy.

        These tests deliberately mutate the environment, and `load` writes to
        os.environ directly -- which monkeypatch cannot track or undo. Without
        this, a leaked CHATBOT_BASE_URL makes an unrelated test later in the
        run resolve a model backend and fail.
        """
        monkeypatch.setattr(os, "environ", dict(os.environ))

    def test_values_reach_the_environment(self, tmp_path, monkeypatch):
        from env_file import load

        env = tmp_path / ".env"
        env.write_text("CHATBOT_BASE_URL=http://localhost:8000/v1\n")
        monkeypatch.delenv("CHATBOT_BASE_URL", raising=False)
        applied = load(env)
        assert applied == {"CHATBOT_BASE_URL": "http://localhost:8000/v1"}
        assert os.environ["CHATBOT_BASE_URL"] == "http://localhost:8000/v1"

    def test_an_exported_variable_beats_the_file(self, tmp_path, monkeypatch):
        """A file is a default. `FOO=x python app.py` must still mean x."""
        from env_file import load

        env = tmp_path / ".env"
        env.write_text("CHATBOT_MODEL=from-file\n")
        monkeypatch.setenv("CHATBOT_MODEL", "from-shell")
        load(env)
        assert os.environ["CHATBOT_MODEL"] == "from-shell"

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        from env_file import load

        assert load(tmp_path / "nothing-here") == {}

    def test_parses_the_shapes_people_actually_write(self):
        from env_file import parse

        pairs = dict(parse(
            "# a comment\n"
            "\n"
            "export CHATBOT_BASE_URL=http://localhost:8000/v1\n"
            'CHATBOT_MODEL="Qwen/Qwen3-8B"\n'
            "CHATBOT_API_KEY='local-no-key-required'\n"
            "EMPTY=\n"
            "not a setting\n"
        ))
        assert pairs["CHATBOT_BASE_URL"] == "http://localhost:8000/v1"
        assert pairs["CHATBOT_MODEL"] == "Qwen/Qwen3-8B"      # quotes stripped
        assert pairs["CHATBOT_API_KEY"] == "local-no-key-required"
        assert pairs["EMPTY"] == ""
        assert "not a setting" not in pairs

    def test_langsmith_settings_from_the_file_actually_switch_tracing_on(
            self, monkeypatch):
        """`.env` is where these will live; they must not be decorative.

        `tracing` resolves the environment once at import. Everything below
        happens in the wrong order if `.env` is read late, and the failure is
        silent -- the app runs, traces just never appear, and you find out
        after a debugging session that there was nothing to look at.
        """
        import tracing

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_not_a_real_key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "regression-check")
        try:
            tracing.reset_for_tests()
            assert tracing.enabled(), tracing.status()
            assert "regression-check" in tracing.status()
        finally:
            for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY",
                         "LANGSMITH_PROJECT", "LANGCHAIN_TRACING_V2",
                         "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
                monkeypatch.delenv(name, raising=False)
            tracing.reset_for_tests()
        assert not tracing.enabled()

    def test_tracing_asked_for_without_a_key_says_so(self, monkeypatch):
        """Half-configured must not look the same as switched off."""
        import tracing

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        try:
            tracing.reset_for_tests()
            assert not tracing.enabled()
            assert "API_KEY" in tracing.status(), tracing.status()
        finally:
            monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
            tracing.reset_for_tests()

    def test_the_agent_is_imported_after_the_env_file_is_read(self):
        """Import order is the whole ballgame for tracing.

        `tracing` decides on or off at import time, and it is pulled in by
        `llm_agent`. If anything imports that chain before `.env` is applied,
        LANGSMITH_TRACING in the file is read too late to matter.
        """
        import inspect

        import app

        source = inspect.getsource(app)
        assert source.index("_load_env_file()") < source.index("import telemetry"), \
            ".env must be applied before any module that reads it at import"

    def test_the_app_loads_it_at_import_time(self):
        """Settings resolve at import, so loading late is the same as never."""
        import inspect

        import app

        assert hasattr(app, "_ENV_FROM_FILE")
        source = inspect.getsource(app)
        assert source.index("_load_env_file") < source.index("from mcp_server"), \
            ".env must be loaded before anything reads os.environ"
