# tests/test_phase6.py

import pytest
from unittest.mock import patch
from state import get_initial_state
from agents.test_writer import (
    validate_tests,
    clean_llm_output,
    run_test_writer
)


# ─────────────────────────────────────────
# Tests for validate_tests()
# ─────────────────────────────────────────

def test_validate_tests_valid():
    """Valid pytest code should pass validation."""
    valid_tests = """
import pytest

def test_button_click():
    assert True

def test_handler_exists():
    assert callable(lambda: None)
"""
    assert validate_tests(valid_tests)["valid"] == True


def test_validate_tests_empty():
    """Empty string should fail."""
    assert validate_tests("")["valid"] == False


def test_validate_tests_no_test_functions():
    """Code without test_ functions should fail."""
    no_tests = """
def helper():
    return True

class MyClass:
    pass
"""
    assert validate_tests(no_tests)["valid"] == False


def test_validate_tests_minimal():
    """Minimal valid test should pass."""
    minimal = "def test_basic():\n    assert True"
    assert validate_tests(minimal)["valid"] == True


# ─────────────────────────────────────────
# Tests for run_test_writer()
# ─────────────────────────────────────────

def test_writer_passes_on_success():
    """
    When tests are generated successfully, test_result should be 'passed'.
    """
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix the onclick handler"
    state["patch"] = """--- a/button.py
+++ b/button.py
@@ -1,2 +1,3 @@
 def render():
-    pass
+    return 'fixed'"""
    state["code_context"] = {
        "button.py": "def render():\n    pass"
    }

    mock_tests = "def test_render():\n    assert True"

    with patch(
        "agents.test_writer.generate_tests",
        return_value=mock_tests
    ):
        updated_state = run_test_writer(state)

    assert updated_state["test_result"] == "passed"
    assert updated_state["tests"] == mock_tests
    assert updated_state["error"] is None
    assert updated_state["steps"] == 1


def test_writer_fails_without_patch():
    """
    If no patch in state, agent should set error immediately.
    """
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix something"
    state["patch"] = ""

    updated_state = run_test_writer(state)

    assert updated_state["test_result"] == "failed"
    assert updated_state["error"] is not None
    assert "no patch" in updated_state["error"].lower()


def test_writer_increments_steps():
    """Steps counter must increment by 1."""
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix it"
    state["patch"] = """--- a/f.py
+++ b/f.py
@@ -1,1 +1,1 @@
-old
+new"""
    state["code_context"] = {"f.py": "old"}

    with patch(
        "agents.test_writer.generate_tests",
        return_value="def test_x():\n    assert True"
    ):
        updated_state = run_test_writer(state)

    assert updated_state["steps"] == 1