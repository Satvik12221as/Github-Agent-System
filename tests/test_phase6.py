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
    assert validate_tests(valid_tests) == True


def test_validate_tests_empty():
    """Empty string should fail."""
    assert validate_tests("") == False


def test_validate_tests_no_test_functions():
    """Code without test_ functions should fail."""
    no_tests = """
def helper():
    return True

class MyClass:
    pass
"""
    assert validate_tests(no_tests) == False


def test_validate_tests_minimal():
    """Minimal valid test should pass."""
    minimal = "def test_basic():\n    assert True"
    assert validate_tests(minimal) == True


# ─────────────────────────────────────────
# Tests for run_test_writer()
# ─────────────────────────────────────────

def test_writer_passes_on_success():
    """
    When tests pass in Docker, test_result should be 'passed'.
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
    mock_result = {
        "status": "passed",
        "output": "1 passed",
        "exit_code": 0
    }

    with patch(
        "agents.test_writer.generate_tests",
        return_value=mock_tests
    ):
        with patch(
            "agents.test_writer.run_tests_in_docker",
            return_value=mock_result
        ):
            updated_state = run_test_writer(state)

    assert updated_state["test_result"] == "passed"
    assert updated_state["tests"] == mock_tests
    assert updated_state["error"] is None
    assert updated_state["steps"] == 1


def test_writer_fails_on_test_failure():
    """
    When tests fail in Docker, test_result should be 'failed'.
    """
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix the handler"
    state["patch"] = """--- a/f.py
+++ b/f.py
@@ -1,1 +1,1 @@
-pass
+broken"""
    state["code_context"] = {"f.py": "pass"}

    mock_tests = "def test_something():\n    assert False"
    mock_result = {
        "status": "failed",
        "output": "1 failed - AssertionError",
        "exit_code": 1
    }

    with patch(
        "agents.test_writer.generate_tests",
        return_value=mock_tests
    ):
        with patch(
            "agents.test_writer.run_tests_in_docker",
            return_value=mock_result
        ):
            updated_state = run_test_writer(state)

    assert updated_state["test_result"] == "failed"
    assert updated_state["error"] is not None
    assert "failed" in updated_state["error"].lower()


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
        with patch(
            "agents.test_writer.run_tests_in_docker",
            return_value={
                "status": "passed",
                "output": "1 passed",
                "exit_code": 0
            }
        ):
            updated_state = run_test_writer(state)

    assert updated_state["steps"] == 1