# tests/test_workflow.py

import pytest
from unittest.mock import patch, MagicMock
from state import get_initial_state
from workflow import (
    route_by_complexity,
    route_after_tests,
    check_for_errors,
    build_workflow
)


def test_route_by_complexity_simple():
    """
    When complexity is simple, routing function
    must return the string "simple".
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["complexity"] = "simple"

    result = route_by_complexity(state)

    assert result == "simple"


def test_route_by_complexity_complex():
    """
    When complexity is complex, routing function
    must return the string "complex".
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["complexity"] = "complex"

    result = route_by_complexity(state)

    assert result == "complex"


def test_route_after_tests_passed():
    """
    When test_result is passed, must return "open_pr".
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["test_result"] = "passed"
    state["retry_count"] = 0

    result = route_after_tests(state)

    assert result == "open_pr"


def test_route_after_tests_failed():
    """
    When test_result is failed and retries remain,
    must return "retry" and increment retry_count.
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["test_result"] = "failed"
    state["retry_count"] = 0

    result = route_after_tests(state)

    assert result == "retry"
    assert state["retry_count"] == 1


def test_route_after_tests_max_retries():
    """
    When retry_count hits 3, must force "open_pr"
    even if tests failed. This is the circuit breaker.
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["test_result"] = "failed"
    state["retry_count"] = 3   # already at max

    result = route_after_tests(state)

    # Must force open PR even though tests failed
    assert result == "open_pr"


def test_check_for_errors_no_error():
    """
    When error is None, must return "no_error".
    Pipeline continues normally.
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["error"] = None

    result = check_for_errors(state)

    assert result == "no_error"


def test_check_for_errors_has_error():
    """
    When error is set, must return "has_error".
    Pipeline stops — goes to END.
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["error"] = "GitHub API rate limit exceeded"

    result = check_for_errors(state)

    assert result == "has_error"


def test_workflow_compiles():
    """
    The graph must compile without raising any exceptions.
    This verifies all nodes are connected and no orphan nodes exist.
    """
    try:
        app = build_workflow()
        assert app is not None
    except Exception as e:
        pytest.fail(f"Workflow failed to compile: {e}")