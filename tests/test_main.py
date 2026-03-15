# tests/test_main.py

import pytest
import sys
from unittest.mock import patch, MagicMock
from main import validate_args, print_run_summary


def test_validate_args_valid_url():
    """Valid GitHub issue URL should pass validation."""
    args = MagicMock()
    args.issue = "https://github.com/user/repo/issues/42"

    # Should not raise or call sys.exit
    try:
        validate_args(args)
    except SystemExit:
        pytest.fail("validate_args raised SystemExit on valid URL")


def test_validate_args_no_issue_number():
    """URL without issue number should exit."""
    args = MagicMock()
    args.issue = "https://github.com/user/repo"

    with pytest.raises(SystemExit):
        validate_args(args)


def test_validate_args_not_github():
    """Non-GitHub URL should exit."""
    args = MagicMock()
    args.issue = "https://gitlab.com/user/repo/issues/1"

    with pytest.raises(SystemExit):
        validate_args(args)


def test_validate_args_empty():
    """Empty issue should exit."""
    args = MagicMock()
    args.issue = None

    with pytest.raises(SystemExit):
        validate_args(args)


def test_print_run_summary_no_crash():
    """print_run_summary should run without errors."""
    from state import get_initial_state
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["issue_title"] = "Test issue"
    state["complexity"] = "simple"
    state["test_result"] = "passed"
    state["pr_url"] = "https://github.com/user/repo/pull/2"

    try:
        print_run_summary(state, 42.5)
    except Exception as e:
        pytest.fail(f"print_run_summary crashed: {e}")