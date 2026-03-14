# tests/test_planner.py

import pytest
from unittest.mock import MagicMock, patch
from state import get_initial_state
from agents.planner import clean_llm_output, build_plan, planner_agent


def test_clean_llm_output_strips_fences():
    """Planner has its own clean function — verify it works."""
    raw = '```json\n{"complexity": "simple"}\n```'
    cleaned = clean_llm_output(raw)
    assert cleaned == '{"complexity": "simple"}'


def test_planner_sets_complexity_in_state():
    """
    Test that planner_agent correctly writes
    complexity into state from the plan data.
    We mock build_plan so we don't call the real Gemini API.
    """
    state = get_initial_state("https://github.com/user/repo/issues/1")
    state["issue_title"] = "Button not working"
    state["issue_body"] = "Submit button fails on mobile"
    state["code_context"] = {"button.py": "def render(): pass"}

    # Mock build_plan to return a controlled response
    # This means the test never calls Gemini — fast and free
    mock_plan = {
        "summary": "Fix the button render function",
        "steps": ["Find the bug", "Fix it", "Test it"],
        "affected_files": ["button.py"],
        "complexity": "simple",
        "risk": "low"
    }

    with patch("agents.planner.build_plan", return_value=mock_plan):
        updated_state = planner_agent(state)

    assert updated_state["complexity"] == "simple"
    assert updated_state["error"] is None
    assert updated_state["steps"] == 1


def test_planner_sets_complex_for_complex_issues():
    """Test that complex complexity gets written correctly too."""
    state = get_initial_state("https://github.com/user/repo/issues/2")
    state["issue_title"] = "Refactor entire auth system"
    state["issue_body"] = "Auth needs full rewrite across 5 files"
    state["code_context"] = {
        "auth.py": "def login(): pass",
        "middleware.py": "def check(): pass",
        "utils.py": "def helper(): pass"
    }

    mock_plan = {
        "summary": "Refactor auth across multiple files",
        "steps": ["Step 1", "Step 2", "Step 3"],
        "affected_files": ["auth.py", "middleware.py", "utils.py"],
        "complexity": "complex",
        "risk": "high"
    }

    with patch("agents.planner.build_plan", return_value=mock_plan):
        updated_state = planner_agent(state)

    assert updated_state["complexity"] == "complex"
    assert updated_state["steps"] == 1


def test_planner_handles_empty_code_context():
    """
    Test that planner doesn't crash when code_context is empty.
    This happens if Code Reader failed or found no files.
    """
    state = get_initial_state("https://github.com/user/repo/issues/3")
    state["issue_title"] = "Something is broken"
    state["issue_body"] = "It does not work"
    state["code_context"] = {}   # deliberately empty

    mock_plan = {
        "summary": "Fix the issue",
        "steps": ["Investigate", "Fix", "Test"],
        "affected_files": [],
        "complexity": "simple",
        "risk": "low"
    }

    with patch("agents.planner.build_plan", return_value=mock_plan):
        updated_state = planner_agent(state)

    # Should not crash — should complete with simple complexity
    assert updated_state["complexity"] == "simple"
    assert updated_state["error"] is None


def test_plan_string_contains_summary():
    """
    Test that the plan string written into state
    actually contains the summary from the plan data.
    """
    state = get_initial_state("https://github.com/user/repo/issues/4")
    state["issue_title"] = "Bug in login"
    state["issue_body"] = "Login fails for new users"
    state["code_context"] = {"auth.py": "def login(): pass"}

    mock_plan = {
        "summary": "Fix login validation for new users",
        "steps": ["Find validation bug", "Fix condition", "Add test"],
        "affected_files": ["auth.py"],
        "complexity": "simple",
        "risk": "low"
    }

    with patch("agents.planner.build_plan", return_value=mock_plan):
        updated_state = planner_agent(state)

    # The plan string must contain the summary
    assert "Fix login validation for new users" in updated_state["plan"]
    # The plan string must contain step numbers
    assert "1." in updated_state["plan"]