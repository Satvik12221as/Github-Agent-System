# tests/test_pr_opener.py

import pytest
from unittest.mock import patch, MagicMock
from state import get_initial_state
from agents.pr_opener import (
    parse_issue_url,
    parse_patch,
    apply_patch_to_content,
    create_pr_description,
    open_pull_request_agent
)


# ─────────────────────────────────────────
# Tests for parse_issue_url()
# ─────────────────────────────────────────

def test_parse_issue_url_correct():
    """URL should be parsed into owner, repo, number."""
    url = "https://github.com/satvik/my-repo/issues/42"
    result = parse_issue_url(url)
    assert result["owner"] == "satvik"
    assert result["repo"] == "my-repo"
    assert result["number"] == 42


def test_parse_issue_url_different_values():
    """Test with different owner and repo names."""
    url = "https://github.com/microsoft/vscode/issues/100"
    result = parse_issue_url(url)
    assert result["owner"] == "microsoft"
    assert result["repo"] == "vscode"
    assert result["number"] == 100


# ─────────────────────────────────────────
# Tests for parse_patch()
# ─────────────────────────────────────────

def test_parse_patch_finds_filename():
    """Parser should extract the filename from the patch."""
    patch = """--- a/button.py
+++ b/button.py
@@ -1,2 +1,3 @@
 def render():
-    pass
+    return 'fixed'"""

    result = parse_patch(patch)
    assert len(result) == 1
    assert result[0]["filename"] == "button.py"


def test_parse_patch_empty():
    """Empty patch should return empty list."""
    result = parse_patch("")
    assert result == []


def test_parse_patch_has_hunks():
    """Parser should find at least one hunk."""
    patch = """--- a/auth.py
+++ b/auth.py
@@ -5,3 +5,4 @@
 def login():
-    pass
+    return True"""

    result = parse_patch(patch)
    assert len(result[0]["hunks"]) >= 1


# ─────────────────────────────────────────
# Tests for apply_patch_to_content()
# ─────────────────────────────────────────

def test_apply_patch_adds_line():
    """Patch should add new line to content."""
    original = "line1\nline2\nline3"
    hunks = [{
        "new_start": 2,
        "lines": [
            ("keep", "line1"),
            ("add",  "new_line"),
            ("keep", "line2"),
            ("keep", "line3")
        ]
    }]
    result = apply_patch_to_content(original, hunks)
    assert "new_line" in result


def test_apply_patch_removes_line():
    """Patch should remove specified line."""
    original = "line1\nremove_me\nline3"
    hunks = [{
        "new_start": 1,
        "lines": [
            ("keep",   "line1"),
            ("remove", "remove_me"),
            ("keep",   "line3")
        ]
    }]
    result = apply_patch_to_content(original, hunks)
    assert "remove_me" not in result


def test_apply_patch_empty_hunks():
    """Empty hunks should return original content unchanged."""
    original = "line1\nline2"
    result = apply_patch_to_content(original, [])
    assert result == original


# ─────────────────────────────────────────
# Tests for create_pr_description()
# ─────────────────────────────────────────

def test_create_pr_description_contains_plan():
    """PR description must contain the plan."""
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix the button handler"
    state["test_result"] = "passed"
    state["complexity"] = "simple"
    state["patch"] = "--- a/f.py\n+++ b/f.py"

    description = create_pr_description(state)

    assert "Fix the button handler" in description


def test_create_pr_description_shows_test_passed():
    """PR description must show test passed with checkmark."""
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix something"
    state["test_result"] = "passed"
    state["complexity"] = "simple"
    state["patch"] = "--- a/f.py"

    description = create_pr_description(state)

    assert "✅" in description


def test_create_pr_description_shows_test_failed():
    """PR description must show test failed with cross."""
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix something"
    state["test_result"] = "failed"
    state["complexity"] = "simple"
    state["patch"] = "--- a/f.py"

    description = create_pr_description(state)

    assert "❌" in description


# ─────────────────────────────────────────
# Tests for open_pull_request_agent()
# ─────────────────────────────────────────

def test_pr_opener_writes_pr_url_to_state():
    """
    When everything works, pr_url must be written to state.
    We mock all GitHub API calls.
    """
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["issue_title"] = "Button broken"
    state["plan"] = "Fix the button"
    state["patch"] = """--- a/button.py
+++ b/button.py
@@ -1,2 +1,3 @@
 def render():
-    pass
+    return 'fixed'"""
    state["test_result"] = "passed"
    state["complexity"] = "simple"
    state["code_context"] = {
        "button.py": "def render():\n    pass"
    }

    # Mock the entire GitHub interaction
    mock_repo = MagicMock()
    mock_repo.default_branch = "main"
    mock_repo.get_git_ref.return_value.object.sha = "abc123"
    mock_repo.get_contents.return_value.decoded_content = (
        b"def render():\n    pass"
    )
    mock_repo.get_contents.return_value.sha = "file_sha_123"
    mock_pr = MagicMock()
    mock_pr.html_url = "https://github.com/user/repo/pull/2"
    mock_repo.create_pull.return_value = mock_pr

    mock_github = MagicMock()
    mock_github.get_repo.return_value = mock_repo

    with patch(
        "agents.pr_opener.get_github_client",
        return_value=mock_github
    ):
        updated_state = open_pull_request_agent(state)

    assert updated_state["pr_url"] == (
        "https://github.com/user/repo/pull/2"
    )
    assert updated_state["error"] is None
    assert updated_state["steps"] == 1


def test_pr_opener_fails_without_patch():
    """If no patch in state, agent must set error."""
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["patch"] = ""

    updated_state = open_pull_request_agent(state)

    assert updated_state["error"] is not None
    assert "no patch" in updated_state["error"].lower()
    assert updated_state["pr_url"] == ""


def test_pr_opener_increments_steps():
    """Steps counter must increment by 1."""
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["issue_title"] = "Bug"
    state["plan"] = "Fix it"
    state["patch"] = """--- a/f.py
+++ b/f.py
@@ -1,1 +1,1 @@
-old
+new"""
    state["test_result"] = "passed"
    state["complexity"] = "simple"

    mock_repo = MagicMock()
    mock_repo.default_branch = "main"
    mock_repo.get_git_ref.return_value.object.sha = "abc123"
    mock_repo.get_contents.return_value.decoded_content = b"old"
    mock_repo.get_contents.return_value.sha = "sha123"
    mock_pr = MagicMock()
    mock_pr.html_url = "https://github.com/user/repo/pull/3"
    mock_repo.create_pull.return_value = mock_pr

    mock_github = MagicMock()
    mock_github.get_repo.return_value = mock_repo

    with patch(
        "agents.pr_opener.get_github_client",
        return_value=mock_github
    ):
        updated_state = open_pull_request_agent(state)

    assert updated_state["steps"] == 1