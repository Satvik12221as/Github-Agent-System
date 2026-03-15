# tests/test_code_writer.py

import pytest
from unittest.mock import patch, MagicMock
from state import get_initial_state
from agents.code_writer import (
    validate_patch,
    clean_llm_output,
    generate_patch,
    code_writer_agent
)


# ─────────────────────────────────────────
# Tests for validate_patch()
# ─────────────────────────────────────────

def test_validate_patch_valid():
    """A properly formatted unified diff should pass."""
    valid_patch = """--- a/button.py
+++ b/button.py
@@ -10,6 +10,8 @@
 def render():
-    onclick = "submit()"
+    onclick = "handleClick()"
+    prevent_default()
     return onclick"""

    assert validate_patch(valid_patch) == True


def test_validate_patch_empty():
    """Empty string should fail validation."""
    assert validate_patch("") == False


def test_validate_patch_no_hunk_header():
    """Patch without @@ hunk header should fail."""
    bad_patch = """--- a/button.py
+++ b/button.py
-old line
+new line"""

    assert validate_patch(bad_patch) == False


def test_validate_patch_no_changes():
    """Patch with headers but no actual changes should fail."""
    bad_patch = """--- a/button.py
+++ b/button.py
@@ -10,3 +10,3 @@
 just context
 no changes here
 still no changes"""

    assert validate_patch(bad_patch) == False


def test_validate_patch_plain_text():
    """Plain text explanation is not a valid patch."""
    not_a_patch = (
        "You should change the onclick handler "
        "to use handleClick() instead of submit()"
    )
    assert validate_patch(not_a_patch) == False


# ─────────────────────────────────────────
# Tests for clean_llm_output()
# ─────────────────────────────────────────

def test_clean_llm_output_strips_diff_fences():
    """Strips ```diff fences from patch output."""
    fenced = "```diff\n--- a/f.py\n+++ b/f.py\n```"
    cleaned = clean_llm_output(fenced)
    assert cleaned == "--- a/f.py\n+++ b/f.py"


def test_clean_llm_output_plain_patch():
    """Plain patch passes through unchanged."""
    plain = "--- a/f.py\n+++ b/f.py"
    assert clean_llm_output(plain) == plain


# ─────────────────────────────────────────
# Tests for code_writer_agent()
# ─────────────────────────────────────────

def test_code_writer_writes_patch_to_state():
    """
    When generate_patch returns a valid patch,
    code_writer_agent should write it to state["patch"].
    """
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix the onclick handler"
    state["code_context"] = {
        "button.py": "def render():\n    pass"
    }

    valid_patch = """--- a/button.py
+++ b/button.py
@@ -1,2 +1,3 @@
 def render():
-    pass
+    onclick = "handleClick()"
+    return onclick"""

    with patch(
        "agents.code_writer.generate_patch",
        return_value=valid_patch
    ):
        updated_state = code_writer_agent(state)

    assert updated_state["patch"] == valid_patch
    assert updated_state["error"] is None
    assert updated_state["steps"] == 1


def test_code_writer_fails_without_plan():
    """
    If state has no plan, agent should set error
    and return without calling Gemini.
    """
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = ""  # no plan
    state["code_context"] = {"button.py": "def render(): pass"}

    updated_state = code_writer_agent(state)

    assert updated_state["error"] is not None
    assert "no plan" in updated_state["error"].lower()
    assert updated_state["patch"] == ""


def test_code_writer_retries_on_invalid_patch():
    """
    When first attempt returns invalid patch but
    second attempt returns valid patch,
    agent should succeed on attempt 2.
    """
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix the handler"
    state["code_context"] = {"button.py": "def render(): pass"}

    invalid_patch = "this is not a valid patch at all"
    valid_patch = """--- a/button.py
+++ b/button.py
@@ -1,2 +1,3 @@
 def render():
-    pass
+    onclick = "handleClick()"
+    return onclick"""

    # First call returns invalid, second call returns valid
    with patch(
        "agents.code_writer.generate_patch",
        side_effect=[invalid_patch, valid_patch]
    ):
        updated_state = code_writer_agent(state)

    assert updated_state["patch"] == valid_patch
    assert updated_state["error"] is None


def test_code_writer_increments_steps():
    """steps counter must increment by 1."""
    state = get_initial_state(
        "https://github.com/user/repo/issues/1"
    )
    state["plan"] = "Fix something"
    state["code_context"] = {"f.py": "pass"}

    valid_patch = """--- a/f.py
+++ b/f.py
@@ -1,1 +1,2 @@
-pass
+def fixed(): pass"""

    with patch(
        "agents.code_writer.generate_patch",
        return_value=valid_patch
    ):
        updated_state = code_writer_agent(state)

    assert updated_state["steps"] == 1