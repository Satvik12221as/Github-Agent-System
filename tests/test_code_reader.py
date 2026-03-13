# tests/test_code_reader.py

import pytest
from unittest.mock import MagicMock, patch
from state import get_initial_state
from agents.code_reader import (
    fetch_issue_details,
    clean_llm_output,
    fetch_file_contents
)


def test_clean_llm_output_strips_fences():
    """Test that markdown fences get removed correctly."""
    raw = '```json\n["auth.py", "utils.py"]\n```'
    cleaned = clean_llm_output(raw)
    assert cleaned == '["auth.py", "utils.py"]'


def test_clean_llm_output_plain_json():
    """Test that plain JSON passes through unchanged."""
    raw = '["auth.py", "utils.py"]'
    cleaned = clean_llm_output(raw)
    assert cleaned == '["auth.py", "utils.py"]'


def test_initial_state_structure():
    """Test that get_initial_state gives us the right shape."""
    state = get_initial_state("https://github.com/user/repo/issues/1")
    assert state["issue_url"] == "https://github.com/user/repo/issues/1"
    assert state["retry_count"] == 0
    assert state["steps"] == 0
    assert state["code_context"] == {}
    assert state["error"] is None