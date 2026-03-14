import pytest
from state import validate_github_url
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

def test_valid_github_url():
    """Valid URL should pass validation."""
    url = "https://github.com/user/repo/issues/42"
    assert validate_github_url(url) == True

def test_invalid_github_url_no_issue_number():
    """URL without issue number should fail."""
    url = "https://github.com/user/repo"
    assert validate_github_url(url) == False

def test_invalid_github_url_random_string():
    """Random string should fail validation."""
    url = "not-a-url-at-all"
    assert validate_github_url(url) == False

def test_get_initial_state_raises_on_invalid_url():
    """get_initial_state() should raise ValueError on bad URL."""
    with pytest.raises(ValueError):
        get_initial_state("https://github.com/user/repo")