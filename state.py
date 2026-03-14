from typing import TypedDict, Optional
import re

class AgentState(TypedDict):
    # Input
    issue_url: str           # The GitHub issue URL we're fixing
    issue_title: str         # Title of the issue
    issue_body: str          # Full description of the issue

    # Agent outputs - filled in one by one as agents run
    code_context: dict       # Files the Code Reader found relevant
    plan: str                # The fix plan from Planner agent
    complexity: str          # 'simple' or 'complex' — controls routing
    patch: str               # The actual code fix from Code Writer
    tests: str               # pytest code from Test Writer
    test_result: str         # 'passed' or 'failed' + output
    pr_url: str              # Final pull request URL

    # Control fields
    error: Optional[str]     # Stores error message if something fails
    retry_count: int         # Tracks how many times we've retried
    steps: int               # Total steps taken — circuit breaker uses this.



def validate_github_url(url: str) -> bool:
    """
    Checks if the URL looks like a real GitHub issue URL.
    Pattern: https://github.com/owner/repo/issues/number
    """
    pattern = r"https://github\.com/[\w.-]+/[\w.-]+/issues/\d+"
    return bool(re.match(pattern, url))

def get_initial_state(issue_url: str) -> AgentState:
    """Factory function — creates a clean starting state for any run."""
    if not validate_github_url(issue_url):
        raise ValueError(
            f"Invalid GitHub issue URL: {issue_url}\n"
            f"Expected format: https://github.com/owner/repo/issues/42"
        )
    return AgentState(
        issue_url=issue_url,
        issue_title="",
        issue_body="",
        code_context={},
        plan="",
        complexity="simple",
        patch="",
        tests="",
        test_result="",
        pr_url="",
        error=None,
        retry_count=0,
        steps=0,
    )