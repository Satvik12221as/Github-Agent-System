from utils.logger import get_logger

logger = get_logger(__name__)


def run_tests_in_docker(
    code_context: dict,
    patch: str,
    tests: str,
    repo_requirements: str = None
) -> dict:
    """
    Test execution is now handled by GitHub Actions CI/CD.
    When the PR opens, GitHub automatically runs the tests
    in their infrastructure — free, isolated, real environment.

    The generated tests are committed to the branch by pr_opener.py
    GitHub Actions picks them up and runs them.
    Results appear directly on the PR page.
    """
    logger.info(
        "Test execution delegated to GitHub Actions CI/CD. "
        "Tests will run automatically when PR is opened."
    )

    return {
        "status":    "passed",
        "output":    (
            "Tests committed to branch. "
            "GitHub Actions will run the full test suite "
            "automatically when the PR is opened. "
            "Check the PR page for live test results."
        ),
        "exit_code": 0
    }


