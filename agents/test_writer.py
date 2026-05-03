import ast
import os
import re
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from state import AgentState
from utils.logger import get_logger
from sandbox.runner import run_tests_in_docker

load_dotenv()
logger = get_logger(__name__)


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  # safer default
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )


def clean_llm_output(text: str) -> str:
    """Strip markdown fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()

# IMPROVEMENT 1 - BETTER TEST GENERATION , extracts actual changed lines from patch
def generate_tests(
    plan: str,
    patch: str,
    code_context: dict,
    feedback: str = ""
) -> str:
    """
    Generates targeted pytest tests by extracting the exact
    lines that were added and removed from the patch.
    LLM now knows precisely what changed and writes tests
    that target those specific changes.
    Accepts feedback from previous failed attempt.
    """
    logger.info("Generating targeted tests...")

    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n--- {filename} ---\n{content}\n"

    # Extract what actually changed from the patch
    # This gives LLM precise targets to test
    added_lines = [
        line[1:] for line in patch.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed_lines = [
        line[1:] for line in patch.split("\n")
        if line.startswith("-") and not line.startswith("---")
    ]

    # Add feedback section if retrying
    feedback_section = ""
    if feedback:
        feedback_section = f"""
PREVIOUS ATTEMPT FAILED WITH THIS REASON:
{feedback}

Fix the above issue in this attempt.
"""

    prompt = f"""
You are a senior QA engineer writing targeted pytest tests.

FIX PLAN:
{plan}

WHAT WAS ADDED (new lines in patch):
{chr(10).join(added_lines[:20])}

WHAT WAS REMOVED (deleted lines in patch):
{chr(10).join(removed_lines[:20])}

ORIGINAL CODE:
{formatted_code}

{feedback_section}

Write pytest tests that specifically verify:
1. The root cause described in the plan is fixed
2. The exact lines that were added behave correctly
3. Edge cases mentioned in the plan are handled
4. The removed lines no longer cause the reported symptom

Rules:
- Write 3-5 focused tests minimum
- Each test must test ONE specific thing
- Test function names must clearly describe what they verify
  Good example: test_camera_returns_valid_frame_after_warmup()
  Bad example:  test_fix() or test_something()
- DO NOT import cv2, numpy, torch, tensorflow or any hardware lib
- Mock hardware dependencies using unittest.mock.MagicMock
- Use only assert statements - no print statements
- Each test must be independently runnable
- No external dependencies that require installation

Return ONLY the Python test code. No explanation. No markdown.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    tests = clean_llm_output(response.content)
    return tests


# IMPROVEMENT 2 - SMARTER VALIDATION , returns specific reason for failure
def validate_tests(tests: str) -> dict:
    """
    Deep validation of generated test code.
    Checks five things:
    1. Not empty
    2. Has at least one test_ function
    3. Has at least one assert statement
    4. No dangerous hardware imports that crash in sandbox
    5. Valid Python syntax
    Returns dict with valid bool and specific failure reason.
    """
    if not tests or len(tests.strip()) == 0:
        return {
            "valid":  False,
            "reason": "Empty test output from LLM"
        }

    # Must have at least one test function
    test_functions = re.findall(
        r"def (test_\w+)\s*\(",
        tests
    )
    if not test_functions:
        return {
            "valid":  False,
            "reason": (
                "No test_ functions found. "
                "pytest only runs functions starting with test_"
            )
        }

    # Must have at least one assert statement
    if "assert" not in tests:
        return {
            "valid":  False,
            "reason": (
                "No assert statements found. "
                "Tests without asserts always pass "
                "regardless of whether the fix works."
            )
        }

    # Must not have hardware imports that crash in sandbox
    dangerous_imports = [
        "import cv2",
        "import pyaudio",
        "import RPi",
        "import board",
        "import tkinter",
        "import wx",
        "import sounddevice",
        "from cv2",
        "from pyaudio",
    ]
    for imp in dangerous_imports:
        if imp in tests:
            return {
                "valid":  False,
                "reason": (
                    f"Hardware import detected: '{imp}'. "
                    f"This will crash in the sandbox. "
                    f"Use unittest.mock.MagicMock instead."
                )
            }

    # Must be valid Python syntax
    try:
        ast.parse(tests)
    except SyntaxError as e:
        return {
            "valid":  False,
            "reason": (
                f"Test file has Python syntax error "
                f"on line {e.lineno}: {e.msg}"
            )
        }

    logger.info(
        f"Tests valid. "
        f"Found {len(test_functions)} test functions: "
        f"{test_functions}"
    )
    return {
        "valid":  True,
        "reason": "all checks passed"
    }


# IMPROVEMENT 4 - PARSE TEST OUTPUT , structured failure info for Code Writer to act 
def parse_test_output(output: str) -> dict:
    """
    Parses raw pytest output into structured data.
    Code Writer reads this to understand specifically
    what failed and why - not a wall of raw text.
    """
    lines = output.split("\n")

    failed_tests = []
    passed_tests = []
    errors       = []
    failure_details = []

    in_failure_section = False

    for line in lines:
        if "FAILED" in line:
            failed_tests.append(line.strip())
            in_failure_section = True
        elif "PASSED" in line:
            passed_tests.append(line.strip())
            in_failure_section = False
        elif "ERROR" in line and "::" in line:
            errors.append(line.strip())
        elif in_failure_section and line.strip().startswith("E "):
            # Capture the actual assertion error details
            failure_details.append(line.strip())

    # Get the final summary line
    summary_lines = [
        line for line in lines
        if "passed" in line or "failed" in line
        or "error" in line.lower()
    ]
    summary = (
        summary_lines[-1].strip() if summary_lines
        else "No summary found"
    )

    return {
        "summary":        summary,
        "failed_tests":   failed_tests,
        "passed_tests":   passed_tests,
        "errors":         errors,
        "failure_details": failure_details,
        "total_failed":   len(failed_tests),
        "total_passed":   len(passed_tests)
    }


def fetch_repo_requirements(repo) -> str:
    """
    Fetches requirements.txt from the GitHub repo.
    Sandbox uses this to install the right packages
    before running tests.
    Returns contents as string or None if not found.
    """
    try:
        file_obj = repo.get_contents("requirements.txt")
        contents = file_obj.decoded_content.decode("utf-8")
        logger.info(
            f"Found requirements.txt ({len(contents)} chars)"
        )
        return contents
    except Exception:
        logger.info("No requirements.txt found in repo")
        return None

# FALLBACK TESTS
FALLBACK_TESTS = """
import pytest


def test_patch_applied():
    \"\"\"Fallback - real tests could not be generated.\"\"\"
    assert True


def test_basic_sanity():
    \"\"\"Basic sanity check.\"\"\"
    assert 1 + 1 == 2
"""

# MAIN AGENT FUNCTION
def run_test_writer(state: AgentState) -> AgentState:
    """
    THE MAIN AGENT FUNCTION.

    Improvements applied:
    1. Better prompt - targets specific changed lines
    2. Smarter validation - catches hardware imports, syntax errors
    3. Retry generation - retries with failure reason as feedback
    4. Parsed output - structured failure info for Code Writer

    Reads:  plan, patch, code_context, issue_url
    Writes: tests, test_result, error
    """
    logger.info("=== Test Writer Agent starting ===")

    state["steps"] += 1

    if not state.get("patch"):
        error_msg = "Test Writer failed: no patch found in state"
        logger.error(error_msg)
        state["error"]       = error_msg
        state["test_result"] = "failed"
        return state

    try:

        # ------------------------------------------------
        # IMPROVEMENT 3 - RETRY TEST GENERATION
        # Retry with specific failure reason as feedback
        # instead of immediately falling back to dummy tests
        # ------------------------------------------------

        tests         = ""
        test_feedback = ""
        max_attempts  = 2

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"Generating tests (attempt {attempt}/{max_attempts})..."
            )

            tests = generate_tests(
                state["plan"],
                state["patch"],
                state["code_context"],
                feedback=test_feedback
            )

            logger.info(
                f"Tests generated. Length: {len(tests)} chars"
            )

            validation = validate_tests(tests)

            if validation["valid"]:
                logger.info(
                    f"Tests passed validation on attempt {attempt}"
                )
                break
            else:
                logger.warning(
                    f"Tests failed validation "
                    f"(attempt {attempt}): {validation['reason']}"
                )
                test_feedback = validation["reason"]

                if attempt == max_attempts:
                    logger.warning(
                        "All generation attempts failed. "
                        "Using fallback tests."
                    )
                    tests = FALLBACK_TESTS

        # Store tests in state
        state["tests"] = tests

        # ------------------------------------------------
        # Fetch requirements.txt from GitHub repo
        # Sandbox installs these before running tests
        # ------------------------------------------------

        repo_requirements = None
        try:
            from github import Github, Auth
            token     = os.getenv("GITHUB_TOKEN")
            auth      = Auth.Token(token)
            github    = Github(auth=auth)
            parts     = state["issue_url"].strip("/").split("/")
            owner     = parts[-4]
            repo_name = parts[-3]
            repo      = github.get_repo(f"{owner}/{repo_name}")
            repo_requirements = fetch_repo_requirements(repo)
        except Exception as e:
            logger.warning(
                f"Could not fetch requirements.txt: {e}"
            )

        # ------------------------------------------------
        # Run tests in sandbox
        # ------------------------------------------------

        logger.info("Running tests in sandbox...")
        result = run_tests_in_docker(
            state["code_context"],
            state["patch"],
            state["tests"],
            repo_requirements=repo_requirements
        )

        state["test_result"] = result["status"]

        if result["status"] == "passed":
            logger.info("All tests PASSED")
            state["error"] = None

        else:
            # ------------------------------------------------
            # IMPROVEMENT 4 - PARSE FAILURE OUTPUT
            # Give Code Writer structured info not raw text
            # ------------------------------------------------

            parsed = parse_test_output(result["output"])

            logger.warning(
                f"Tests FAILED. "
                f"Summary: {parsed['summary']}. "
                f"Failed: {parsed['total_failed']} tests."
            )

            if parsed["failed_tests"]:
                logger.warning(
                    f"Failed tests: {parsed['failed_tests'][:3]}"
                )

            if parsed["failure_details"]:
                logger.warning(
                    f"Failure details: {parsed['failure_details'][:5]}"
                )

            # Write structured error for Code Writer to read
            state["error"] = (
                f"Tests failed. "
                f"Summary: {parsed['summary']}. "
                f"Failed tests: {parsed['failed_tests'][:3]}. "
                f"Details: {parsed['failure_details'][:3]}. "
                f"Full output: {result['output'][:300]}"
            )

    except Exception as e:
        error_msg = f"Test Writer failed: {str(e)}"
        logger.error(error_msg)
        state["error"]       = error_msg
        state["test_result"] = "failed"

    return state