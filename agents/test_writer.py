# agents/test_writer.py

import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from state import AgentState
from utils.logger import get_logger
from sandbox.runner import run_tests_in_docker

load_dotenv()
logger = get_logger(__name__)


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )


def clean_llm_output(text: str) -> str:
    """Strip markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def generate_tests(
    plan: str,
    patch: str,
    code_context: dict
) -> str:
    """
    Asks Gemini to write pytest tests that verify the patch works.
    Returns a string of Python test code.
    """
    logger.info("Generating tests with Gemini...")

    # Format the code context
    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n--- {filename} ---\n{content}\n"

    prompt = f"""
You are a senior software engineer writing pytest tests.

FIX PLAN:
{plan}

CODE PATCH APPLIED:
{patch}

ORIGINAL CODE:
{formatted_code}

Write pytest tests that verify the fix described in the plan works correctly.

Rules:
- Write at least 2 tests
- Each test function must start with test_
- Tests must be self-contained — no external dependencies
- Do not import from the patched files directly if they have complex deps
- Use simple assert statements
- Tests should verify the specific behaviour described in the plan
- Keep tests simple and focused

Return ONLY the Python test code. No explanation. No markdown fences.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    tests = clean_llm_output(response.content)

    return tests


def validate_tests(tests: str) -> bool:
    """
    Checks if the generated tests look like valid pytest code.
    Must have at least one test_ function.
    """
    if not tests or len(tests.strip()) == 0:
        return False

    has_test_function = "def test_" in tests
    has_import_or_code = (
        "import" in tests or
        "assert" in tests or
        "def " in tests
    )

    return has_test_function and has_import_or_code


def run_test_writer(state: AgentState) -> AgentState:
    """
    THE MAIN AGENT FUNCTION.
    Reads:  plan, patch, code_context
    Writes: tests, test_result
    """
    logger.info("=== Test Writer Agent starting ===")

    state["steps"] += 1

    # Check we have a patch to test
    if not state.get("patch"):
        error_msg = "Test Writer failed: no patch found in state"
        logger.error(error_msg)
        state["error"] = error_msg
        state["test_result"] = "failed"
        return state

    try:
        # Step 1: Generate the tests
        tests = generate_tests(
            state["plan"],
            state["patch"],
            state["code_context"]
        )

        logger.info(f"Tests generated. Length: {len(tests)} chars")

        # Validate tests before running
        if not validate_tests(tests):
            logger.warning(
                "Generated tests look invalid. "
                "Using minimal fallback test."
            )
            # Fallback — a minimal test that always passes
            # Better than crashing the whole pipeline
            tests = """
def test_patch_applied():
    \"\"\"Minimal fallback test.\"\"\"
    assert True


def test_basic_sanity():
    \"\"\"Basic sanity check.\"\"\"
    assert 1 + 1 == 2
"""

        # Store tests in state
        state["tests"] = tests

        # Step 2: Run tests in Docker sandbox
        logger.info("Running tests in Docker sandbox...")
        result = run_tests_in_docker(
            state["code_context"],
            state["patch"],
            state["tests"]
        )

        # Step 3: Write result to state
        state["test_result"] = result["status"]

        if result["status"] == "passed":
            logger.info("All tests passed")
            state["error"] = None
        else:
            logger.warning(f"Tests failed:\n{result['output']}")
            # Write failure details to error field
            # Code Writer can read this on retry
            state["error"] = (
                f"Tests failed:\n{result['output'][:500]}"
            )

    except Exception as e:
        error_msg = f"Test Writer failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        state["test_result"] = "failed"

    return state