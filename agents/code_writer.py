import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from state import AgentState
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
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


def validate_patch(patch: str) -> bool:
    """
    Checks if the patch looks like a valid unified diff.
    A valid patch must have:
    - At least one line starting with ---
    - At least one line starting with +++
    - At least one line starting with @@ (hunk header)
    - At least one line starting with + or - (actual change)

    Returns True if valid, False if not.
    """
    if not patch or len(patch.strip()) == 0:
        logger.warning("Patch is empty")
        return False

    has_old_file = any(
        line.startswith("---") for line in patch.split("\n")
    )
    has_new_file = any(
        line.startswith("+++") for line in patch.split("\n")
    )
    has_hunk = any(
        line.startswith("@@") for line in patch.split("\n")
    )
    has_changes = any(
        line.startswith("+") or line.startswith("-")
        for line in patch.split("\n")
        if not line.startswith("---") and not line.startswith("+++")
    )

    is_valid = has_old_file and has_new_file and has_hunk and has_changes

    if not is_valid:
        logger.warning(
            f"Invalid patch. "
            f"has_old={has_old_file} "
            f"has_new={has_new_file} "
            f"has_hunk={has_hunk} "
            f"has_changes={has_changes}"
        )

    return is_valid


def generate_patch(
    plan: str,
    code_context: dict,
    attempt: int = 1
) -> str:
    """
    Sends the plan and code context to Gemini.
    Asks for a unified diff patch back.
    attempt parameter lets us adjust the prompt on retries.
    """
    logger.info(f"Generating patch (attempt {attempt})...")

    # Format code context into readable text
    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n--- {filename} ---\n{content}\n"

    # On retries we make the instruction stricter
    strictness = ""
    if attempt == 2:
        strictness = "\nIMPORTANT: Your previous response was not a valid unified diff. Make sure to include --- and +++ headers and @@ hunk markers."
    elif attempt >= 3:
        strictness = "\nCRITICAL: Return ONLY a unified diff. Nothing else. No explanation. Start with --- immediately."

    prompt = f"""
You are a senior software engineer writing a code fix.

FIX PLAN:
{plan}

CURRENT CODE:
{formatted_code}

Write a unified diff patch that implements the fix described in the plan.

The patch must follow this exact format:
--- a/filename.py
+++ b/filename.py
@@ -line,count +line,count @@
 context line (unchanged)
-removed line
+added line
 context line (unchanged)

Rules:
- Only change what is necessary to fix the issue
- Keep the patch minimal and focused
- Include 3 lines of context around each change
- Do not change unrelated code
{strictness}

Return ONLY the unified diff patch. No explanation. No markdown fences.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    patch = clean_llm_output(response.content)

    return patch


def code_writer_agent(state: AgentState) -> AgentState:
    """
    THE MAIN AGENT FUNCTION.
    Reads:  plan, code_context
    Writes: patch
    Retries up to 3 times if patch is invalid.
    """
    logger.info("=== Code Writer Agent starting ===")

    state["steps"] += 1

    # Check we have what we need
    if not state.get("plan"):
        error_msg = "Code Writer failed: no plan found in state"
        logger.error(error_msg)
        state["error"] = error_msg
        return state

    if not state.get("code_context"):
        logger.warning(
            "No code context found. "
            "Writing patch from plan only."
        )

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            # Generate the patch
            patch = generate_patch(
                state["plan"],
                state["code_context"],
                attempt=attempt
            )

            logger.info(
                f"Patch generated "
                f"(attempt {attempt}). "
                f"Length: {len(patch)} chars"
            )
            logger.debug(f"Patch content:\n{patch}")

            # Validate the patch
            if validate_patch(patch):
                # Valid patch — write to state and done
                state["patch"] = patch
                state["error"] = None
                logger.info(
                    f"Valid patch found on attempt {attempt}"
                )
                return state
            else:
                logger.warning(
                    f"Attempt {attempt} produced invalid patch. "
                    f"Retrying..."
                )

        except Exception as e:
            logger.error(
                f"Attempt {attempt} failed with error: {e}"
            )
            if attempt == max_attempts:
                state["error"] = (
                    f"Code Writer failed after "
                    f"{max_attempts} attempts: {str(e)}"
                )
                return state

    # All attempts exhausted — store best effort patch anyway
    logger.error(
        "All attempts produced invalid patches. "
        "Storing last attempt as best effort."
    )
    state["patch"] = patch
    state["error"] = (
        "Warning: patch may be invalid after "
        f"{max_attempts} attempts"
    )

    return state