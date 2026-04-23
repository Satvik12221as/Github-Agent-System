import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from state import AgentState
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
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
    attempt: int = 1,
    feedback: str = ""
) -> str:
    """
    Sends the plan and code context to LLM.
    Asks for a unified diff patch back.
    attempt parameter lets us adjust the prompt on retries.
    feedback parameter passes review/syntax errors from previous attempt.
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

    # Add feedback from previous attempt if available
    # This tells the LLM exactly what was wrong last time
    feedback_section = ""
    if feedback:
        feedback_section = f"""
PREVIOUS ATTEMPT FAILED WITH THIS ERROR:
{feedback}

Fix the above issue in this attempt.
"""

    prompt = f"""
You are a senior software engineer writing a code fix.

FIX PLAN:
{plan}

CURRENT CODE:
{formatted_code}

{feedback_section}

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

def review_patch(
    plan: str,
    patch: str,
    code_context: dict
) -> dict:
    """
    Asks the LLM to review the generated patch.
    Returns a dict with approved (bool) and feedback (str).
    """
    logger.info("Reviewing patch with LLM...")

    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n--- {filename} ---\n{content}\n"

    prompt = f"""
You are a senior software engineer doing a code review.

ORIGINAL FIX PLAN:
{plan}

GENERATED PATCH:
{patch}

ORIGINAL CODE:
{formatted_code}

Review this patch carefully and answer:

1. Does the patch actually implement what the plan describes?
2. Are there any syntax errors in the changed lines?
3. Are there any obvious logic errors?
4. Does the patch break any existing functionality?
5. Is the patch complete or does it miss something?

Return ONLY this JSON:
{{
    "approved": true or false,
    "confidence": "high" or "medium" or "low",
    "issues": ["issue 1", "issue 2"],
    "suggestion": "one sentence on what to improve if not approved"
}}

Return ONLY the JSON. No explanation. No markdown.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_llm_output(response.content)

    try:
        review = json.loads(raw)
        logger.info(
            f"Review result: approved={review.get('approved')} "
            f"confidence={review.get('confidence')}"
        )
        if review.get("issues"):
            logger.warning(f"Issues found: {review['issues']}")
        return review
    except json.JSONDecodeError:
        logger.warning("Could not parse review. Approving by default.")
        return {
            "approved":   True,
            "confidence": "low",
            "issues":     [],
            "suggestion": ""
        }
    

def code_writer_agent(state: AgentState) -> AgentState:
    logger.info("=== Code Writer Agent starting ===")
    state["steps"] += 1

    if not state.get("plan"):
        state["error"] = "Code Writer failed: no plan found in state"
        return state

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            patch = generate_patch(
                state["plan"],
                state["code_context"],
                attempt=attempt,
                feedback=state.get("last_review_feedback", "")
            )

            logger.info(
                f"Patch generated (attempt {attempt}). "
                f"Length: {len(patch)} chars"
            )

            # Layer 1 — format validation
            if not validate_patch(patch):
                logger.warning(
                    f"Attempt {attempt} invalid format. Retrying."
                )
                continue

            # Layer 2 — LLM code review
            review = review_patch(
                state["plan"],
                patch,
                state["code_context"]
            )

            if review.get("approved"):
                state["patch"]          = patch
                state["error"]          = None
                state["patch_confidence"] = review.get(
                    "confidence", "medium"
                )
                logger.info(
                    f"Patch approved on attempt {attempt} "
                    f"with {state['patch_confidence']} confidence"
                )
                return state
            else:
                logger.warning(
                    f"Attempt {attempt} rejected by review. "
                    f"Issues: {review.get('issues')}. "
                    f"Suggestion: {review.get('suggestion')}"
                )
                # Feed the feedback into the next attempt
                state["last_review_feedback"] = review.get(
                    "suggestion", ""
                )

        except Exception as e:
            logger.error(f"Attempt {attempt} error: {e}")
            if attempt == max_attempts:
                state["error"] = f"Code Writer failed: {str(e)}"
                return state

    # All attempts exhausted
    logger.error("All attempts failed review. Using best effort.")
    state["patch"] = patch
    state["error"] = "Warning: patch did not pass review"
    return state


import ast
import tempfile
import os


def apply_patch_locally(
    original_content: str,
    patch: str,
    filename: str
) -> str:
    """
    Applies the patch to the original content.
    Returns the resulting file content.
    Simple line-by-line application.
    """
    lines = patch.split("\n")
    result_lines = original_content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("@@"):
            # Parse hunk header @@ -old +new @@
            import re
            match = re.search(r"\+(\d+)", line)
            if match:
                new_start = int(match.group(1)) - 1
                # Apply changes from this hunk
                j = i + 1
                current_line = new_start

                while j < len(lines) and not lines[j].startswith("@@"):
                    hunk_line = lines[j]
                    if hunk_line.startswith("+") and not hunk_line.startswith("+++"):
                        result_lines.insert(
                            current_line,
                            hunk_line[1:]
                        )
                        current_line += 1
                    elif hunk_line.startswith("-") and not hunk_line.startswith("---"):
                        if current_line < len(result_lines):
                            result_lines.pop(current_line)
                    else:
                        current_line += 1
                    j += 1
        i += 1

    return "\n".join(result_lines)


def verify_syntax(
    patched_content: str,
    filename: str
) -> dict:
    """
    Checks if the patched file has valid Python syntax.
    Returns dict with valid (bool) and error (str).
    Only works for Python files.
    """
    if not filename.endswith(".py"):
        return {"valid": True, "error": None}

    try:
        ast.parse(patched_content)
        logger.info(f"Syntax valid: {filename}")
        return {"valid": True, "error": None}

    except SyntaxError as e:
        logger.warning(
            f"Syntax error in {filename}: "
            f"line {e.lineno} — {e.msg}"
        )
        return {
            "valid": False,
            "error": f"SyntaxError on line {e.lineno}: {e.msg}"
        }


def verify_patch_applies_cleanly(
    patch: str,
    code_context: dict
) -> dict:
    """
    Applies the patch to each affected file and checks
    the result has valid Python syntax.

    Returns:
    {
        "valid": True/False,
        "errors": ["file.py: SyntaxError on line 42"]
    }
    """
    errors = []

    # Find which files this patch affects
    import re
    affected_files = re.findall(r"--- a/(.*?)\n", patch)

    for filename in affected_files:
        if filename not in code_context:
            logger.warning(f"File not in context: {filename}")
            continue

        original = code_context[filename]
        patched  = apply_patch_locally(original, patch, filename)
        result   = verify_syntax(patched, filename)

        if not result["valid"]:
            errors.append(f"{filename}: {result['error']}")

    return {
        "valid":  len(errors) == 0,
        "errors": errors
    }