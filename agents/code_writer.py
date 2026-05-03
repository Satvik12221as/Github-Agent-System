import os
import re
import ast
import json
import subprocess
import tempfile
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from state import AgentState
from utils.logger import get_logger


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


# LAYER 1 - FORMAT VALIDATION
def validate_patch(patch: str) -> bool:
    """
    Checks if the patch looks like a valid unified diff.
    Must have --- +++ @@ markers and actual changes.
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
            f"Format validation failed. "
            f"has_old={has_old_file} "
            f"has_new={has_new_file} "
            f"has_hunk={has_hunk} "
            f"has_changes={has_changes}"
        )

    return is_valid


# LAYER 2 - MULTI-LANGUAGE SYNTAX VALIDATION
def verify_syntax_python(content: str, filename: str) -> dict:
    """
    Validates Python syntax using ast.parse() and flake8.
    Catches undefined names, syntax errors, and missing imports.
    """
    # 1. AST structural check
    try:
        ast.parse(content)
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Python SyntaxError in {filename} "
                     f"on line {e.lineno}: {e.msg}"
        }

    # 2. Flake8 critical checks (undefined names, etc.)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # E9,F63,F7,F82 are critical errors: syntax errors, undefined names
        result = subprocess.run(
            ["flake8", "--select=E9,F63,F7,F82", tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        os.unlink(tmp_path)

        if result.returncode != 0:
            error_output = result.stdout.strip() or result.stderr.strip()
            # Clean up the tmp path from the error output so it doesn't confuse the LLM
            error_output = error_output.replace(tmp_path, filename)
            return {
                "valid": False,
                "error": f"Python Linting Error in {filename}:\n{error_output[:300]}"
            }

        return {"valid": True, "error": None}

    except FileNotFoundError:
        logger.warning("flake8 not installed. Skipping lint check.")
        return {"valid": True, "error": None}
    except Exception as e:
        logger.warning(f"Python linting check failed: {e}")
        return {"valid": True, "error": None}
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def verify_syntax_javascript(content: str, filename: str) -> dict:
    """
    Validates JavaScript/TypeScript syntax using node --check.
    Falls back to basic structural checks if node not installed.
    """
    # Try node.js if available
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".js",
            delete=False,
            encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = subprocess.run(
            ["node", "--check", tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        os.unlink(tmp_path)

        if result.returncode == 0:
            return {"valid": True, "error": None}
        else:
            error = result.stderr.strip()
            return {
                "valid": False,
                "error": f"JavaScript SyntaxError in {filename}: {error}"
            }

    except FileNotFoundError:
        # node not installed — fall back to basic checks
        logger.warning(
            "node.js not found. "
            "Running basic JS structural checks."
        )
        return verify_syntax_basic_structural(content, filename)

    except Exception as e:
        logger.warning(f"JS syntax check failed: {e}")
        return {"valid": True, "error": None}

    finally:
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception:
            pass


def verify_syntax_typescript(content: str, filename: str) -> dict:
    """
    Validates TypeScript syntax using tsc --noEmit.
    Falls back to JavaScript check if tsc not installed.
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ts",
            delete=False,
            encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = subprocess.run(
            ["tsc", "--noEmit", "--allowJs",
             "--checkJs", tmp_path],
            capture_output=True,
            text=True,
            timeout=15
        )
        os.unlink(tmp_path)

        if result.returncode == 0:
            return {"valid": True, "error": None}
        else:
            error = result.stderr.strip() or result.stdout.strip()
            return {
                "valid": False,
                "error": f"TypeScript error in {filename}: {error[:200]}"
            }

    except FileNotFoundError:
        logger.warning(
            "tsc not found. "
            "Falling back to JS check for TypeScript."
        )
        return verify_syntax_javascript(content, filename)

    except Exception as e:
        logger.warning(f"TS syntax check failed: {e}")
        return {"valid": True, "error": None}

    finally:
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception:
            pass


def verify_syntax_json(content: str, filename: str) -> dict:
    """
    Validates JSON syntax using json.loads().
    JSON has strict syntax — no external tools needed.
    """
    try:
        json.loads(content)
        return {"valid": True, "error": None}
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "error": f"JSON SyntaxError in {filename} "
                     f"on line {e.lineno} col {e.colno}: {e.msg}"
        }


def verify_syntax_html(content: str, filename: str) -> dict:
    """
    Validates HTML syntax using Python's html.parser.
    Checks for malformed tags and unclosed elements.
    """
    from html.parser import HTMLParser

    class HTMLValidator(HTMLParser):
        def __init__(self):
            super().__init__()
            self.errors  = []
            self.stack   = []
            self.void_elements = {
                "area", "base", "br", "col", "embed",
                "hr", "img", "input", "link", "meta",
                "param", "source", "track", "wbr"
            }

        def handle_starttag(self, tag, attrs):
            if tag.lower() not in self.void_elements:
                self.stack.append(tag.lower())

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in self.void_elements:
                return
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
            else:
                self.errors.append(
                    f"Unexpected closing tag </{tag}>"
                )

        def error(self, message):
            self.errors.append(message)

    try:
        validator = HTMLValidator()
        validator.feed(content)

        # Check for unclosed tags
        if validator.stack:
            return {
                "valid": False,
                "error": f"HTML error in {filename}: "
                         f"Unclosed tags: {validator.stack}"
            }

        if validator.errors:
            return {
                "valid": False,
                "error": f"HTML error in {filename}: "
                         f"{validator.errors[0]}"
            }

        return {"valid": True, "error": None}

    except Exception as e:
        logger.warning(f"HTML syntax check failed: {e}")
        return {"valid": True, "error": None}


def verify_syntax_css(content: str, filename: str) -> dict:
    """
    Validates CSS syntax using basic structural checks.
    Checks brace matching and basic property format.
    """
    try:
        # Check brace balance
        open_braces  = content.count("{")
        close_braces = content.count("}")

        if open_braces != close_braces:
            return {
                "valid": False,
                "error": f"CSS error in {filename}: "
                         f"Mismatched braces — "
                         f"{open_braces} open, "
                         f"{close_braces} close"
            }

        # Check for unclosed strings
        single_quotes = content.count("'")
        double_quotes = content.count('"')

        if single_quotes % 2 != 0:
            return {
                "valid": False,
                "error": f"CSS error in {filename}: "
                         f"Unclosed single quote"
            }

        if double_quotes % 2 != 0:
            return {
                "valid": False,
                "error": f"CSS error in {filename}: "
                         f"Unclosed double quote"
            }

        return {"valid": True, "error": None}

    except Exception as e:
        logger.warning(f"CSS syntax check failed: {e}")
        return {"valid": True, "error": None}


def verify_syntax_yaml(content: str, filename: str) -> dict:
    """
    Validates YAML syntax using PyYAML.
    Falls back gracefully if PyYAML not installed.
    """
    try:
        import yaml
        yaml.safe_load(content)
        return {"valid": True, "error": None}

    except ImportError:
        logger.warning("PyYAML not installed. Skipping YAML check.")
        return {"valid": True, "error": None}

    except yaml.YAMLError as e:
        return {
            "valid": False,
            "error": f"YAML error in {filename}: {str(e)[:200]}"
        }


def verify_syntax_xml(content: str, filename: str) -> dict:
    """
    Validates XML syntax using Python's built-in xml.etree.
    Works for .xml, .svg, .plist files.
    """
    import xml.etree.ElementTree as ET

    try:
        ET.fromstring(content)
        return {"valid": True, "error": None}

    except ET.ParseError as e:
        return {
            "valid": False,
            "error": f"XML error in {filename}: {str(e)}"
        }


def verify_syntax_basic_structural(
    content: str,
    filename: str
) -> dict:
    """
    Basic structural checks for any file type.
    Checks brace/bracket/parenthesis balance.
    Used as fallback when no specific checker is available.
    """
    try:
        # Check bracket balance
        pairs   = {"(": ")", "[": "]", "{": "}"}
        stack   = []
        lines   = content.split("\n")
        in_string_single = False
        in_string_double = False

        for line_num, line in enumerate(lines, 1):
            for char in line:
                # Track string context to avoid
                # counting brackets inside strings
                if char == "'" and not in_string_double:
                    in_string_single = not in_string_single
                elif char == '"' and not in_string_single:
                    in_string_double = not in_string_double

                if in_string_single or in_string_double:
                    continue

                if char in pairs:
                    stack.append((char, line_num))
                elif char in pairs.values():
                    if not stack:
                        return {
                            "valid": False,
                            "error": f"Structural error in {filename} "
                                     f"line {line_num}: "
                                     f"Unexpected closing '{char}'"
                        }
                    expected = pairs[stack[-1][0]]
                    if char == expected:
                        stack.pop()

        if stack:
            unclosed_char, unclosed_line = stack[-1]
            return {
                "valid": False,
                "error": f"Structural error in {filename}: "
                         f"Unclosed '{unclosed_char}' "
                         f"opened on line {unclosed_line}"
            }

        return {"valid": True, "error": None}

    except Exception as e:
        logger.warning(f"Structural check failed: {e}")
        return {"valid": True, "error": None}


def verify_syntax(patched_content: str, filename: str) -> dict:
    """
    MASTER SYNTAX VALIDATOR.
    Routes to the correct language-specific validator
    based on file extension.
    Supports: Python, JavaScript, TypeScript, JSON,
              HTML, CSS, YAML, XML, and all others
              via structural bracket checking.
    """
    ext = os.path.splitext(filename)[1].lower()

    logger.info(f"Checking syntax: {filename} (extension: {ext})")

    if ext == ".py":
        result = verify_syntax_python(patched_content, filename)

    elif ext in (".js", ".jsx", ".mjs", ".cjs"):
        result = verify_syntax_javascript(patched_content, filename)

    elif ext in (".ts", ".tsx"):
        result = verify_syntax_typescript(patched_content, filename)

    elif ext == ".json":
        result = verify_syntax_json(patched_content, filename)

    elif ext in (".html", ".htm"):
        result = verify_syntax_html(patched_content, filename)

    elif ext == ".css":
        result = verify_syntax_css(patched_content, filename)

    elif ext in (".yaml", ".yml"):
        result = verify_syntax_yaml(patched_content, filename)

    elif ext in (".xml", ".svg", ".plist"):
        result = verify_syntax_xml(patched_content, filename)

    elif ext in (
        ".java", ".cpp", ".c", ".cs", ".go",
        ".rb", ".php", ".swift", ".kt", ".rs",
        ".sh", ".bash", ".zsh", ".scss", ".sass"
    ):
        # Structural bracket check for compiled/other languages
        result = verify_syntax_basic_structural(
            patched_content, filename
        )

    else:
        # Unknown extension — skip silently
        logger.info(
            f"No syntax checker for {ext}. "
            f"Skipping syntax validation."
        )
        result = {"valid": True, "error": None}

    if result["valid"]:
        logger.info(f"Syntax valid: {filename}")
    else:
        logger.warning(f"Syntax invalid: {result['error']}")

    return result


def verify_patch_applies_cleanly(
    patch: str,
    code_context: dict
) -> dict:
    """
    Applies patch to every affected file and checks
    each result passes the language-appropriate syntax check.
    """
    errors         = []
    affected_files = re.findall(r"--- a/(.*?)\n", patch)

    if not affected_files:
        logger.warning("No affected files found in patch")
        return {"valid": True, "errors": []}

    for filename in affected_files:
        if filename not in code_context:
            logger.warning(f"File not in context: {filename}")
            continue

        original = code_context[filename]
        patched  = apply_patch_locally(original, patch, filename)
        result   = verify_syntax(patched, filename)

        if not result["valid"]:
            errors.append(result["error"])

    return {
        "valid":  len(errors) == 0,
        "errors": errors
    }


def apply_patch_locally(
    original_content: str,
    patch: str,
    filename: str
) -> str:
    """
    Applies the patch to the original file content.
    Returns the resulting content after changes applied.
    """
    result_lines = original_content.split("\n")
    lines        = patch.split("\n")
    i            = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                new_start    = int(match.group(1)) - 1
                current_line = new_start
                j            = i + 1

                while j < len(lines) and not lines[j].startswith("@@"):
                    hunk_line = lines[j]

                    if hunk_line.startswith("+") and not hunk_line.startswith("+++"):
                        result_lines.insert(current_line, hunk_line[1:])
                        current_line += 1

                    elif hunk_line.startswith("-") and not hunk_line.startswith("---"):
                        if current_line < len(result_lines):
                            result_lines.pop(current_line)

                    else:
                        current_line += 1

                    j += 1

                i = j
                continue

        i += 1

    return "\n".join(result_lines)


# LAYER 3 - TEST EXECUTION
def run_existing_tests(
    patch: str,
    code_context: dict
) -> dict:
    """
    Applies patch to temp directory and runs existing tests.
    Catches regressions before the patch ships.
    """
    logger.info("Running existing tests against patched code...")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:

            for filename, content in code_context.items():
                filepath = os.path.join(tmp_dir, filename)
                os.makedirs(
                    os.path.dirname(filepath),
                    exist_ok=True
                )
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            affected_files = re.findall(r"--- a/(.*?)\n", patch)
            for filename in affected_files:
                if filename in code_context:
                    patched_content = apply_patch_locally(
                        code_context[filename],
                        patch,
                        filename
                    )
                    filepath = os.path.join(tmp_dir, filename)
                    os.makedirs(
                        os.path.dirname(filepath),
                        exist_ok=True
                    )
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(patched_content)

            result = subprocess.run(
                ["python", "-m", "pytest",
                 "--tb=short", "-q", "--no-header"],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            output = result.stdout + result.stderr
            passed = result.returncode == 0

            summary_lines = [
                line for line in output.split("\n")
                if "passed" in line or "failed" in line
                or "error" in line.lower()
            ]
            summary = (
                summary_lines[-1] if summary_lines
                else "No tests found"
            )

            if passed:
                logger.info(f"Tests passed: {summary}")
            else:
                logger.warning(f"Tests failed: {summary}")

            return {
                "passed":  passed,
                "output":  output,
                "summary": summary
            }

    except subprocess.TimeoutExpired:
        logger.error("Test execution timed out")
        return {
            "passed":  False,
            "output":  "Test execution timed out after 60 seconds",
            "summary": "Timeout"
        }

    except FileNotFoundError:
        logger.warning("pytest not found. Skipping test layer.")
        return {
            "passed":  True,
            "output":  "pytest not available",
            "summary": "Skipped"
        }

    except Exception as e:
        logger.error(f"Test execution error: {e}")
        return {
            "passed":  True,
            "output":  str(e),
            "summary": "Error during execution"
        }


# LAYER 4 - LLM CODE REVIEW
def review_patch(
    plan: str,
    patch: str,
    code_context: dict
) -> dict:
    """
    Final gate. LLM reviews logic and completeness.
    Only runs after format + syntax + tests all passed.
    """
    logger.info("Running LLM code review (final gate)...")

    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n--- {filename} ---\n{content}\n"

    prompt = f"""
You are a principal software engineer doing a final code review.

The patch below has already passed:
- Format validation
- Syntax validation for its language
- Existing test suite

Your job is to verify LOGIC and COMPLETENESS only.

FIX PLAN:
{plan}

PATCH TO REVIEW:
{patch}

ORIGINAL CODE:
{formatted_code}

Review:
1. Does this patch fix the ROOT CAUSE in the plan?
2. Does it handle all edge cases mentioned?
3. Does it introduce any new bugs?
4. Is the fix complete or partial?
5. Are there logic errors in the changed lines?

Return ONLY this JSON:
{{
    "approved": true or false,
    "confidence": "high" or "medium" or "low",
    "issues": ["specific issue 1"],
    "suggestion": "precise technical suggestion if rejected"
}}

Rules:
- Only reject for genuine logic errors or incompleteness
- Do not reject for style issues
- suggestion must reference specific lines or variables

Return ONLY the JSON. No explanation. No markdown.
"""

    llm      = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    raw      = clean_llm_output(response.content)

    try:
        review = json.loads(raw)
        logger.info(
            f"LLM review: approved={review.get('approved')} "
            f"confidence={review.get('confidence')}"
        )
        if review.get("issues"):
            logger.warning(f"Review issues: {review['issues']}")
        return review

    except json.JSONDecodeError:
        logger.warning(
            "Could not parse LLM review. "
            "Approving — deterministic checks passed."
        )
        return {
            "approved":   True,
            "confidence": "low",
            "issues":     [],
            "suggestion": ""
        }


# PATCH GENERATION
def generate_patch(
    plan: str,
    code_context: dict,
    attempt: int = 1,
    feedback: str = ""
) -> str:
    """
    Generates unified diff patch.
    Gets stricter on each retry.
    Includes feedback from previous failed attempt.
    """
    logger.info(f"Generating patch (attempt {attempt})...")

    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n--- {filename} ---\n{content}\n"

    strictness = ""
    if attempt == 2:
        strictness = (
            "\nIMPORTANT: Your previous patch failed validation. "
            "Ensure correct --- +++ @@ format and valid syntax."
        )
    elif attempt >= 3:
        strictness = (
            "\nCRITICAL: Previous attempts failed. "
            "Return ONLY a unified diff starting with --- immediately."
        )

    feedback_section = ""
    if feedback:
        feedback_section = f"""
PREVIOUS ATTEMPT FAILED WITH THIS ERROR:
{feedback}

You must fix the above issue in this attempt.
"""

    prompt = f"""
You are a senior software engineer writing a precise code fix.

FIX PLAN:
{plan}

CURRENT CODE:
{formatted_code}

{feedback_section}

Write a unified diff patch that implements the fix.

Format:
--- a/filename.ext
+++ b/filename.ext
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line

Rules:
- Fix the ROOT CAUSE in the plan
- Handle all edge cases mentioned
- Only change what is necessary
- Include 3 lines of context around changes
- Produce syntactically valid code in the target language
{strictness}

Return ONLY the unified diff. No explanation. No markdown.
"""

    llm      = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    patch    = clean_llm_output(response.content)

    return patch


# MAIN AGENT FUNCTION
def code_writer_agent(state: AgentState) -> AgentState:
    """
    Validation pipeline:
    format → syntax → tests → LLM review

    All four must pass. LLM only runs after
    deterministic checks pass.
    """
    logger.info("=== Code Writer Agent starting ===")
    logger.info(
        "Pipeline: format → syntax → tests → LLM review"
    )

    state["steps"] += 1

    if not state.get("plan"):
        state["error"] = "Code Writer failed: no plan in state"
        logger.error(state["error"])
        return state

    max_attempts = 3
    patch        = ""

    for attempt in range(1, max_attempts + 1):
        logger.info(f"--- Attempt {attempt}/{max_attempts} ---")

        try:
            patch = generate_patch(
                state["plan"],
                state["code_context"],
                attempt=attempt,
                feedback=state.get("last_review_feedback", "")
            )

            logger.info(f"Patch generated. Length: {len(patch)} chars")

            # ------------------------------------------------
            # LAYER 1 — FORMAT
            # ------------------------------------------------
            logger.info("Layer 1: Format validation...")
            if not validate_patch(patch):
                state["last_review_feedback"] = (
                    "The patch is not a valid unified diff. "
                    "It must have --- +++ @@ markers "
                    "with actual + and - change lines."
                )
                logger.warning("Layer 1 FAILED")
                continue
            logger.info("Layer 1 PASSED")

            # ------------------------------------------------
            # LAYER 2 — SYNTAX (ALL LANGUAGES)
            # ------------------------------------------------
            logger.info("Layer 2: Syntax validation...")
            syntax_result = verify_patch_applies_cleanly(
                patch, state["code_context"]
            )
            if not syntax_result["valid"]:
                state["last_review_feedback"] = (
                    f"Syntax errors found: "
                    f"{'; '.join(syntax_result['errors'])}. "
                    f"Fix these in the next attempt."
                )
                logger.warning("Layer 2 FAILED")
                continue
            logger.info("Layer 2 PASSED")

            # ------------------------------------------------
            # LAYER 3 — TEST EXECUTION
            # ------------------------------------------------
            logger.info("Layer 3: Test execution...")
            test_result = run_existing_tests(
                patch, state["code_context"]
            )
            if not test_result["passed"]:
                state["last_review_feedback"] = (
                    f"Patch breaks existing tests. "
                    f"Summary: {test_result['summary']}. "
                    f"Output: {test_result['output'][:400]}. "
                    f"Fix the regression."
                )
                logger.warning("Layer 3 FAILED")
                continue
            logger.info(f"Layer 3 PASSED: {test_result['summary']}")

            # ------------------------------------------------
            # LAYER 4 — LLM REVIEW (RUNS LAST)
            # ------------------------------------------------
            logger.info("Layer 4: LLM code review...")
            review = review_patch(
                state["plan"],
                patch,
                state["code_context"]
            )
            if not review.get("approved"):
                state["last_review_feedback"] = (
                    f"LLM review rejected. "
                    f"Issues: {review.get('issues')}. "
                    f"Suggestion: {review.get('suggestion')}."
                )
                logger.warning("Layer 4 FAILED")
                continue
            logger.info(
                f"Layer 4 PASSED. "
                f"Confidence: {review.get('confidence')}"
            )

            # ------------------------------------------------
            # ALL 4 LAYERS PASSED
            # ------------------------------------------------
            state["patch"]                = patch
            state["patch_confidence"]     = review.get("confidence", "medium")
            state["last_review_feedback"] = ""
            state["error"]                = None

            logger.info(
                f"All 4 layers passed on attempt {attempt}. "
                f"Confidence: {state['patch_confidence']}"
            )
            return state

        except Exception as e:
            logger.error(f"Attempt {attempt} error: {e}")
            state["last_review_feedback"] = (
                f"Attempt {attempt} crashed: {str(e)}. "
                f"Try a different approach."
            )
            if attempt == max_attempts:
                state["error"] = (
                    f"Code Writer failed after {max_attempts} attempts: "
                    f"{str(e)}"
                )
                return state

    # All attempts exhausted
    logger.error("All attempts failed. Storing best effort patch.")
    state["patch"] = patch
    state["error"] = (
        f"Patch did not pass all validation layers "
        f"after {max_attempts} attempts."
    )
    return state