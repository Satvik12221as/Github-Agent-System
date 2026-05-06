import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from state import AgentState, update_token_usage
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
    """Same fence stripper we used in code_reader."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def build_plan(issue_title: str, issue_body: str, code_context: dict, state: AgentState) -> dict:
    """
    Deep root cause analysis.
    Does not just describe what to fix — identifies WHY the bug exists
    at the code level and produces a precise surgical fix plan.
    """
    logger.info("Performing deep root cause analysis...")

    # Format code with line numbers so LLM can reference specific lines
    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n{'='*60}\n"
        formatted_code += f"FILE: {filename}\n"
        formatted_code += f"{'='*60}\n"
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            formatted_code += f"{i:4d} | {line}\n"

    prompt = f"""
You are a principal software engineer doing a deep root cause analysis.
Your job is NOT to describe the symptom. Your job is to find exactly
WHY the bug exists in the code and HOW to fix it permanently.

ISSUE TITLE: {issue_title}

ISSUE DESCRIPTION: {issue_body}

CODEBASE:
{formatted_code}

Perform a deep root cause analysis:

1. Read every line of the code carefully
2. Find the EXACT line or lines causing the bug
3. Understand WHY those lines cause the bug
4. Design a fix that addresses the root cause permanently
5. Consider edge cases the fix must handle

Return ONLY this JSON object:

{{
    "root_cause": "exact technical explanation of WHY the bug exists — reference specific line numbers and variable names from the code",
    "bug_location": {{
        "file": "exact filename",
        "lines": "line range like 42-48",
        "code_snippet": "the exact buggy code"
    }},
    "why_it_fails": "step by step explanation of the failure chain — what happens at runtime that causes the symptom",
    "fix_approach": "the precise technical fix — what exactly changes and why this permanently solves the root cause",
    "steps": [
        "specific step referencing exact function/variable names",
        "specific step referencing exact function/variable names"
    ],
    "affected_files": ["file1.py"],
    "edge_cases": [
        "edge case this fix must handle",
        "another edge case"
    ],
    "risk": "low" or "medium" or "high",
    "risk_reason": "why this risk level — what could go wrong"
}}

Rules:
- root_cause must reference specific line numbers from the code above
- fix_approach must be technically precise — not vague
- steps must reference actual function names, variable names, class names from the code
- Never write generic steps like "fix the bug" or "update the code"
- Every step must be something a developer can implement directly

Return ONLY the JSON. No explanation. No markdown.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    update_token_usage(state, response)
    raw = clean_llm_output(response.content)

    try:
        plan = json.loads(raw)

        logger.info(f"Root cause: {plan.get('root_cause')}")
        logger.info(f"Bug location: {plan.get('bug_location')}")
        logger.info(f"Fix approach: {plan.get('fix_approach')}")
        logger.info(f"Risk: {plan.get('risk')} — {plan.get('risk_reason')}")

        return plan

    except json.JSONDecodeError:
        logger.error(f"Could not parse plan JSON: {raw}")
        return {
            "root_cause":   "Could not determine root cause",
            "bug_location": {
                "file":         list(code_context.keys())[0] if code_context else "",
                "lines":        "unknown",
                "code_snippet": ""
            },
            "why_it_fails":  "Unknown",
            "fix_approach":  "Investigate and fix the reported issue",
            "steps":         ["Investigate the root cause", "Apply targeted fix", "Test edge cases"],
            "affected_files": list(code_context.keys()),
            "edge_cases":    [],
            "risk":          "medium",
            "risk_reason":   "Unknown root cause"
        }


def planner_agent(state: AgentState) -> AgentState:
    logger.info("=== Planner Agent starting ===")
    state["steps"] += 1

    if not state.get("code_context"):
        logger.warning("No code context. Planning from issue only.")

    try:
        plan_data = build_plan(
            state["issue_title"],
            state["issue_body"],
            state["code_context"],
            state
        )

        # Format into a rich plan string that Code Writer reads
        # Include root cause so Code Writer knows exactly what to fix
        edge_cases_text = "\n".join([
            f"  - {ec}"
            for ec in plan_data.get("edge_cases", [])
        ]) or "  - None identified"

        bug_location = plan_data.get("bug_location", {})

        state["plan"] = f"""
ROOT CAUSE:
{plan_data.get('root_cause', 'Unknown')}

BUG LOCATION:
  File:    {bug_location.get('file', 'unknown')}
  Lines:   {bug_location.get('lines', 'unknown')}
  Code:    {bug_location.get('code_snippet', 'unknown')}

WHY IT FAILS:
{plan_data.get('why_it_fails', 'Unknown')}

FIX APPROACH:
{plan_data.get('fix_approach', 'Unknown')}

STEPS TO FIX:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(plan_data.get("steps", [])))}

EDGE CASES TO HANDLE:
{edge_cases_text}

AFFECTED FILES: {', '.join(plan_data.get('affected_files', []))}

RISK: {plan_data.get('risk', 'unknown')} — {plan_data.get('risk_reason', '')}
""".strip()

        # Complexity is now derived from risk and affected files
        # not just a simple label
        affected_count = len(plan_data.get("affected_files", []))
        risk           = plan_data.get("risk", "low")

        if affected_count >= 3 or risk == "high":
            state["complexity"] = "complex"
        else:
            state["complexity"] = "simple"

        state["error"] = None

        logger.info(f"Planner complete. Complexity={state['complexity']}")
        logger.info(f"Plan:\n{state['plan']}")

    except Exception as e:
        error_msg = f"Planner failed: {str(e)}"
        logger.error(error_msg)
        state["error"]      = error_msg
        state["complexity"] = "simple"

    return state