import os
import json
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
    """Same fence stripper we used in code_reader."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def build_plan(issue_title: str, issue_body: str, code_context: dict) -> dict:
    """
    Sends the issue + relevant code to Gemini.
    Asks for a structured JSON plan back.
    Returns a dict with: summary, steps, affected_files, complexity
    """
    logger.info("Building fix plan with Gemini...")

    # Format the code context for the prompt
    # We turn the dict into readable text: filename + contents
    formatted_code = ""
    for filename, content in code_context.items():
        formatted_code += f"\n--- {filename} ---\n{content}\n"

    prompt = f"""
You are a senior software engineer.
Analyze this GitHub issue and the relevant code, then produce a fix plan.

ISSUE TITLE: {issue_title}

ISSUE BODY: {issue_body}

RELEVANT CODE:
{formatted_code}

Produce a JSON response with exactly these fields:
{{
  "summary": "one sentence describing what needs to be fixed",
  "steps": ["step 1", "step 2", "step 3"],
  "affected_files": ["file1.py", "file2.py"],
  "complexity": "simple" or "complex",
  "risk": "low" or "medium" or "high"
}}

Complexity rules:
- "simple" = fix is in 1-2 files, straightforward change, low risk
- "complex" = fix spans 3+ files, needs refactoring, or touches critical logic

Return ONLY the JSON object. No explanation. No markdown.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_llm_output(response.content)

    try:
        plan = json.loads(raw)
        logger.info(f"Plan built. Complexity: {plan.get('complexity')}")
        logger.info(f"Summary: {plan.get('summary')}")
        return plan

    except json.JSONDecodeError:
        logger.error(f"Could not parse plan JSON: {raw}")
        # Return a safe fallback plan so the system keeps running
        return {
            "summary": "Fix the reported issue",
            "steps": ["Investigate the issue", "Apply fix", "Test fix"],
            "affected_files": list(code_context.keys()),
            "complexity": "simple",
            "risk": "low"
        }


def planner_agent(state: AgentState) -> AgentState:
    """
    THE MAIN AGENT FUNCTION.
    Reads: issue_title, issue_body, code_context
    Writes: plan, complexity
    """
    logger.info("=== Planner Agent starting ===")

    state["steps"] += 1

    # Safety check — if code_reader failed and context is empty
    # we still try to plan from just the issue description
    if not state["code_context"]:
        logger.warning("No code context found. Planning from issue only.")

    try:
        plan_data = build_plan(
            state["issue_title"],
            state["issue_body"],
            state["code_context"]
        )

        # Write the human readable plan as a formatted string into state
        # This is what Code Writer will read
        state["plan"] = f"""
SUMMARY: {plan_data['summary']}

STEPS TO FIX:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(plan_data['steps']))}

AFFECTED FILES: {', '.join(plan_data['affected_files'])}

RISK LEVEL: {plan_data['risk']}
""".strip()

        # Write complexity separately — orchestrator reads this for routing
        state["complexity"] = plan_data.get("complexity", "simple")
        state["error"] = None

        logger.info(f"Planner complete. Complexity={state['complexity']}")
        logger.info(f"Plan:\n{state['plan']}")

    except Exception as e:
        error_msg = f"Planner failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        # Default to simple so the system can still attempt to continue
        state["complexity"] = "simple"

    return state