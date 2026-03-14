from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from state import AgentState
from agents.code_reader import code_reader_agent
from agents.planner import planner_agent
from agents.placeholder import (
    code_writer_agent,
    test_writer_agent,
    pr_opener_agent
)
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


# ← TOP LEVEL — NOT inside any function
def route_by_complexity(state: AgentState) -> str:
    complexity = state.get("complexity", "simple")
    logger.info(f"Routing decision: complexity = {complexity}")
    if complexity == "complex":
        return "complex"
    return "simple"


# ← TOP LEVEL — NOT inside any function
def route_after_tests(state: AgentState) -> str:
    test_result = state.get("test_result", "failed")
    retry_count = state.get("retry_count", 0)
    logger.info(f"Test result: {test_result}, Retry count: {retry_count}")
    if retry_count >= 3:
        logger.warning("Max retries reached. Forcing PR open.")
        return "open_pr"
    if test_result == "passed":
        return "open_pr"
    else:
        state["retry_count"] = retry_count + 1
        return "retry"


# ← TOP LEVEL — NOT inside any function
def check_for_errors(state: AgentState) -> str:
    if state.get("error"):
        logger.error(f"Error detected after Code Reader: {state['error']}")
        return "has_error"
    return "no_error"


# ← TOP LEVEL — build_workflow is also top level
def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("code_reader", code_reader_agent)
    workflow.add_node("planner",     planner_agent)
    workflow.add_node("code_writer", code_writer_agent)
    workflow.add_node("test_writer", test_writer_agent)
    workflow.add_node("pr_opener",   pr_opener_agent)

    workflow.set_entry_point("code_reader")

    workflow.add_conditional_edges(
        "code_reader",
        check_for_errors,
        {
            "no_error":  "planner",
            "has_error": END
        }
    )

    workflow.add_conditional_edges(
        "planner",
        route_by_complexity,
        {
            "simple":  "code_writer",
            "complex": "code_writer"
        }
    )

    workflow.add_edge("code_writer", "test_writer")

    workflow.add_conditional_edges(
        "test_writer",
        route_after_tests,
        {
            "open_pr": "pr_opener",
            "retry":   "code_writer"
        }
    )

    workflow.add_edge("pr_opener", END)

    app = workflow.compile()
    logger.info("Workflow compiled successfully")

    return app


def run_workflow(issue_url: str) -> AgentState:
    from state import get_initial_state
    logger.info(f"Starting workflow for: {issue_url}")
    initial_state = get_initial_state(issue_url)
    app = build_workflow()
    final_state = app.invoke(initial_state)
    logger.info(f"Workflow complete. PR URL: {final_state.get('pr_url')}")
    return final_state