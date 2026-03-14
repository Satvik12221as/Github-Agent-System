
from state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)


def code_writer_agent(state: AgentState) -> AgentState:
    """
    Placeholder for Phase 5.
    Does nothing except log and increment steps.
    Gets replaced with real implementation in Phase 5.
    """
    logger.info("=== Code Writer Agent (placeholder) ===")
    state["steps"] += 1
    state["patch"] = "# placeholder patch"
    return state


def test_writer_agent(state: AgentState) -> AgentState:
    """
    Placeholder for Phase 6.
    """
    logger.info("=== Test Writer Agent (placeholder) ===")
    state["steps"] += 1
    state["tests"] = "# placeholder tests"
    state["test_result"] = "passed"
    return state


def pr_opener_agent(state: AgentState) -> AgentState:
    """
    Placeholder for Phase 7.
    """
    logger.info("=== PR Opener Agent (placeholder) ===")
    state["steps"] += 1
    state["pr_url"] = "https://github.com/placeholder/pr/1"
    return state