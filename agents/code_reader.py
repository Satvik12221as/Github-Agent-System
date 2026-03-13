import os
import json
from github import Github
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from state import AgentState
from utils.logger import get_logger

# Load .env variables
load_dotenv()

# Create this agent's logger
logger = get_logger(__name__)


def get_llm():
    """
    Creates and returns the Gemini LLM instance.
    We call this inside the function so it's always fresh
    and always has the latest env variables loaded.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1   # Low temperature = more focused, less creative
    )                      # We want precise answers, not imaginative ones


def get_github_client():
    """
    Creates and returns an authenticated GitHub client.
    PyGithub uses our token to make authenticated requests
    """
    token = os.getenv("GITHUB_TOKEN")
    return Github(token)


def clean_llm_output(text: str) -> str:
    """
    Gemini sometimes wraps output in markdown code fences like:
```json
    ["file.py", "other.py"]
```
    This function strips those fences so we can parse clean JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        lines = text.split("\n")
        lines = lines[1:]  # remove first line
        # Remove closing fence
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def fetch_issue_details(issue_url: str) -> dict:
    """
    takes a GitHub issue URL like:
    https://github.com/username/reponame/issues/42

    Returns a dict with the issue number, title, and body.
    """
    logger.info(f"Fetching issue from: {issue_url}")

    # Parse the URL to extract owner, repo, issue number
    # URL format: https://github.com/OWNER/REPO/issues/NUMBER
    parts = issue_url.strip("/").split("/")
    owner = parts[-4]     # e.g. "username"
    repo_name = parts[-3] # e.g. "reponame"
    issue_number = int(parts[-1])  # e.g. 42.

    # Connect to GitHub and get the repo
    github = get_github_client()
    repo = github.get_repo(f"{owner}/{repo_name}")

    # Fetch the specific issue
    issue = repo.get_issue(number=issue_number)

    logger.info(f"Issue fetched: '{issue.title}'")

    return {
        "number": issue.number,
        "title": issue.title,
        "body": issue.body or "No description provided",
        "owner": owner,
        "repo_name": repo_name
    }


def get_relevant_files(issue_title: str, issue_body: str, repo) -> list[str]:
    """
    Asks Gemini: given this issue, which files in the repo are likely relevant?
    Returns a list of file paths like ["src/auth.py", "utils/helpers.py"]

    This is RAG-lite reasoning — instead of reading every file,
    we ask the LLM to filter first. Smart and cheap.
    """
    logger.info("Asking LLM to identify relevant files...")

    # First, get all file paths in the repo
    all_files = []
    contents = repo.get_contents("")  # Start at root

    # Walk through the repo tree
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            # It's a folder — go inside it
            contents.extend(repo.get_contents(file_content.path))
        else:
            # It's a file — add its path to our list
            # Only care about Python files for this project
            if file_content.path.endswith(".py"):
                all_files.append(file_content.path)

    logger.info(f"Found {len(all_files)} Python files in repo")

    # Now ask Gemini which ones are relevant to the issue
    prompt = f"""
You are a senior software engineer analyzing a GitHub issue.

ISSUE TITLE: {issue_title}

ISSUE BODY: {issue_body}

REPOSITORY FILES:
{chr(10).join(all_files)}

Based on the issue description, which files are MOST LIKELY to need changes?
Return ONLY a JSON array of file paths. Maximum 5 files.
Example: ["src/auth.py", "utils/helpers.py"]

Return ONLY the JSON array, no explanation, no markdown.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])

    # Clean and parse the response
    raw = clean_llm_output(response.content)

    try:
        relevant_files = json.loads(raw)
        logger.info(f"LLM identified files: {relevant_files}")
        return relevant_files
    except json.JSONDecodeError:
        logger.error(f"Could not parse LLM response: {raw}")
        # Fallback: return first 3 files if LLM gave bad output
        return all_files[:3]


def fetch_file_contents(file_paths: list[str], repo) -> dict:
    """
    Takes a list of file paths and fetches their actual code content.
    Returns a dict: { "filepath": "file content as string" }
    """
    code_context = {}

    for path in file_paths:
        try:
            logger.info(f"Fetching file: {path}")
            file_obj = repo.get_contents(path)

            # file_obj.decoded_content gives us bytes
            # .decode("utf-8") turns bytes into a readable string
            code_context[path] = file_obj.decoded_content.decode("utf-8")

        except Exception as e:
            logger.warning(f"Could not fetch {path}: {e}")
            # Don't crash — just skip this file and continue
            continue

    return code_context


def code_reader_agent(state: AgentState) -> AgentState:
    """
    THE MAIN AGENT FUNCTION.
    LangGraph calls this function and passes in the current state.
    We read from state, do our work, write back to state, return it.

    This is the contract every agent follows:
    Input: AgentState
    Output: AgentState (updated)
    """
    logger.info("=== Code Reader Agent starting ===")

    # Increment step counter — orchestrator uses this as circuit breaker
    state["steps"] += 1

    try:
        # STEP 1: Fetch the GitHub issue
        issue_data = fetch_issue_details(state["issue_url"])

        # Write issue details into state
        state["issue_title"] = issue_data["title"]
        state["issue_body"] = issue_data["body"]

        # STEP 2: Get a reference to the repo
        github = get_github_client()
        repo = github.get_repo(
            f"{issue_data['owner']}/{issue_data['repo_name']}"
        )

        # STEP 3: Ask LLM which files are relevant
        relevant_files = get_relevant_files(
            issue_data["title"],
            issue_data["body"],
            repo
        )

        # STEP 4: Fetch contents of those files
        code_context = fetch_file_contents(relevant_files, repo)

        # STEP 5: Write everything into state
        state["code_context"] = code_context
        state["error"] = None  # Clear any previous error

        logger.info(
            f"Code Reader complete. "
            f"Fetched {len(code_context)} files: {list(code_context.keys())}"
        )

    except Exception as e:
        # Something went wrong — write error to state
        # Don't crash the whole system
        error_msg = f"Code Reader failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg

    return state