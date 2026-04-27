import os
import json
from github import Github, Auth
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from state import AgentState
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

MAX_FILES_TO_FETCH = 5

SKIP_FOLDERS = {
    "node_modules", "vendor", ".git",
    "dist", "build", "__pycache__",
    ".venv", "venv", "env"
}


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )


def get_github_client():
    token = os.getenv("GITHUB_TOKEN")
    auth = Auth.Token(token)
    return Github(auth=auth)


def clean_llm_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def fetch_issue_details(issue_url: str) -> dict:
    logger.info(f"Fetching issue from: {issue_url}")

    parts = issue_url.strip("/").split("/")
    owner = parts[-4]
    repo_name = parts[-3]
    issue_number = int(parts[-1])

    github = get_github_client()
    repo = github.get_repo(f"{owner}/{repo_name}")
    issue = repo.get_issue(number=issue_number)

    logger.info(f"Issue fetched: '{issue.title}'")

    return {
        "number":    issue.number,
        "title":     issue.title,
        "body":      issue.body or "No description provided",
        "owner":     owner,
        "repo_name": repo_name
    }


def get_relevant_extensions(
    issue_title: str,
    issue_body: str
) -> list[str]:
    """
    Step 1 — send only the issue to LLM.
    Ask what file extensions are relevant.
    Returns list like [".js", ".html", ".css"]
    This call is tiny — ~200 tokens.
    """
    logger.info("Asking LLM what file extensions are relevant...")

    prompt = f"""
You are a senior software engineer.
Read this GitHub issue carefully.

ISSUE TITLE: {issue_title}

ISSUE BODY: {issue_body}

What file extensions are most likely relevant to fix this issue?

Examples:
- UI/frontend bug → [".js", ".html", ".css"]
- Python backend bug → [".py"]
- Mixed issue → [".py", ".js"]
- Config issue → [".json", ".yaml"]

Return ONLY a JSON array of file extensions.
Maximum 4 extensions.
Example: [".js", ".html"]

Return ONLY the JSON array. No explanation. No markdown.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_llm_output(response.content)

    try:
        extensions = json.loads(raw)
        logger.info(f"Relevant extensions: {extensions}")
        return extensions
    except json.JSONDecodeError:
        logger.warning(
            "Could not parse extensions. "
            "Using safe defaults."
        )
        return [".py", ".js", ".ts", ".html"]


def get_relevant_files(
    issue_title: str,
    issue_body: str,
    repo
) -> list[str]:
    """
    Step 2 — use extensions from Step 1 to filter files.
    Only scan files matching those extensions.
    Then ask LLM which of those files are most relevant.
    """

    # Step 1 — ask LLM what extensions to look for.
    extensions = get_relevant_extensions(
        issue_title,
        issue_body
    )
    extensions_tuple = tuple(extensions)

    logger.info(
        f"Scanning repo for files ending in: {extensions}"
    )

    # Step 2 — walk repo but only collect matching files
    matching_files = []
    contents = repo.get_contents("")

    while contents:
        item = contents.pop(0)

        if item.type == "dir":
            folder_name = item.path.split("/")[-1]
            if folder_name in SKIP_FOLDERS:
                logger.info(f"Skipping folder: {item.path}")
                continue
            contents.extend(repo.get_contents(item.path))

        else:
            if item.path.endswith(extensions_tuple):
                matching_files.append(item.path)

    logger.info(
        f"Found {len(matching_files)} files "
        f"matching extensions {extensions}"
    )

    # Fallback if nothing found with suggested extensions
    if not matching_files:
        logger.warning(
            "No files found with suggested extensions. "
            "Falling back to .py and .js and .ts and .html"
        )
        extensions_tuple = (".py", ".js", ".ts", ".html")
        contents = repo.get_contents("")
        while contents:
            item = contents.pop(0)
            if item.type == "dir":
                folder_name = item.path.split("/")[-1]
                if folder_name in SKIP_FOLDERS:
                    continue
                contents.extend(repo.get_contents(item.path))
            else:
                if item.path.endswith(extensions_tuple):
                    matching_files.append(item.path)

    # Cap at 50 files to keep prompt small
    matching_files = matching_files[:50]

    # Step 3 — ask LLM to pick from small filtered list
    logger.info(
        f"Asking LLM to pick most relevant files "
        f"from {len(matching_files)} candidates..."
    )

    prompt = f"""
You are a senior software engineer analyzing a GitHub issue.

ISSUE TITLE: {issue_title}

ISSUE BODY: {issue_body}

These are the files in the repository matching
the relevant extensions:

{chr(10).join(matching_files)}

Which files are most likely to need changes
to fix this issue?

Return ONLY a JSON array of file paths.
Maximum {MAX_FILES_TO_FETCH} files.
Example: ["popup.js", "background.js"]

Return ONLY the JSON array. No explanation. No markdown.
"""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_llm_output(response.content)

    try:
        relevant_files = json.loads(raw)
        logger.info(f"LLM identified files: {relevant_files}")
        return relevant_files
    except json.JSONDecodeError:
        logger.error(f"Could not parse LLM response: {raw}")
        return matching_files[:MAX_FILES_TO_FETCH]


def fetch_file_contents(file_paths: list[str], repo) -> dict:
    code_context = {}

    for path in file_paths:
        try:
            logger.info(f"Fetching file: {path}")
            file_obj = repo.get_contents(path)
            code_context[path] = (
                file_obj.decoded_content.decode("utf-8")
            )
        except Exception as e:
            logger.warning(f"Could not fetch {path}: {e}")
            continue

    return code_context


def code_reader_agent(state: AgentState) -> AgentState:
    """
    THE MAIN AGENT FUNCTION.
    Input: AgentState
    Output: AgentState (updated)
    """
    logger.info("=== Code Reader Agent starting ===")

    state["steps"] += 1

    try:
        # STEP 1: Fetch the GitHub issue
        issue_data = fetch_issue_details(state["issue_url"])
        state["issue_title"] = issue_data["title"]
        state["issue_body"]  = issue_data["body"]

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
        state["error"] = None

        logger.info(
            f"Code Reader complete. "
            f"Fetched {len(code_context)} files: "
            f"{list(code_context.keys())}"
        )

    except Exception as e:
        error_msg = f"Code Reader failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg

    return state