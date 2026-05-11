import os
import json
from github import Github, Auth
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState, update_token_usage
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

MAX_FILES_TO_FETCH = 5

SKIP_FOLDERS = {
    "node_modules", "vendor", ".git",
    "dist", "build", "__pycache__",
    ".venv", "venv", "env"
}

# LLM SETUP
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )



def get_github_client():
    token = os.getenv("GITHUB_TOKEN")
    auth  = Auth.Token(token)
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


# ISSUE COMMENT READING
def fetch_issue_comments(
    issue,
    max_comments: int = 10
) -> str:
    """
    Fetches all comments on the GitHub issue.

    WHY THIS MATTERS:
    The issue body is what the reporter wrote initially.
    Comments contain everything that happened after:
    - Stack traces the reporter added later
    - Maintainer hints pointing to the exact file
    - Other developers reproducing the bug differently
    - Links to related PRs that attempted fixes before
    - Workarounds that reveal what the actual problem is

    Without comments your agent only has 20% of the context.
    With comments it has 100%.

    Example of what gets missed without this:
    Issue body: "The camera shows a black screen"
    Comment from maintainer: "This is happening in line 47
    of gaze_tracking.py where cap.read() returns None
    during initialization"

    Without reading comments the agent has to guess.
    With comments it knows exactly where to look.
    """
    try:
        comments = list(issue.get_comments()[:max_comments])

        if not comments:
            logger.info("No comments on this issue")
            return ""

        formatted = "\n\nISSUE DISCUSSION (comments):\n"
        formatted += "=" * 50 + "\n"

        for i, comment in enumerate(comments, 1):
            formatted += f"\n[Comment {i} by {comment.user.login}]:\n"
            # Limit each comment to 800 chars to avoid token explosion
            body = comment.body or ""
            if len(body) > 800:
                body = body[:800] + "... [truncated]"
            formatted += body + "\n"
            formatted += "-" * 40 + "\n"

        logger.info(
            f"Fetched {len(comments)} issue comments "
            f"providing additional context"
        )
        return formatted

    except Exception as e:
        logger.warning(f"Could not fetch issue comments: {e}")
        return ""


# SIMILAR ISSUE DETECTION
def find_similar_closed_issues(
    repo,
    issue_title: str,
    issue_number: int,
    max_results: int = 3
) -> str:
    """
    Searches the repo's closed issues for similar problems
    that were already fixed.

    WHY THIS MATTERS:
    If the same bug or a related bug was fixed before,
    the solution pattern is already known. Your agent
    does not need to figure it out from scratch.

    Example:
    Current issue: "Black screen before camera opens"
    Similar closed issue: "Camera feed delay on startup"
    That closed issue might have a linked PR showing exactly
    how cap.read() initialization was fixed last time.

    This gives the Planner a huge advantage:
    instead of analyzing cold, it sees what worked before
    and can apply the same pattern or avoid known pitfalls.

    We extract:
    - The title of similar issues (pattern matching)
    - Their descriptions (additional context)
    - Whether they have linked PRs (solution reference)
    """
    try:
        # Extract meaningful keywords from the issue title
        # Skip common words that would match everything
        stop_words = {
            "the", "a", "an", "is", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "not",
            "with", "from", "by", "this", "that", "it"
        }
        keywords = [
            word.lower()
            for word in issue_title.split()
            if word.lower() not in stop_words
            and len(word) > 2
        ]

        if not keywords:
            logger.info("No meaningful keywords for similarity search")
            return ""

        logger.info(
            f"Searching for similar closed issues "
            f"using keywords: {keywords}"
        )

        similar   = []
        checked   = 0
        max_check = 50  # check last 50 closed issues max

        # Get recently closed issues
        closed_issues = repo.get_issues(
            state="closed",
            sort="updated",
            direction="desc"
        )

        for closed_issue in closed_issues:
            if checked >= max_check:
                break

            # Skip the current issue itself
            if closed_issue.number == issue_number:
                checked += 1
                continue

            # Check if any keyword appears in the title
            title_lower = closed_issue.title.lower()
            matches = sum(
                1 for kw in keywords
                if kw in title_lower
            )

            # Need at least 2 keyword matches
            # to consider it truly similar
            if matches >= 2:
                similar.append({
                    "issue":   closed_issue,
                    "matches": matches
                })

            checked += 1

            if len(similar) >= max_results:
                break

        if not similar:
            logger.info("No similar closed issues found")
            return ""

        # Sort by number of keyword matches
        similar.sort(key=lambda x: x["matches"], reverse=True)

        formatted  = "\n\nSIMILAR PREVIOUSLY FIXED ISSUES:\n"
        formatted += "=" * 50 + "\n"
        formatted += (
            "These issues were already fixed. "
            "Use their context to understand the problem pattern.\n\n"
        )

        for item in similar:
            issue  = item["issue"]
            formatted += f"#{issue.number}: {issue.title}\n"
            formatted += f"Keyword matches: {item['matches']}\n"

            # Include issue body preview
            if issue.body:
                preview = issue.body[:400]
                if len(issue.body) > 400:
                    preview += "... [truncated]"
                formatted += f"Description: {preview}\n"

            # Check if there was a linked PR
            # GitHub auto-links PRs that mention issue numbers
            try:
                events = list(issue.get_events())
                for event in events:
                    if event.event == "cross-referenced":
                        formatted += (
                            f"Linked PR/commit: "
                            f"{event.source.issue.html_url}\n"
                        )
                        break
            except Exception:
                pass

            formatted += "-" * 40 + "\n"

        logger.info(
            f"Found {len(similar)} similar closed issues"
        )
        return formatted

    except Exception as e:
        logger.warning(f"Could not search similar issues: {e}")
        return ""


# UPDATED FETCH ISSUE DETAILS (includes comments and similar issues)
def fetch_issue_details(issue_url: str, state: AgentState) -> dict:
    """
    Fetches the complete issue context:
    - Title and body (original)
    - All comments (new - provides additional context)
    - Similar closed issues (new - provides solution patterns)

    The issue body is extended with comments and similar issues
    so the Planner sees the complete picture when it reads
    state["issue_body"].
    """
    logger.info(f"Fetching issue from: {issue_url}")

    parts        = issue_url.strip("/").split("/")
    owner        = parts[-4]
    repo_name    = parts[-3]
    issue_number = int(parts[-1])

    github = get_github_client()
    repo   = github.get_repo(f"{owner}/{repo_name}")
    issue  = repo.get_issue(number=issue_number)

    logger.info(f"Issue fetched: '{issue.title}'")

    # Get original body
    original_body = issue.body or "No description provided"

    # fetch comments for additional context
    logger.info("Fetching issue comments...")
    comments_context = fetch_issue_comments(issue)

    # find similar closed issues
    logger.info("Searching for similar closed issues...")
    similar_context = find_similar_closed_issues(
        repo,
        issue.title,
        issue_number
    )

    # Combine everything into one rich context string
    enriched_body = original_body

    if comments_context:
        enriched_body += comments_context

    if similar_context:
        enriched_body += similar_context

    if comments_context or similar_context:
        logger.info(
            "Issue body enriched with "
            f"{'comments ' if comments_context else ''}"
            f"{'and similar issues ' if similar_context else ''}"
        )

    return {
        "number":    issue.number,
        "title":     issue.title,
        "body":      enriched_body,
        "owner":     owner,
        "repo_name": repo_name
    }


def get_relevant_extensions(
    issue_title: str,
    issue_body: str,
    state: AgentState
) -> list[str]:
    """
    Step 1 - send only the issue to LLM.
    Ask what file extensions are relevant.
    Returns list like [".js", ".html", ".css"]
    This call is tiny - ~200 tokens.
    """
    logger.info("Asking LLM what file extensions are relevant...")

    prompt = f"""
You are a senior software engineer.
Read this GitHub issue carefully.

ISSUE TITLE: {issue_title}

ISSUE BODY: {issue_body[:1000]}

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

    llm      = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    update_token_usage(state, response)
    raw      = clean_llm_output(response.content)

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
    repo,
    state: AgentState
) -> list[str]:
    """
    Step 2 - use extensions from Step 1 to filter files.
    Only scan files matching those extensions.
    Then ask LLM which of those files are most relevant.
    """

    extensions       = get_relevant_extensions(
        issue_title, issue_body, state
    )
    extensions_tuple = tuple(extensions)

    logger.info(
        f"Scanning repo for files ending in: {extensions}"
    )

    matching_files = []
    contents       = repo.get_contents("")

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

    if not matching_files:
        logger.warning(
            "No files found with suggested extensions. "
            "Falling back to .py .js .ts .html"
        )
        extensions_tuple = (".py", ".js", ".ts", ".html")
        contents         = repo.get_contents("")
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

    matching_files = matching_files[:50]

    logger.info(
        f"Asking LLM to pick most relevant files "
        f"from {len(matching_files)} candidates..."
    )

    # Use truncated issue body here to keep prompt small
    # Full body with comments is only needed by the Planner
    prompt = f"""
You are a senior software engineer analyzing a GitHub issue.

ISSUE TITLE: {issue_title}

ISSUE BODY: {issue_body[:500]}

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

    llm      = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    update_token_usage(state, response)
    raw      = clean_llm_output(response.content)

    try:
        relevant_files = json.loads(raw)
        logger.info(f"LLM identified files: {relevant_files}")
        return relevant_files
    except json.JSONDecodeError:
        logger.error(f"Could not parse LLM response: {raw}")
        return matching_files[:MAX_FILES_TO_FETCH]


def fetch_file_contents(
    file_paths: list[str],
    repo
) -> dict:
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

    What changed:
    fetch_issue_details() now also fetches:
    1. Issue comments - for additional bug context
    2. Similar closed issues - for solution patterns

    Both are appended to state["issue_body"] so the Planner
    reads a fully enriched context automatically.
    No other agent needs to change.
    """
    logger.info("=== Code Reader Agent starting ===")

    state["steps"] += 1

    try:
        # STEP 1 - Fetch issue with full context
        # Now includes comments and similar issues
        issue_data = fetch_issue_details(
            state["issue_url"],
            state
        )
        state["issue_title"] = issue_data["title"]
        state["issue_body"]  = issue_data["body"]

        # STEP 2 - Get repo reference
        github = get_github_client()
        repo   = github.get_repo(
            f"{issue_data['owner']}/{issue_data['repo_name']}"
        )

        # STEP 3 - Find relevant files
        # Note: we pass only the first 500 chars of issue body
        # to file selection - full enriched body is for Planner
        relevant_files = get_relevant_files(
            issue_data["title"],
            issue_data["body"][:500],
            repo,
            state
        )

        # STEP 4 - Fetch file contents
        code_context = fetch_file_contents(relevant_files, repo)

        # STEP 5 - Write to state
        state["code_context"] = code_context
        state["error"]        = None

        logger.info(
            f"Code Reader complete. "
            f"Fetched {len(code_context)} files: "
            f"{list(code_context.keys())}"
        )

        # Log context richness for debugging
        body_length = len(state["issue_body"])
        has_comments = "ISSUE DISCUSSION" in state["issue_body"]
        has_similar  = "SIMILAR PREVIOUSLY" in state["issue_body"]
        logger.info(
            f"Issue body enriched: "
            f"{body_length} chars | "
            f"comments={'yes' if has_comments else 'no'} | "
            f"similar_issues={'yes' if has_similar else 'no'}"
        )

    except Exception as e:
        error_msg = f"Code Reader failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg

    return state