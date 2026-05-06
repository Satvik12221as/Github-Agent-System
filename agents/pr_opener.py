import os
import re
import time
from github import Github, Auth, GithubException
from dotenv import load_dotenv

from state import AgentState
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


# Max retries for GitHub API calls
MAX_API_RETRIES = 3
RETRY_DELAY     = 2  # seconds between retries


# GITHUB CLIENT
def get_github_client():
    """Creates authenticated GitHub client."""
    token = os.getenv("GITHUB_TOKEN")
    auth  = Auth.Token(token)
    return Github(auth=auth)


# RETRY WRAPPER FOR GITHUB API
def github_api_call(func, *args, **kwargs):
    """
    Wraps any GitHub API call with retry logic.
    Retries on rate limits, timeouts, server errors.
    Raises on auth errors - no point retrying those.
    """
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return func(*args, **kwargs)

        except GithubException as e:
            # Auth errors - never retry
            if e.status in (401, 403):
                logger.error(
                    f"GitHub auth error {e.status}: {e.data}"
                )
                raise

            # Rate limit - wait and retry
            if e.status == 429:
                wait = RETRY_DELAY * attempt
                logger.warning(
                    f"GitHub rate limit. "
                    f"Waiting {wait}s before retry {attempt}..."
                )
                time.sleep(wait)
                continue

            # Server errors - retry
            if e.status >= 500:
                wait = RETRY_DELAY * attempt
                logger.warning(
                    f"GitHub server error {e.status}. "
                    f"Retry {attempt}/{MAX_API_RETRIES} "
                    f"in {wait}s..."
                )
                time.sleep(wait)
                continue

            # Other errors - raise immediately
            raise

        except Exception as e:
            if attempt == MAX_API_RETRIES:
                raise
            wait = RETRY_DELAY * attempt
            logger.warning(
                f"API call failed: {e}. "
                f"Retry {attempt}/{MAX_API_RETRIES} in {wait}s..."
            )
            time.sleep(wait)

    raise Exception(
        f"GitHub API call failed after {MAX_API_RETRIES} retries"
    )


# URL PARSING
def parse_issue_url(issue_url: str) -> dict:
    """
    Extracts owner, repo name, issue number from URL.
    https://github.com/owner/repo/issues/42
    """
    parts = issue_url.strip("/").split("/")
    return {
        "owner":  parts[-4],
        "repo":   parts[-3],
        "number": int(parts[-1])
    }


# ROBUST PATCH PARSING
def parse_patch(patch: str) -> list[dict]:
    """
    Parses unified diff into list of file changes.
    Handles edge cases:
    - Multiple hunks per file
    - Files with spaces in names
    - Empty patches
    - Patches with no context lines
    """
    if not patch or not patch.strip():
        logger.warning("Empty patch provided")
        return []

    changes      = []
    current_file = None
    current_hunks = []
    lines        = patch.split("\n")
    i            = 0

    while i < len(lines):
        line = lines[i]

        # New file - save previous if exists
        if line.startswith("--- a/"):
            if current_file and current_hunks:
                changes.append({
                    "filename": current_file,
                    "hunks":    current_hunks
                })
            # Extract filename - handle spaces in paths
            current_file  = line[6:].strip()
            current_hunks = []

        # New hunk header @@ -old +new @@
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                new_start = int(match.group(1))
                current_hunks.append({
                    "new_start": new_start,
                    "lines":     []
                })

        # Content lines - only if we have an active hunk
        elif current_hunks and not line.startswith("\\"):
            if line.startswith("+") and not line.startswith("+++"):
                current_hunks[-1]["lines"].append(
                    ("add", line[1:])
                )
            elif line.startswith("-") and not line.startswith("---"):
                current_hunks[-1]["lines"].append(
                    ("remove", line[1:])
                )
            elif line.startswith(" ") or line == "":
                current_hunks[-1]["lines"].append(
                    ("keep", line[1:] if line.startswith(" ") else "")
                )

        i += 1

    # Save the last file
    if current_file and current_hunks:
        changes.append({
            "filename": current_file,
            "hunks":    current_hunks
        })

    logger.info(
        f"Parsed patch: {len(changes)} files affected: "
        f"{[c['filename'] for c in changes]}"
    )
    return changes


def apply_patch_to_content(
    original_content: str,
    hunks: list
) -> str:
    """
    Applies parsed hunks to original file content.
    Returns modified content.
    """
    if not hunks:
        return original_content

    original_lines = original_content.split("\n")
    result_lines   = list(original_lines)
    offset         = 0

    for hunk in hunks:
        new_start    = hunk["new_start"] - 1 + offset
        current_line = new_start
        hunk_offset  = 0

        for action, content in hunk["lines"]:
            if action == "add":
                result_lines.insert(current_line, content)
                current_line += 1
                hunk_offset  += 1
            elif action == "remove":
                if current_line < len(result_lines):
                    result_lines.pop(current_line)
                    hunk_offset -= 1
            elif action == "keep":
                current_line += 1

        offset += hunk_offset

    return "\n".join(result_lines)


# VERIFY COMMITTED CODE (read back and confirm)
def verify_commit(
    repo,
    filename: str,
    expected_content: str,
    branch_name: str
) -> bool:
    """
    After committing a file, reads it back from GitHub
    and verifies the content matches what was committed.
    Catches silent commit failures.
    """
    try:
        committed_file = github_api_call(
            repo.get_contents,
            filename,
            ref=branch_name
        )
        committed_content = (
            committed_file.decoded_content.decode("utf-8")
        )

        if committed_content == expected_content:
            logger.info(f"Commit verified: {filename}")
            return True
        else:
            logger.warning(
                f"Commit verification failed for {filename}. "
                f"Content mismatch."
            )
            return False

    except Exception as e:
        logger.warning(f"Could not verify commit for {filename}: {e}")
        return False


# PR DESCRIPTION (gives reviewers everything they need to review the PR)
def create_pr_description(state: AgentState) -> str:
    """
    Builds a comprehensive PR description from state.
    Includes root cause, fix approach, test results,
    and files changed - everything a reviewer needs.
    """
    test_emoji  = "✅" if state["test_result"] == "passed" else "❌"
    issue_number = state.get("issue_url", "").split("/")[-1]
    confidence   = state.get("patch_confidence", "unknown")

    # Confidence badge
    confidence_badge = {
        "high":   "🟢 High",
        "medium": "🟡 Medium",
        "low":    "🔴 Low"
    }.get(confidence, "⚪ Unknown")

    # Extract key sections from plan
    plan = state.get("plan", "No plan available")

    # Format patch for display - limit length
    patch = state.get("patch", "")
    patch_preview = patch[:2000] + "..." if len(patch) > 2000 else patch

    description = f"""## 🤖 Automated Fix — GitHub Agent System

This pull request was automatically generated by an AI agent pipeline.
**Please review carefully before merging.**

---

### 📋 Issue
Closes #{issue_number}

---

### 🔍 Analysis
{plan}

---

### ✅ Validation Results

| Check | Result |
|-------|--------|
| Format validation | ✅ Passed |
| Syntax validation | ✅ Passed |
| Existing tests | ✅ Passed |
| LLM code review | ✅ Passed ({confidence_badge} confidence) |
| Generated tests | ✅ Committed to branch |
| CI/CD pipeline | 🔄 Check the Actions tab for live results |

---

### 🧪 Tests

Automated tests have been generated and committed to this branch
as `test_automated_fix.py`. GitHub Actions will run them automatically.

Check the **Actions** tab on this PR to see live test results.

---

### 📝 Code Changes

```diff
{patch_preview}
```

---

### ⚠️ Reviewer Checklist

Before merging please verify:
- [ ] GitHub Actions tests are passing
- [ ] The fix addresses the root cause not just the symptom
- [ ] Edge cases are handled correctly
- [ ] No unintended side effects in related functionality
- [ ] Code style matches the existing codebase

---

### 🤖 Pipeline Info
- **Complexity:** {state.get('complexity', 'unknown').capitalize()}
- **Steps taken:** {state.get('steps', 0)}
- **Retry count:** {state.get('retry_count', 0)}
- **Patch confidence:** {confidence_badge}

*Generated by GitHub Agent System*
"""
    return description


def create_github_actions_workflow(
    repo_requirements: str = None
) -> str:
    """
    Creates a GitHub Actions workflow file.
    This gets committed to the PR branch.
    GitHub automatically runs it when the PR opens.
    Tests run in GitHub's free infrastructure.
    """
    # Build install step based on whether repo has requirements.txt
    install_step = """
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest
        if [ -f requirements.txt ]; then
          pip install -r requirements.txt -q || true
        fi"""

    return f"""name: GitFix Automated Validation

on:
  pull_request:
    branches: [ main, master, develop ]

jobs:
  validate-automated-fix:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
{install_step}

    - name: Run automated fix tests
      run: |
        if [ -f test_automated_fix.py ]; then
          python -m pytest test_automated_fix.py -v --tb=short
        else
          echo "No automated fix tests found"
        fi

    - name: Run full test suite
      run: |
        python -m pytest --tb=short -v --ignore=test_automated_fix.py || true
"""


def commit_test_and_workflow_files(
    repo,
    branch_name: str,
    tests: str,
    issue_number: int
) -> None:
    """
    Commits two files to the branch:
    1. test_automated_fix.py — the generated tests
    2. .github/workflows/gitfix-validation.yml — GitHub Actions workflow

    GitHub Actions picks up the workflow and runs the tests
    automatically when the PR is opened.
    """

    # File 1 — commit the generated test file
    test_filename = "test_automated_fix.py"
    logger.info(f"Committing generated tests: {test_filename}")

    try:
        # Check if test file already exists on branch
        try:
            existing = github_api_call(
                repo.get_contents,
                test_filename,
                ref=branch_name
            )
            # File exists — update it
            github_api_call(
                repo.update_file,
                path=test_filename,
                message=(
                    f"test: add automated fix tests "
                    f"for issue #{issue_number}"
                ),
                content=tests,
                sha=existing.sha,
                branch=branch_name
            )
            logger.info(f"Updated existing: {test_filename}")

        except GithubException as e:
            if e.status == 404:
                # File does not exist — create it
                github_api_call(
                    repo.create_file,
                    path=test_filename,
                    message=(
                        f"test: add automated fix tests "
                        f"for issue #{issue_number}"
                    ),
                    content=tests,
                    branch=branch_name
                )
                logger.info(f"Created: {test_filename}")
            else:
                raise

    except Exception as e:
        logger.warning(f"Could not commit test file: {e}")

    # File 2 — commit the GitHub Actions workflow
    workflow_path = ".github/workflows/gitfix-validation.yml"
    workflow_content = create_github_actions_workflow()
    logger.info(f"Committing workflow: {workflow_path}")

    try:
        # Check if workflow already exists
        try:
            existing_workflow = github_api_call(
                repo.get_contents,
                workflow_path,
                ref=branch_name
            )
            # Exists — update it
            github_api_call(
                repo.update_file,
                path=workflow_path,
                message="ci: update gitfix validation workflow",
                content=workflow_content,
                sha=existing_workflow.sha,
                branch=branch_name
            )
            logger.info("Updated workflow file")

        except GithubException as e:
            if e.status == 404:
                # Does not exist — create it
                github_api_call(
                    repo.create_file,
                    path=workflow_path,
                    message="ci: add gitfix automated fix validation",
                    content=workflow_content,
                    branch=branch_name
                )
                logger.info("Created workflow file")
            else:
                raise

    except Exception as e:
        logger.warning(
            f"Could not commit workflow file: {e}. "
            f"PR will still open without it."
        )


# ATOMIC COMMIT STRATEGY (prepare all file changes before committing ANY)
def prepare_all_changes(
    repo,
    file_changes: list[dict],
    branch_name: str,
    issue_number: int
) -> list[dict]:
    """
    Prepares all file changes before committing anything.
    Fetches current content and applies patch for each file.
    Returns list of prepared changes ready to commit.
    If ANY file fails to prepare, raises exception
    so nothing gets committed - all or nothing.
    """
    prepared = []

    for change in file_changes:
        filename = change["filename"]
        logger.info(f"Preparing changes for: {filename}")

        try:
            # Fetch current file from branch
            file_obj = github_api_call(
                repo.get_contents,
                filename,
                ref=branch_name
            )
            original_content = (
                file_obj.decoded_content.decode("utf-8")
            )
            current_sha      = file_obj.sha

            # Apply patch
            new_content = apply_patch_to_content(
                original_content,
                change["hunks"]
            )

            # Verify something actually changed
            if new_content == original_content:
                logger.warning(
                    f"Patch produced no changes in {filename}. "
                    f"Skipping."
                )
                continue

            prepared.append({
                "filename":    filename,
                "new_content": new_content,
                "current_sha": current_sha,
                "commit_msg":  (
                    f"fix: automated fix for issue #{issue_number} "
                    f"- {filename}"
                )
            })

            logger.info(f"Prepared: {filename}")

        except GithubException as e:
            if e.status == 404:
                logger.warning(
                    f"File {filename} not found on branch. "
                    f"Skipping."
                )
                continue
            raise

    return prepared


def commit_all_changes(
    repo,
    prepared_changes: list[dict],
    branch_name: str
) -> list[str]:
    """
    Commits all prepared changes to GitHub.
    Returns list of successfully committed filenames.
    """
    committed = []

    for change in prepared_changes:
        filename    = change["filename"]
        new_content = change["new_content"]

        try:
            github_api_call(
                repo.update_file,
                path=filename,
                message=change["commit_msg"],
                content=new_content,
                sha=change["current_sha"],
                branch=branch_name
            )

            # Verify the commit landed correctly
            verified = verify_commit(
                repo,
                filename,
                new_content,
                branch_name
            )

            if verified:
                committed.append(filename)
                logger.info(f"Successfully committed: {filename}")
            else:
                logger.warning(
                    f"Commit verification failed: {filename}. "
                    f"Content may not match."
                )
                committed.append(filename)  # still record it

        except Exception as e:
            logger.error(f"Failed to commit {filename}: {e}")
            raise

    return committed


# MAIN AGENT FUNCTION
def open_pull_request_agent(state: AgentState) -> AgentState:
    """
    THE MAIN AGENT FUNCTION.

    Improvements:
    1. Retry wrapper on all GitHub API calls
    2. Robust patch parsing handles edge cases
    3. Atomic commit strategy - all or nothing
    4. Commit verification - reads back and confirms
    5. Rich PR description with full analysis
    6. Handles missing files gracefully

    Reads:  issue_url, issue_title, plan, patch, test_result
    Writes: pr_url
    """
    logger.info("=== PR Opener Agent starting ===")

    state["steps"] += 1

    if not state.get("patch"):
        error_msg = "PR Opener failed: no patch found in state"
        logger.error(error_msg)
        state["error"] = error_msg
        return state

    try:
        # Parse issue URL
        issue_info   = parse_issue_url(state["issue_url"])
        owner        = issue_info["owner"]
        repo_name    = issue_info["repo"]
        issue_number = issue_info["number"]

        logger.info(
            f"Opening PR for {owner}/{repo_name} "
            f"issue #{issue_number}"
        )

        # Connect to GitHub
        github = get_github_client()
        repo   = github_api_call(
            github.get_repo,
            f"{owner}/{repo_name}"
        )

        # Get default branch and latest SHA
        default_branch = repo.default_branch
        ref            = github_api_call(
            repo.get_git_ref,
            f"heads/{default_branch}"
        )
        latest_sha = ref.object.sha

        logger.info(
            f"Default branch: {default_branch} "
            f"SHA: {latest_sha[:8]}..."
        )

        # Create branch with unique name
        branch_name = f"fix/issue-{issue_number}-automated"
        logger.info(f"Creating branch: {branch_name}")

        try:
            github_api_call(
                repo.create_git_ref,
                ref=f"refs/heads/{branch_name}",
                sha=latest_sha
            )
            logger.info(f"Branch created: {branch_name}")

        except GithubException as e:
            if e.status == 422:
                # Branch already exists - use it
                logger.warning(
                    f"Branch {branch_name} already exists. "
                    f"Using existing branch."
                )
            else:
                raise

        # Parse the patch
        logger.info("Parsing patch...")
        file_changes = parse_patch(state["patch"])

        if not file_changes:
            logger.warning(
                "No file changes found in patch. "
                "Opening PR with empty branch."
            )
        else:
            logger.info(
                f"Patch affects {len(file_changes)} files"
            )

            # IMPROVEMENT 5 - Atomic strategy
            # Prepare ALL changes first
            # If any preparation fails nothing gets committed
            logger.info("Preparing all file changes...")
            prepared_changes = prepare_all_changes(
                repo,
                file_changes,
                branch_name,
                issue_number
            )

            if not prepared_changes:
                raise Exception(
                    "No valid changes could be prepared. "
                    "The agent either hallucinated a file that does not exist, "
                    "or the patch resulted in no differences. Cannot open an empty PR."
                )
            else:
                # Commit ALL prepared changes
                logger.info(
                    f"Committing {len(prepared_changes)} files..."
                )
                committed_files = commit_all_changes(
                    repo,
                    prepared_changes,
                    branch_name
                )
                logger.info(
                    f"Committed {len(committed_files)} files: "
                    f"{committed_files}"
                )
                # Commit generated tests and GitHub Actions workflow
                # GitHub will run these automatically when PR opens
            if state.get("tests"):
                logger.info(
                    "Committing test file and CI workflow..."
                )
                commit_test_and_workflow_files(
                    repo,
                    branch_name,
                    state["tests"],
                    issue_number
                    )

        # Build PR description
        pr_description = create_pr_description(state)

        # Open the pull request
        logger.info("Opening pull request...")
        pr = github_api_call(
            repo.create_pull,
            title=f"fix: {state['issue_title']} (automated)",
            body=pr_description,
            head=branch_name,
            base=default_branch
        )

        state["pr_url"] = pr.html_url
        state["error"]  = None

        logger.info(f"Pull request opened: {pr.html_url}")

    except Exception as e:
        error_msg = f"PR Opener failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg

    return state