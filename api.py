from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import Optional
from workflow import run_workflow
from state import validate_github_url
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="GitHub Agent System API",
    description="Autonomous bug fixing pipeline",
    version="1.0.0"
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=3)


class FixRequest(BaseModel):
    issue_url: str

class FixResponse(BaseModel):
    success:     bool
    pr_url:      str
    issue_title: str
    complexity:  str
    steps:       int
    error:       Optional[str] = None


@app.get("/")
def root():
    return {"status": "GitHub Agent System is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/fix", response_model=FixResponse)
async def fix_issue(request: FixRequest):
    """
    Main endpoint. Takes a GitHub issue URL.
    Runs the full pipeline and returns the PR URL.
    """
    logger.info(f"Fix request received: {request.issue_url}")

    # Validate URL
    if not validate_github_url(request.issue_url):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid GitHub issue URL. "
                "Expected: https://github.com/owner/repo/issues/42"
            )
        )

    try:
        # Run pipeline in thread pool so it does not
        # block the async event loop
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            executor,
            run_workflow,
            request.issue_url
        )

        return FixResponse(
            success=    final_state.get("pr_url") != "",
            pr_url=     final_state.get("pr_url", ""),
            issue_title=final_state.get("issue_title", ""),
            complexity= final_state.get("complexity", ""),
            steps=      final_state.get("steps", 0),
            error=      final_state.get("error")
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )