# Save this as verify.py temporarily
from dotenv import load_dotenv
import os
from state import AgentState, get_initial_state

load_dotenv()

# Check env vars loaded
print("GitHub Token:", os.getenv("GITHUB_TOKEN")[:10] + "...")
print("Gemini Key:", os.getenv("GOOGLE_API_KEY")[:10] + "...")

# Check state works
state = get_initial_state("https://github.com/user/repo/issues/1")
print("State created:", state["issue_url"])
print("Retry count:", state["retry_count"])
print("Steps:", state["steps"])
print("\nPhase 1 complete!")
