import argparse
import sys
import time
from dotenv import load_dotenv

from workflow import run_workflow
from utils.logger import get_logger
from utils.cost_tracker import cost_tracker

load_dotenv()
logger = get_logger(__name__)


def print_banner():
    """Prints a clean startup banner."""
    print("\n" + "=" * 60)
    print("   GITHUB AGENT SYSTEM")
    print("   Autonomous Bug Fixing Pipeline")
    print("=" * 60)

#summary of the run, showing all important fields from final state in a clean format
def print_run_summary(final_state: dict, duration: float): 
    """
    Prints a clean summary of the entire run.
    Shows every important field from final state.
    """
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)

    # Issue details
    print(f"\n Issue:      {final_state.get('issue_title', 'unknown')}")
    print(f" URL:        {final_state.get('issue_url', '')}")

    # Pipeline results
    print(f"\n Complexity:  {final_state.get('complexity', 'unknown')}")
    print(f" Steps taken: {final_state.get('steps', 0)}")
    print(f" Retries:     {final_state.get('retry_count', 0)}")

    # Test results
    test_result = final_state.get('test_result', 'unknown')
    test_icon = "✅" if test_result == "passed" else "❌"
    print(f" Tests:       {test_icon} {test_result}")

    # Final output
    pr_url = final_state.get('pr_url', '')
    if pr_url and 'placeholder' not in pr_url:
        print(f"\n Pull Request: {pr_url}")
    else:
        print(f"\n Pull Request: not created")

    # Error if any
    error = final_state.get('error')
    if error:
        print(f"\n Error:       {error[:100]}...")

    # Timing
    print(f"\n Duration:    {duration:.1f} seconds")

    print("=" * 60)


def validate_args(args):
    """
    Validates command line arguments before running.
    Fails fast with clear messages.
    """
    if not args.issue:
        print("ERROR: --issue URL is required")
        print(
            "Example: python main.py "
            "--issue https://github.com/user/repo/issues/42"
        )
        sys.exit(1)

    if "github.com" not in args.issue:
        print(f"ERROR: URL does not look like a GitHub URL: {args.issue}")
        sys.exit(1)

    if "/issues/" not in args.issue:
        print(
            f"ERROR: URL must point to a GitHub issue: {args.issue}\n"
            f"Expected format: https://github.com/owner/repo/issues/42"
        )
        sys.exit(1)


def main():
    """
    Main entry point for the CLI.
    Parses arguments, runs the pipeline, prints summary.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="GitHub Agent System — Autonomous Bug Fixing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --issue https://github.com/user/repo/issues/42
  python main.py --issue https://github.com/user/repo/issues/42 --verbose
        """
    )

    parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="GitHub issue URL to fix"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs during execution"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but skip opening the real PR"
    )

    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Print startup banner
    print_banner()
    print(f"\n Target issue: {args.issue}")

    if args.dry_run:
        print(" Mode: DRY RUN (no PR will be opened)")

    print("\n Starting pipeline...\n")

    # Record start time
    start_time = time.time()

    try:
        # Run the full pipeline
        final_state = run_workflow(args.issue)

        # Record end time
        duration = time.time() - start_time

        # Print run summary
        print_run_summary(final_state, duration)

        # Print cost summary
        cost_tracker.print_summary()

        # Exit with appropriate code
        # 0 = success, 1 = failure
        if final_state.get("error"):
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nRun interrupted by user.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# To run this script, use the command line:
# python main.py --issue, followed by the GitHub issue URL you want to fix. For example:.