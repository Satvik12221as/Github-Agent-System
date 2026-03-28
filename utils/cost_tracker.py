from utils.logger import get_logger

logger = get_logger(__name__)

# Claude Sonnet 4.5 pricing
COST_PER_1000_INPUT_TOKENS  = 0.003    # $3 per 1M tokens
COST_PER_1000_OUTPUT_TOKENS = 0.015    # $15 per 1M tokens


class CostTracker:
    """
    Tracks token usage and estimated cost across all agents.
    Each agent calls track() after every LLM call.
    At the end of the run, print_summary() shows the total.
    """

    def __init__(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.calls_by_agent      = {}

    def track(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Records token usage for one LLM call.
        Call this after every llm.invoke() in any agent.
        """
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens

        if agent_name not in self.calls_by_agent:
            self.calls_by_agent[agent_name] = {
                "calls":         0,
                "input_tokens":  0,
                "output_tokens": 0
            }

        self.calls_by_agent[agent_name]["calls"]         += 1
        self.calls_by_agent[agent_name]["input_tokens"]  += input_tokens
        self.calls_by_agent[agent_name]["output_tokens"] += output_tokens

    def get_total_cost(self) -> float:
        """Returns estimated total cost in USD."""
        input_cost  = (
            self.total_input_tokens / 1000
        ) * COST_PER_1000_INPUT_TOKENS

        output_cost = (
            self.total_output_tokens / 1000
        ) * COST_PER_1000_OUTPUT_TOKENS

        return input_cost + output_cost

    def print_summary(self):
        """Prints a clean cost summary to the terminal."""
        print("\n" + "=" * 50)
        print("COST SUMMARY")
        print("=" * 50)

        for agent, data in self.calls_by_agent.items():
            print(
                f"{agent:<20} "
                f"calls={data['calls']} "
                f"in={data['input_tokens']} "
                f"out={data['output_tokens']}"
            )

        print("-" * 50)
        print(f"Total input tokens:  {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(
            f"Estimated cost:      "
            f"${self.get_total_cost():.4f} USD"
        )
        print("=" * 50)


# Single global instance used by all agents
cost_tracker = CostTracker()