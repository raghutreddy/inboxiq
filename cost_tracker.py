# cost_tracker.py - Tracks token usage and costs for every API call
# Central utility used by all pipeline components

import time

# GPT-4o-mini pricing (per 1 million tokens) - as of 2025
PRICING = {
    "gpt-4o-mini": {
        "input": 0.15,    # $0.15 per 1M input tokens
        "output": 0.60    # $0.60 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.50,    # $2.50 per 1M input tokens
        "output": 10.00   # $10.00 per 1M output tokens
    }
}


def calculate_cost(model, input_tokens, output_tokens):
    """
    Calculates the dollar cost of an API call.
    Returns cost in dollars (e.g., 0.000045)
    """
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6)
    }


class PipelineTracker:
    """
    Tracks costs and timing across an entire pipeline run.
    Collects stats from every API call, then generates a summary.
    """

    def __init__(self):
        self.calls = []
        self.start_time = time.time()

    def log_call(self, step_name, model, response):
        """
        Logs one API call's usage. 
        Pass the raw OpenAI response object — it contains token counts.
        """
        usage = response.usage
        cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        entry = {
            "step": step_name,
            "model": model,
            **cost
        }
        self.calls.append(entry)
        return cost

    def get_summary(self):
        """Returns a complete cost summary for the pipeline run."""
        total_input = sum(c["input_tokens"] for c in self.calls)
        total_output = sum(c["output_tokens"] for c in self.calls)
        total_cost = sum(c["total_cost"] for c in self.calls)
        elapsed = round(time.time() - self.start_time, 2)

        return {
            "total_api_calls": len(self.calls),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "elapsed_seconds": elapsed,
            "per_call_breakdown": self.calls
        }

    def print_summary(self):
        """Pretty-prints the cost summary."""
        s = self.get_summary()

        print(f"\n{'='*60}")
        print(f"  💰 COST REPORT")
        print(f"{'='*60}")
        print(f"  API calls made:     {s['total_api_calls']}")
        print(f"  Input tokens:       {s['total_input_tokens']:,}")
        print(f"  Output tokens:      {s['total_output_tokens']:,}")
        print(f"  Total tokens:       {s['total_tokens']:,}")
        print(f"  Total cost:         ${s['total_cost_usd']:.4f}")
        print(f"  Time elapsed:       {s['elapsed_seconds']}s")

        if s['total_api_calls'] > 0:
            avg_cost = s['total_cost_usd'] / s['total_api_calls']
            print(f"  Avg cost per call:  ${avg_cost:.6f}")

        print(f"\n  Per-step breakdown:")
        print(f"  {'-'*50}")
        for call in self.calls:
            print(f"    {call['step']:20s} | {call['total_tokens']:5d} tokens | ${call['total_cost']:.6f}")
        print()