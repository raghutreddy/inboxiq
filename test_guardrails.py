# test_guardrails.py - Tests that guardrails catch bad inputs
# Run this to prove InboxIQ doesn't crash on edge cases

from pipeline import process_email
from cost_tracker import PipelineTracker


def test_edge_cases():
    """Tests the pipeline with deliberately bad/weird inputs."""

    tracker = PipelineTracker()

    test_cases = [
        ("Empty string", ""),
        ("Too short", "Hi"),
        ("None value", None),
        ("Number instead of string", 12345),
        ("Normal email (should work)", """
            From: boss@company.com
            Subject: Quick update needed
            
            Hey, can you send me the latest sales numbers? 
            Need them for the meeting this afternoon. Thanks!
        """),
        ("Very long email (should be rejected)", "A" * 15000),
    ]

    print(f"\n{'='*60}")
    print(f"  🛡️  GUARDRAILS TEST SUITE")
    print(f"{'='*60}\n")

    for label, email_input in test_cases:
        print(f"\n  Test: {label}")
        print(f"  {'-'*40}")

        try:
            result = process_email(email_input, tracker)
            category = result["classification"].get("category", "N/A")
            error = result["classification"].get("error", None)

            if error:
                print(f"  Result: ❌ Rejected — {error}")
            else:
                print(f"  Result: ✅ Processed — {category}")
        except Exception as e:
            print(f"  Result: 💥 CRASHED — {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Tests complete.")
    tracker.print_summary()


if __name__ == "__main__":
    test_edge_cases()