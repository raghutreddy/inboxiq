# batch_processor.py - Processes multiple emails from a JSON file
# Generates a complete triage report with cost summary

import json
import time
from pipeline import process_email, display_result
from cost_tracker import PipelineTracker


def load_emails(filepath):
    """Load emails from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_batch(filepath):
    """
    Processes all emails in a JSON file through the pipeline.
    Returns results and prints a complete report.
    """

    # Load emails
    emails = load_emails(filepath)
    print(f"\n🚀 InboxIQ Batch Processor")
    print(f"   Loaded {len(emails)} emails from {filepath}\n")

    # Create tracker for entire batch
    tracker = PipelineTracker()
    results = []

    # Process each email
    for i, email in enumerate(emails, 1):
        # Combine subject and body into full email text
        email_text = f"From: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"
        label = f"[{email['id']}] {email['subject'][:40]}..."

        print(f"Processing email {i}/{len(emails)}...")
        start = time.time()

        result = process_email(email_text, tracker)
        elapsed = round(time.time() - start, 2)

        result["email_id"] = email["id"]
        result["processing_time"] = elapsed
        results.append(result)

        display_result(label, result)

    # ---- Batch Summary ----
    print(f"\n{'='*60}")
    print(f"  📊 BATCH SUMMARY")
    print(f"{'='*60}")

    # Count categories
    categories = {}
    for r in results:
        cat = r["classification"].get("category", "UNKNOWN")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  Emails processed: {len(results)}")
    print(f"\n  Category breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat:20s}: {count}")

    replies_drafted = sum(1 for r in results if r["reply_draft"] is not None)
    total_actions = sum(
        len(r["action_items"].get("action_items", []))
        for r in results
    )
    avg_time = round(sum(r["processing_time"] for r in results) / len(results), 2)

    print(f"\n  Replies drafted:  {replies_drafted}/{len(results)}")
    print(f"  Action items found: {total_actions}")
    print(f"  Avg time per email: {avg_time}s")

    # Print cost report
    tracker.print_summary()

    # Calculate per-email cost
    summary = tracker.get_summary()
    per_email = round(summary["total_cost_usd"] / len(results), 6)
    print(f"  💰 Cost per email:  ${per_email:.6f}")
    print(f"\n✅ Batch complete.\n")

    return results


if __name__ == "__main__":
    run_batch("sample_emails.json")