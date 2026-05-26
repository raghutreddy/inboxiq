# eval_runner.py - Automated evaluation suite for InboxIQ
# Runs test emails through the classifier and measures accuracy
# This file is what separates a demo from a production-grade project

import json
import time
from classifier import classify_email
from cost_tracker import PipelineTracker


def load_eval_dataset(filepath):
    """Load eval dataset with ground truth labels."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(filepath="eval_dataset.json"):
    """
    Runs all eval emails through the classifier.
    Compares predictions to ground truth.
    Returns detailed accuracy metrics.
    """

    dataset = load_eval_dataset(filepath)
    tracker = PipelineTracker()

    print(f"\n{'='*60}")
    print(f"  🧪 InboxIQ EVALUATION SUITE")
    print(f"  Dataset: {filepath} ({len(dataset)} test cases)")
    print(f"{'='*60}\n")

    results = []
    correct = 0
    total = 0
    json_parse_success = 0
    json_parse_fail = 0

    # Per-category tracking
    category_stats = {}

    for i, email in enumerate(dataset, 1):
        email_text = f"From: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"
        expected = email["expected_category"]

        print(f"  [{i}/{len(dataset)}] {email['id']}: {email['subject'][:45]}...")

        # Run classifier
        try:
            classification, response = classify_email(email_text)
            tracker.log_call("eval_classify", "gpt-4o-mini", response)

            # Check if JSON parsed correctly
            if "error" in classification:
                json_parse_fail += 1
                predicted = "PARSE_ERROR"
                print(f"           ❌ JSON parse failed")
            else:
                json_parse_success += 1
                predicted = classification.get("category", "UNKNOWN")
        except Exception as e:
            json_parse_fail += 1
            predicted = "ERROR"
            print(f"           ❌ API error: {str(e)[:50]}")

        # Compare prediction to ground truth
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
            print(f"           ✅ {predicted} (correct)")
        else:
            print(f"           ❌ {predicted} (expected: {expected})")

        total += 1

        # Track per-category stats
        if expected not in category_stats:
            category_stats[expected] = {"correct": 0, "total": 0, "predictions": []}
        category_stats[expected]["total"] += 1
        category_stats[expected]["predictions"].append(predicted)
        if is_correct:
            category_stats[expected]["correct"] += 1

        results.append({
            "id": email["id"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "reasoning": classification.get("reasoning", "N/A") if isinstance(classification, dict) else "N/A"
        })

    # ---- Calculate Metrics ----
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0
    parse_rate = round((json_parse_success / total) * 100, 1) if total > 0 else 0

    # ---- Print Results ----
    print(f"\n{'='*60}")
    print(f"  📊 EVALUATION RESULTS")
    print(f"{'='*60}")

    print(f"\n  Overall Accuracy:    {correct}/{total} ({accuracy}%)")
    print(f"  JSON Parse Rate:     {json_parse_success}/{total} ({parse_rate}%)")

    # Per-category accuracy
    print(f"\n  Per-Category Breakdown:")
    print(f"  {'-'*50}")
    print(f"  {'Category':20s} | {'Correct':>7s} | {'Total':>5s} | {'Accuracy':>8s}")
    print(f"  {'-'*50}")

    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        cat_accuracy = round((stats["correct"] / stats["total"]) * 100, 1)
        print(f"  {cat:20s} | {stats['correct']:>7d} | {stats['total']:>5d} | {cat_accuracy:>7.1f}%")

    # Misclassifications
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n  ❌ Misclassifications ({len(misses)}):")
        print(f"  {'-'*50}")
        for m in misses:
            print(f"    {m['id']}: predicted {m['predicted']}, expected {m['expected']}")
            print(f"      Reasoning: {m['reasoning']}")
    else:
        print(f"\n  🎯 Perfect score — zero misclassifications!")

    # Cost report
    tracker.print_summary()

    # Summary dict for programmatic use
    summary = {
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "correct": correct,
        "total": total,
        "per_category": {
            cat: round((stats["correct"] / stats["total"]) * 100, 1)
            for cat, stats in category_stats.items()
        },
        "misclassifications": misses,
        "cost": tracker.get_summary()
    }

    return summary


if __name__ == "__main__":
    summary = run_eval()

    # Print the resume-ready one-liner
    cost = summary["cost"]["total_cost_usd"]
    print(f"\n  📝 RESUME BULLET:")
    print(f"  {'-'*50}")
    print(f"  \"Evaluated email classifier on {summary['total']}-case test suite:")
    print(f"   {summary['accuracy']}% accuracy, {summary['parse_rate']}% JSON parse rate,")
    print(f"   ${cost:.4f} total eval cost.\"\n")