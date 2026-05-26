# pipeline.py - The InboxIQ Pipeline with Cost Tracking
# Chains: Classify → Draft Reply (if needed) → Extract Actions
# Now tracks every API call's token usage and cost

import json
from classifier import classify_email
from reply_drafter import draft_reply
from action_extractor import extract_actions
from cost_tracker import PipelineTracker


MODEL = "gpt-4o-mini"


def process_email(email_text, tracker):
    """
    The main pipeline. Takes a raw email + tracker,
    runs all three AI steps, logs costs, returns complete result.
    """

    result = {
        "classification": None,
        "reply_draft": None,
        "action_items": None
    }

    # ---- Step 1: Classify ----
    print("  [1/3] Classifying email...")
    classification, cls_response = classify_email(email_text)
    tracker.log_call("classify", MODEL, cls_response)
    result["classification"] = classification

    # ---- Step 2: Draft reply (only if needed) ----
    category = classification.get("category", "")
    urgency = classification.get("urgency_score", 1)

    if category in ["URGENT_ACTION", "NEEDS_REPLY"]:
        print("  [2/3] Drafting reply...")
        reply, reply_response = draft_reply(email_text, category, urgency)
        tracker.log_call("reply_draft", MODEL, reply_response)
        result["reply_draft"] = reply
    else:
        print("  [2/3] No reply needed — skipping.")

    # ---- Step 3: Extract action items (always) ----
    print("  [3/3] Extracting action items...")
    actions, act_response = extract_actions(email_text)
    tracker.log_call("extract_actions", MODEL, act_response)
    result["action_items"] = actions

    return result


def display_result(label, result):
    """Pretty-prints the full pipeline result."""

    print(f"\n{'='*60}")
    print(f"  EMAIL: {label}")
    print(f"{'='*60}")

    # Classification
    c = result["classification"]
    print(f"\n  📧 Category:    {c.get('category', 'N/A')}")
    print(f"  🔥 Urgency:     {c.get('urgency_score', 'N/A')}/5")
    print(f"  💭 Reasoning:   {c.get('reasoning', 'N/A')}")
    print(f"  👉 Suggestion:  {c.get('suggested_action', 'N/A')}")

    # Reply draft
    if result["reply_draft"]:
        print(f"\n  ✉️  DRAFT REPLY:")
        print(f"  {'-'*40}")
        for line in result["reply_draft"].split("\n"):
            print(f"    {line}")
    else:
        print(f"\n  ✉️  No reply drafted (not needed)")

    # Action items
    actions = result["action_items"]
    items = actions.get("action_items", [])
    if items:
        print(f"\n  ✅ ACTION ITEMS ({len(items)}):")
        print(f"  {'-'*40}")
        for i, item in enumerate(items, 1):
            print(f"    {i}. {item['task']}")
            print(f"       Owner: {item['owner']} | Deadline: {item['deadline']} | Priority: {item['priority']}")
    else:
        print(f"\n  ✅ No action items found.")

    print()


# ---- Run the full pipeline ----

if __name__ == "__main__":

    emails = [
        (
            "Urgent from boss",
            """
            From: boss@company.com
            Subject: URGENT: Client presentation moved to tomorrow 9 AM
            
            Hi team,
            The client presentation has been moved up to tomorrow morning at 9 AM.
            Please have all slides finalized by tonight. Send me your sections by 
            6 PM today. Sarah, book the large conference room for 8:30 AM.
            """
        ),
        (
            "Colleague asking a question",
            """
            From: colleague@company.com
            Subject: Quick question about the API docs
            
            Hey,
            I was looking at the API documentation and I'm confused about the 
            authentication flow. Could you explain how the OAuth token refresh 
            works? No rush, just whenever you have a few minutes.
            Thanks!
            """
        ),
        (
            "Holiday notice",
            """
            From: hr@company.com
            Subject: Office closed Monday
            
            Hi everyone,
            Just a reminder that the office will be closed next Monday for the 
            national holiday. Enjoy your long weekend!
            """
        ),
        (
            "Spam email",
            """
            From: deals@megastore-offers.com
            Subject: 🔥 90% OFF EVERYTHING - CLICK NOW!!!
            
            CONGRATULATIONS! You've been selected for our EXCLUSIVE sale!
            Click here NOW before this offer expires! Limited time only!
            """
        ),
    ]

    # Create a tracker for this entire run
    tracker = PipelineTracker()

    print("\n🚀 InboxIQ Pipeline - Processing emails...\n")

    for label, email_text in emails:
        result = process_email(email_text, tracker)
        display_result(label, result)

    # Print the cost report at the end
    tracker.print_summary()

    print("✅ All emails processed.\n")