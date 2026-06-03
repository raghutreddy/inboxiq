# classifier.py - Email Classification Engine for InboxIQ
# Now uses the LLM Router for multi-provider support

from llm_router import call_llm, route_request
from guardrails import safe_parse_json, validate_classification


def classify_email(email_text):
    """
    Takes raw email text as input.
    Routes to the best model based on complexity.
    Returns classification result and usage stats.
    """

    system_prompt = """You are an expert email triage assistant. 
Your job is to classify emails into exactly ONE of these categories:

1. URGENT_ACTION - Requires immediate response or action (deadlines, emergencies, critical requests)
2. NEEDS_REPLY - Requires a response but not urgent (questions, requests, follow-ups)
3. FYI - Informational only, no response needed (newsletters, announcements, updates)
4. MEETING - Meeting invitations, schedule changes, calendar-related
5. SPAM - Promotional, marketing, or irrelevant emails

Respond ONLY in this exact JSON format, nothing else:
{
    "category": "CATEGORY_NAME",
    "urgency_score": 1-5,
    "reasoning": "One sentence explaining why",
    "suggested_action": "What the user should do"
}"""

    # Route to the best model based on email complexity
    model_key, complexity = route_request(email_text)

    # Make the call via the router
    result_text, model_used, usage = call_llm(
        system_prompt=system_prompt,
        user_prompt=f"Classify this email:\n\n{email_text}",
        model_key=model_key,
        max_tokens=200,
        temperature=0.1
    )

    # Safe JSON parsing
    result, parse_success = safe_parse_json(result_text)

    # Validate output
    if parse_success:
        is_valid, result, error_msg = validate_classification(result)
        if not is_valid:
            print(f"    ⚠️  Classification validation warning: {error_msg}")

    # Add routing info to usage
    usage["complexity"] = complexity
    usage["model_used"] = model_used

    return result, usage