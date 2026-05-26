# classifier.py - Email Classification Engine for InboxIQ
# Now with retry logic, safe JSON parsing, and output validation

import os
from dotenv import load_dotenv
from openai import OpenAI
from guardrails import call_with_retry, safe_parse_json, validate_classification

load_dotenv()
client = OpenAI()

def classify_email(email_text):
    """
    Takes raw email text as input.
    Returns a classification with category, urgency, and reasoning.
    Now protected by retry logic and output validation.
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

    # Wrap API call in retry logic
    def make_api_call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this email:\n\n{email_text}"}
            ],
            max_tokens=200,
            temperature=0.1
        )

    response = call_with_retry(make_api_call, step_name="classify")
    result_text = response.choices[0].message.content

    # Safe JSON parsing (handles markdown fences, extra text, etc.)
    result, parse_success = safe_parse_json(result_text)

    # Validate the classification output
    if parse_success:
        is_valid, result, error_msg = validate_classification(result)
        if not is_valid:
            print(f"    ⚠️  Classification validation warning: {error_msg}")

    return result, response