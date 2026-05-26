# action_extractor.py - Extracts action items / to-dos from emails
# Now with retry logic and safe JSON parsing

import os
from dotenv import load_dotenv
from openai import OpenAI
from guardrails import call_with_retry, safe_parse_json

load_dotenv()
client = OpenAI()

def extract_actions(email_text):
    """
    Takes raw email text.
    Returns a list of action items with owners and deadlines.
    Protected by retry logic and safe JSON parsing.
    """

    system_prompt = """You are an action item extraction assistant.
Your job is to read an email and extract every concrete action item.

Rules:
- Only extract REAL action items — things someone needs to DO
- Ignore greetings, pleasantries, and background information
- If there's a deadline mentioned, include it
- If there's a specific person responsible, include them
- If no action items exist, return an empty list

Respond ONLY in this exact JSON format, nothing else:
{
    "action_items": [
        {
            "task": "What needs to be done",
            "owner": "Who should do it (or 'me' if directed at reader)",
            "deadline": "When it's due (or 'none' if not specified)",
            "priority": "high/medium/low"
        }
    ],
    "total_count": 0
}"""

    def make_api_call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract action items from this email:\n\n{email_text}"}
            ],
            max_tokens=400,
            temperature=0.1
        )

    response = call_with_retry(make_api_call, step_name="extract_actions")
    result_text = response.choices[0].message.content

    result, parse_success = safe_parse_json(result_text)

    return result, response