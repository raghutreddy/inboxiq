# action_extractor.py - Extracts action items / to-dos from emails
# Works on ALL email categories, not just urgent ones

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def extract_actions(email_text):
    """
    Takes raw email text.
    Returns a list of action items with owners and deadlines.
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract action items from this email:\n\n{email_text}"}
        ],
        max_tokens=400,
        temperature=0.1
    )

    result_text = response.choices[0].message.content

    try:
        result = json.loads(result_text)
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": result_text}


# ---- Test ----
if __name__ == "__main__":

    # Email with multiple action items
    test_email = """
    From: boss@company.com
    Subject: URGENT: Client presentation moved to tomorrow 9 AM

    Hi team,
    The client presentation has been moved up to tomorrow morning at 9 AM.
    Please have all slides finalized by tonight. This is our biggest account
    and we cannot afford any mistakes. Send me your sections by 6 PM today.
    
    Also, Sarah — please book the large conference room for 8:30 AM so we 
    can do a dry run before the client arrives.
    
    Mike, make sure the demo environment is working and tested by end of day.
    """

    result = extract_actions(test_email)
    print("=" * 50)
    print("EXTRACTED ACTION ITEMS:")
    print("=" * 50)
    print(json.dumps(result, indent=2))