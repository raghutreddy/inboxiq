# classifier.py - Email Classification Engine for InboxIQ
# Takes an email and classifies it into one of 5 categories

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables and create client
load_dotenv()
client = OpenAI()

def classify_email(email_text):
    """
    Takes raw email text as input.
    Returns a classification with category, urgency, and reasoning.
    """

    # This is the SYSTEM PROMPT - instructions that tell the AI how to behave
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

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this email:\n\n{email_text}"}
        ],
        max_tokens=200,
        temperature=0.1
    )

    # Extract the text response
    result_text = response.choices[0].message.content

    # Parse the JSON string into a Python dictionary
    try:
        result = json.loads(result_text)
        return result, response
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": result_text}, response

# ---- Test with sample emails ----

if __name__ == "__main__":

    # Test Email 1: Urgent
    email1 = """
    From: boss@company.com
    Subject: URGENT: Client presentation moved to tomorrow 9 AM
    
    Hi team,
    The client presentation has been moved up to tomorrow morning at 9 AM. 
    Please have all slides finalized by tonight. This is our biggest account 
    and we cannot afford any mistakes. Send me your sections by 6 PM today.
    """

    # Test Email 2: FYI
    email2 = """
    From: hr@company.com
    Subject: Office closed on Monday for holiday
    
    Hi everyone,
    Just a reminder that the office will be closed next Monday for the 
    national holiday. Enjoy your long weekend!
    """

    # Test Email 3: Spam
    email3 = """
    From: deals@superstore-offers.com
    Subject: 🔥 MASSIVE SALE - 90% OFF EVERYTHING!!!
    
    CONGRATULATIONS! You've been selected for our EXCLUSIVE VIP sale! 
    Click here NOW to claim your discount before it expires! 
    Limited time only! Act fast!
    """

    # Run classification on all three
    test_emails = [
        ("URGENT email from boss", email1),
        ("FYI holiday notice", email2),
        ("SPAM promotional", email3)
    ]

    for label, email in test_emails:
        print(f"\n{'='*50}")
        print(f"Testing: {label}")
        print(f"{'='*50}")
        result = classify_email(email)
        print(json.dumps(result, indent=2))