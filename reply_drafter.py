# reply_drafter.py - Drafts professional email replies
# Now with retry logic for reliability

import os
from dotenv import load_dotenv
from openai import OpenAI
from guardrails import call_with_retry

load_dotenv()
client = OpenAI()

def draft_reply(email_text, category, urgency_score):
    """
    Takes the original email + classification results.
    Returns a professional reply draft.
    Protected by retry logic.
    """

    if urgency_score >= 4:
        tone_instruction = "Use a prompt, action-oriented tone. Show urgency without panic."
    elif urgency_score >= 2:
        tone_instruction = "Use a professional, helpful tone. Be clear and courteous."
    else:
        tone_instruction = "Use a casual, friendly tone. Keep it brief."

    system_prompt = f"""You are a professional email reply assistant.
Your job is to draft a reply to the email below.

Rules:
- Keep it concise (3-5 sentences max)
- Be professional but natural — not robotic
- {tone_instruction}
- If the email asks for something, acknowledge it clearly
- If there's a deadline, confirm you're aware of it
- Never make up facts or commitments the user didn't authorize
- Sign off with just "Best regards" (no name — user will add their own)

Respond ONLY with the reply text. No subject line, no "Here's a draft", no explanation.
Just the reply body itself, ready to send."""

    def make_api_call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Draft a reply to this email:\n\n{email_text}"}
            ],
            max_tokens=300,
            temperature=0.4
        )

    response = call_with_retry(make_api_call, step_name="reply_draft")
    return response.choices[0].message.content, response