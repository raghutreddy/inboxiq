# guardrails.py - Input validation, output validation, and retry logic
# The safety net that makes InboxIQ production-ready

import time
import json
from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError
)


# ---- CONFIGURATION ----

MAX_EMAIL_LENGTH = 10000       # characters (~2500 tokens)
MIN_EMAIL_LENGTH = 10          # reject empty/tiny emails
MAX_RETRIES = 3                # retry failed API calls up to 3 times
RETRY_BASE_DELAY = 2           # seconds before first retry
VALID_CATEGORIES = [
    "URGENT_ACTION",
    "NEEDS_REPLY", 
    "FYI",
    "MEETING",
    "SPAM"
]


# ---- INPUT VALIDATION ----

def validate_email_input(email_text):
    """
    Validates email text before sending to the API.
    Returns (is_valid, error_message).
    Catches problems BEFORE they cost you money.
    """

    # Check if email is None or not a string
    if email_text is None:
        return False, "Email text is None"
    
    if not isinstance(email_text, str):
        return False, f"Email text must be a string, got {type(email_text).__name__}"

    # Strip whitespace and check length
    cleaned = email_text.strip()

    if len(cleaned) < MIN_EMAIL_LENGTH:
        return False, f"Email too short ({len(cleaned)} chars). Minimum: {MIN_EMAIL_LENGTH}"

    if len(cleaned) > MAX_EMAIL_LENGTH:
        return False, f"Email too long ({len(cleaned)} chars). Maximum: {MAX_EMAIL_LENGTH}"

    return True, "Valid"


# ---- OUTPUT VALIDATION ----

def validate_classification(result):
    """
    Validates that the classifier's output is well-formed.
    Returns (is_valid, cleaned_result, error_message).
    """

    # Check if it's a dictionary
    if not isinstance(result, dict):
        return False, result, "Result is not a dictionary"

    # Check for required fields
    required_fields = ["category", "urgency_score", "reasoning", "suggested_action"]
    missing = [f for f in required_fields if f not in result]
    if missing:
        return False, result, f"Missing fields: {missing}"

    # Check category is valid
    if result["category"] not in VALID_CATEGORIES:
        return False, result, f"Invalid category: {result['category']}. Must be one of {VALID_CATEGORIES}"

    # Check urgency score is in range
    urgency = result.get("urgency_score", 0)
    if not isinstance(urgency, (int, float)) or urgency < 1 or urgency > 5:
        # Auto-fix: clamp to valid range
        result["urgency_score"] = max(1, min(5, int(urgency) if isinstance(urgency, (int, float)) else 3))

    return True, result, "Valid"


# ---- RETRY LOGIC ----

def call_with_retry(api_call_func, step_name="api_call"):
    """
    Wraps any API call with retry logic and exponential backoff.
    
    api_call_func: a function that takes no arguments and makes the API call
    Returns: the API response
    Raises: the last exception if all retries fail
    """

    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = api_call_func()
            return response

        except RateLimitError as e:
            last_exception = e
            wait_time = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 2, 4, 8 seconds
            print(f"    ⚠️  [{step_name}] Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                  f"Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APITimeoutError as e:
            last_exception = e
            wait_time = RETRY_BASE_DELAY * attempt  # 2, 4, 6 seconds
            print(f"    ⚠️  [{step_name}] Timeout (attempt {attempt}/{MAX_RETRIES}). "
                  f"Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIConnectionError as e:
            last_exception = e
            wait_time = RETRY_BASE_DELAY * attempt
            print(f"    ⚠️  [{step_name}] Connection error (attempt {attempt}/{MAX_RETRIES}). "
                  f"Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIError as e:
            last_exception = e
            if e.status_code and e.status_code >= 500:
                # Server error — retry
                wait_time = RETRY_BASE_DELAY * attempt
                print(f"    ⚠️  [{step_name}] Server error {e.status_code} "
                      f"(attempt {attempt}/{MAX_RETRIES}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Client error (400, 401, etc.) — don't retry, it won't help
                raise e

    # All retries exhausted
    print(f"    ❌ [{step_name}] All {MAX_RETRIES} retries failed.")
    raise last_exception


# ---- SAFE JSON PARSING ----

def safe_parse_json(text):
    """
    Attempts to parse JSON from AI response.
    Handles common issues: markdown fences, extra text before/after JSON.
    Returns (parsed_dict, success_bool).
    """

    # Try direct parse first
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code fences (```json ... ```)
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        try:
            return json.loads(cleaned), True
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end]), True
        except json.JSONDecodeError:
            pass

    # All parsing attempts failed
    return {"error": "Failed to parse JSON", "raw_response": text[:200]}, False