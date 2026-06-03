# llm_router.py - Multi-Provider LLM Router for InboxIQ
# Routes requests to OpenAI or Anthropic based on complexity
# Falls back to the other provider if one fails

import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from guardrails import call_with_retry

load_dotenv()

# Initialize both clients
openai_client = OpenAI()
anthropic_client = Anthropic()

# Model configurations
MODELS = {
    "openai_cheap": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60
    },
    "anthropic_smart": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "cost_per_1m_input": 0.80,
        "cost_per_1m_output": 4.00
    }
}
# Default routing
PRIMARY_MODEL = "openai_cheap"
FALLBACK_MODEL = "anthropic_smart"


def estimate_complexity(email_text):
    """
    Estimates email complexity based on simple heuristics.
    Returns 'simple' or 'complex'.
    
    Simple emails: short, clear intent, common patterns
    Complex emails: long, multiple topics, ambiguous intent
    """

    text = email_text.lower()
    complexity_score = 0

    # Length factor
    if len(text) > 1000:
        complexity_score += 2
    elif len(text) > 500:
        complexity_score += 1

    # Multiple questions = more complex
    question_count = text.count("?")
    if question_count >= 3:
        complexity_score += 2
    elif question_count >= 2:
        complexity_score += 1

    # Multiple people mentioned = more complex
    people_indicators = ["@", "dear", "hi ", "hey ", "cc:", "bcc:"]
    people_count = sum(1 for p in people_indicators if p in text)
    if people_count >= 3:
        complexity_score += 1

    # Urgency keywords suggest straightforward classification
    urgent_keywords = ["urgent", "asap", "immediately", "deadline", "critical"]
    if any(k in text for k in urgent_keywords):
        complexity_score -= 1  # Urgent emails are usually simple to classify

    # Ambiguity markers
    ambiguous_keywords = ["maybe", "possibly", "not sure", "depends", "either"]
    if any(k in text for k in ambiguous_keywords):
        complexity_score += 2

    return "complex" if complexity_score >= 3 else "simple"


def route_request(email_text):
    """
    Decides which model to use based on email complexity.
    Returns the model key from MODELS dict.
    """
    complexity = estimate_complexity(email_text)

    if complexity == "complex":
        return FALLBACK_MODEL, complexity  # Use smarter model
    else:
        return PRIMARY_MODEL, complexity   # Use cheaper model


def call_llm(system_prompt, user_prompt, model_key, max_tokens=200, temperature=0.1):
    """
    Unified LLM call that works with both OpenAI and Anthropic.
    Handles provider-specific API differences.
    Falls back to other provider on failure.
    Returns (response_text, model_used, usage_dict).
    """

    model_config = MODELS[model_key]
    provider = model_config["provider"]
    model_name = model_config["model"]

    # Try primary model
    try:
        text, usage = _call_provider(
            provider, model_name, system_prompt, user_prompt, 
            max_tokens, temperature
        )
        return text, model_key, usage

    except Exception as primary_error:
        print(f"    ⚠️  {provider} ({model_name}) failed: {str(primary_error)[:80]}")

        # Determine fallback
        fallback_key = FALLBACK_MODEL if model_key == PRIMARY_MODEL else PRIMARY_MODEL
        fallback_config = MODELS[fallback_key]
        print(f"    🔄 Falling back to {fallback_config['provider']} ({fallback_config['model']})...")

        try:
            text, usage = _call_provider(
                fallback_config["provider"], fallback_config["model"],
                system_prompt, user_prompt, max_tokens, temperature
            )
            return text, fallback_key, usage

        except Exception as fallback_error:
            print(f"    ❌ Fallback also failed: {str(fallback_error)[:80]}")
            raise fallback_error


def _call_provider(provider, model_name, system_prompt, user_prompt, max_tokens, temperature):
    """
    Makes the actual API call to a specific provider.
    Returns (response_text, usage_dict).
    """

    if provider == "openai":
        def make_call():
            return openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

        response = call_with_retry(make_call, step_name=f"openai/{model_name}")
        text = response.choices[0].message.content
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model_name,
            "provider": "openai"
        }
        return text, usage

    elif provider == "anthropic":
        def make_call():
            return anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                system=system_prompt,
                temperature=temperature
            )

        response = call_with_retry(make_call, step_name=f"anthropic/{model_name}")
        text = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model_name,
            "provider": "anthropic"
        }
        return text, usage

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---- Test ----
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  🔀 LLM Router Test")
    print("="*60)

    # Test 1: Simple email → should route to OpenAI (cheap)
    simple_email = "Meeting tomorrow at 3 PM in room 5B. Please confirm."
    model_key, complexity = route_request(simple_email)
    print(f"\n  Simple email → complexity: {complexity} → model: {model_key}")

    # Test 2: Complex email → should route to Anthropic (smart)
    complex_email = """
    I'm not sure if we should proceed with the vendor or maybe wait for 
    the quarterly review? There are possibly some budget concerns, and 
    I'm wondering if the timeline depends on the hiring decisions? Also, 
    could you check with Sarah about the compliance review, and maybe 
    ask Tom if the infrastructure team has capacity? What do you think 
    about the risk assessment? Should we possibly delay?
    """
    model_key, complexity = route_request(complex_email)
    print(f"  Complex email → complexity: {complexity} → model: {model_key}")

    # Test 3: Actually call both providers
    print(f"\n  Testing OpenAI call...")
    text, model_used, usage = call_llm(
        system_prompt="Respond in exactly 5 words.",
        user_prompt="Say hello.",
        model_key="openai_cheap"
    )
    print(f"  Response: {text}")
    print(f"  Model: {model_used} | Tokens: {usage['input_tokens']}+{usage['output_tokens']}")

    print(f"\n  Testing Anthropic call...")
    text, model_used, usage = call_llm(
        system_prompt="Respond in exactly 5 words.",
        user_prompt="Say hello.",
        model_key="anthropic_smart"
    )
    print(f"  Response: {text}")
    print(f"  Model: {model_used} | Tokens: {usage['input_tokens']}+{usage['output_tokens']}")

    print(f"\n{'='*60}")
    print("  ✅ Both providers working!")
    print(f"{'='*60}\n")