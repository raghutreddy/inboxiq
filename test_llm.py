# test_llm.py - Our first LLM call
# This script proves: Python + OpenAI + API key all work together

import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load the API key from .env file
load_dotenv()

# Step 2: Create the OpenAI client
client = OpenAI()

# Step 3: Send a message to GPT-4o-mini and get a response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say hello and tell me what you are in one sentence."}
    ],
    max_tokens=50
)

# Step 4: Print the response
print(response.choices[0].message.content)