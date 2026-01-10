#!/usr/bin/env python3
"""
RAG Knowledge Assistant for Insurellm (OpenRouter Version)
Converted from Jupyter notebook to standalone Python script
"""

import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from openai import OpenAI

# Setting up OpenRouter
load_dotenv(override=True)

# OpenRouter configuration
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if openrouter_api_key:
    print(f"OpenRouter API Key exists and begins {openrouter_api_key[:8]}")
else:
    print("OpenRouter API Key not set")

MODEL = "deepseek/deepseek-chat"

# Initialize OpenAI client with OpenRouter endpoint
openai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

print(f"Using model: {MODEL}")

# Load knowledge base
knowledge = {}

# Load employee data
filenames = glob.glob("knowledge-base/employees/*")
for filename in filenames:
    name = Path(filename).stem.split(' ')[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

print(f"Loaded {len(knowledge)} employee records")

# Load product data
filenames = glob.glob("knowledge-base/products/*")
for filename in filenames:
    name = Path(filename).stem
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

print(f"Total knowledge base entries: {len(knowledge)}")

SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""

def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]

def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result

def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    
    try:
        response = openai.chat.completions.create(
            model=MODEL, 
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Test the chat function
    print("\nTesting chat function...")
    test_response = chat("Who is Lancaster?", [])
    print(f"Test response: {test_response[:100]}...")
    
    # Launch Gradio chat interface
    print("\nLaunching Gradio interface...")
    view = gr.ChatInterface(
        chat, 
        type="messages",
        title="Insurellm Knowledge Assistant (OpenRouter)",
        description=f"Ask questions about Insurellm employees and products. Using {MODEL} via OpenRouter."
    ).launch(inbrowser=True, share=False)

if __name__ == "__main__":
    main()
