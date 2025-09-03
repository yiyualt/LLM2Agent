# mini_llm_agent.py

import os
import json
import openai
from dotenv import load_dotenv

# -------------------------------
# Load OpenAI API Key
# -------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# -------------------------------
# Define mock tools (you can replace these with real API calls)
# -------------------------------

def get_weather(city: str) -> str:
    return f"{city}: Sunny and 25°C (mocked)"

def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# -------------------------------
# Tool dispatcher
# -------------------------------

def call_tool(tool_name, input_data):
    if tool_name == "get_weather":
        return get_weather(**input_data)
    elif tool_name == "calculate":
        return calculate(**input_data)
    else:
        return f"Unknown tool: {tool_name}"


# -------------------------------
# System Prompt: Tool definitions
# -------------------------------

SYSTEM_PROMPT = """
You are an assistant that can use tools.

You have access to the following tools:

1. get_weather
   - description: Get the current weather in a city.
   - input: { "city": string }

2. calculate
   - description: Evaluate a math expression like "3 + 4 * 2".
   - input: { "expression": string }

When answering, if you need a tool, respond ONLY with JSON like:
{ "tool": "tool_name", "input": { ... } }

If no tool is needed, just answer directly.
"""


# -------------------------------
# Ask LLM for tool decision
# -------------------------------

def ask_llm(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response["choices"][0]["message"]["content"]


# -------------------------------
# Main Agent Control Logic
# -------------------------------

def run(user_input: str) -> str:
    print(f"\n🧠 User asked: {user_input}")
    response = ask_llm(user_input)
    print(f"\n🤖 LLM replied:\n{response}")

    try:
        parsed = json.loads(response)
        tool_name = parsed["tool"]
        input_data = parsed["input"]
        tool_result = call_tool(tool_name, input_data)

        print(f"\n🔧 Tool `{tool_name}` returned: {tool_result}")

        # Ask LLM to finish the answer using tool output
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Tool output: {tool_result}\nPlease complete the final answer."}
        ]
        final_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return f"\n✅ Final Answer:\n{final_response['choices'][0]['message']['content']}"

    except Exception as e:
        return f"\n⚠️ Failed to parse tool call. Raw response:\n{response}"


# -------------------------------
# Run this file directly
# -------------------------------

if __name__ == "__main__":
    print("💬 Type your question (Ctrl+C to exit):")
    while True:
        try:
            user_question = input("\n> ")
            answer = run(user_question)
            print(answer)
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

