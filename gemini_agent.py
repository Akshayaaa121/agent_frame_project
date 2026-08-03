"""
A minimal AI agent built from scratch — using Google's Gemini API (free tier).

Setup:
    pip install google-generativeai
    Get a free key at: https://aistudio.google.com  (click "Get API key")
    export GOOGLE_API_KEY="your-key-here"

Run:
    python gemini_agent.py
"""

import os
import datetime
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

MODEL = "gemini-2.0-flash"


# -----------------------------
# STEP 1: Write real functions the agent can call
# -----------------------------

def get_current_time():
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str):
    """Evaluate a basic math expression, e.g. '12 * 7' or '(3+5)/2'."""
    try:
        # NOTE: eval() is unsafe for untrusted input in real apps.
        # Fine for a learning project, not for production.
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


# Map tool name -> actual Python function
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}

# -----------------------------
# STEP 2: Gemini can read plain Python function signatures + docstrings
# directly to build its own tool schema — no manual JSON schema needed.
# -----------------------------

model = genai.GenerativeModel(
    model_name=MODEL,
    tools=[get_current_time, calculate],
    system_instruction="You are a helpful assistant. Use tools when needed.",
)


# -----------------------------
# STEP 3: The agent loop itself
# -----------------------------

def run_agent(user_input, max_iterations=5):
    chat = model.start_chat(enable_automatic_function_calling=False)

    response = chat.send_message(user_input)

    for i in range(max_iterations):
        # Check if the model wants to call a function
        function_call = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                function_call = part.function_call
                break

        if function_call:
            name = function_call.name
            args = dict(function_call.args)

            print(f"  [tool call] {name}({args})")

            if name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[name](**args)
            else:
                result = f"Error: unknown tool '{name}'"

            # Send the result back to the model
            response = chat.send_message(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=name,
                            response={"result": str(result)},
                        )
                    )]
                )
            )
            continue  # loop again so the model can use the result

        # No function call -> this is the final text answer
        return response.text

    return "Gave up after too many steps — the agent may be stuck."


# -----------------------------
# STEP 4: Try it
# -----------------------------

if __name__ == "__main__":
    print("Simple agent (Gemini) ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        answer = run_agent(user_input)
        print(f"Agent: {answer}\n")
