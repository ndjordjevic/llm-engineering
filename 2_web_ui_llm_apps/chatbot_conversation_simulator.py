# imports
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai

# Load environment variables in a file called .env
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check API keys
if not openai_api_key:
    raise ValueError("OpenAI API Key not set")
if not anthropic_api_key:
    raise ValueError("Anthropic API Key not set")
if not google_api_key:
    raise ValueError("Google API Key not set")

# Initialize OpenAI and Anthropic clients
openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure(api_key=google_api_key)

# Define system prompts for GPT, Claude, and Gemini
gpt_system = (
    "You are a chatbot who is very argumentative; "
    "you disagree with anything in the conversation and you challenge everything, in a snarky way."
)
claude_system = (
    "You are a very polite, courteous chatbot. You try to agree with "
    "everything the other person says, or find common ground. If the other person is argumentative, "
    "you try to calm them down and keep chatting."
)
gemini_system = (
    "You are a creative and humorous chatbot. You try to add a fun and lighthearted "
    "perspective to the conversation while keeping it engaging."
)

# Initialize conversation history
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]
gemini_messages = ["Hello everyone!"]

# Function to call GPT
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude, gemini in zip(gpt_messages, claude_messages, gemini_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude})
        messages.append({"role": "user", "content": gemini})
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content

# Function to call Claude
def call_claude():
    messages = []
    for gpt, claude_message, gemini in zip(gpt_messages, claude_messages, gemini_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
        messages.append({"role": "user", "content": gemini})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    message = claude.messages.create(
        model="claude-3-haiku-20240307",
        system=claude_system,
        messages=messages,
        max_tokens=500,
    )
    return message.content[0].text

# Function to call Gemini
def call_gemini():
    messages = [{"role": "system", "content": gemini_system}]
    for gpt, claude, gemini in zip(gpt_messages, claude_messages, gemini_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "user", "content": claude})
        messages.append({"role": "assistant", "content": gemini})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    gemini = google.generativeai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        system_instruction=gemini_system
    )
    response = gemini.generate_content(gpt_messages[-1])
    return response.text

# Function to simulate the conversation
def simulate_conversation(rounds=5):
    print(f"GPT:\n{gpt_messages[0]}\n")
    print(f"Claude:\n{claude_messages[0]}\n")
    print(f"Gemini:\n{gemini_messages[0]}\n")
    for _ in range(rounds):
        gpt_next = call_gpt()
        print(f"GPT:\n{gpt_next}\n")
        gpt_messages.append(gpt_next)

        claude_next = call_claude()
        print(f"Claude:\n{claude_next}\n")
        claude_messages.append(claude_next)

        gemini_next = call_gemini()
        print(f"Gemini:\n{gemini_next}\n")
        gemini_messages.append(gemini_next)

# Example usage
if __name__ == "__main__":
    simulate_conversation()
