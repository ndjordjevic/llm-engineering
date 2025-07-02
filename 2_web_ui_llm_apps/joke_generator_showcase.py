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
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

# Initialize OpenAI and Anthropic clients
openai = OpenAI()
claude = anthropic.Anthropic()


# Function to generate a joke using GPT-3.5-Turbo
def generate_gpt35_turbo_joke():
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    prompts = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    completion = openai.chat.completions.create(model="gpt-3.5-turbo", messages=prompts)
    return completion.choices[0].message.content


# Function to generate a joke using GPT-4o-mini
def generate_gpt4o_mini_joke():
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    prompts = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", messages=prompts, temperature=0.7
    )
    return completion.choices[0].message.content

# Function to generate a joke using GPT-4o
def generate_gpt4o_joke():
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    prompts = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    completion = openai.chat.completions.create(
        model="gpt-4o", messages=prompts, temperature=0.4
    )
    return completion.choices[0].message.content


# Function to generate a joke using Anthropic Claude
def generate_claude_joke():
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    message = claude.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=200,
        temperature=0.7,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# Function to generate a joke using Claude with streaming
def generate_claude_stream_joke():
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    result = claude.messages.stream(
        model="claude-3-5-sonnet-latest",
        max_tokens=200,
        temperature=0.7,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt}],
    )
    joke = ""
    with result as stream:
        for chunk in stream.text_stream:
            joke += chunk
    return joke


# Function to generate a joke using Google Gemini
def generate_gemini_joke():
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    gemini = google.generativeai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        system_instruction=system_message
    )
    response = gemini.generate_content(user_prompt)
    return response.text


# Alternative function to generate a joke using Google Gemini via OpenAI client
def generate_gemini_via_openai_joke():
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    prompts = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    gemini_via_openai_client = OpenAI(
        api_key=google_api_key, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    response = gemini_via_openai_client.chat.completions.create(
        model="gemini-2.0-flash-exp",
        messages=prompts
    )
    return response.choices[0].message.content


# Function to display jokes
def display_jokes():
    print("GPT-3.5-Turbo Joke:")
    print(generate_gpt35_turbo_joke())
    print("\nGPT-4o-mini Joke:")
    print(generate_gpt4o_mini_joke())
    print("\nGPT-4o Joke:")
    print(generate_gpt4o_joke())
    print("\nClaude Joke:")
    print(generate_claude_joke())
    print("\nClaude Stream Joke:")
    print(generate_claude_stream_joke())
    print("\nGoogle Gemini Joke (Direct):")
    print(generate_gemini_joke())
    print("\nGoogle Gemini Joke (via OpenAI):")
    print(generate_gemini_via_openai_joke())


# Example usage
if __name__ == "__main__":
    display_jokes()
