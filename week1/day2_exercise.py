# imports
import requests
import ollama
from openai import OpenAI

# Constants
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

# Create a messages list using the same format that we used for OpenAI
messages = [
    {
        "role": "user",
        "content": "Describe some of the business applications of Generative AI",
    }
]

payload = {"model": MODEL, "messages": messages, "stream": False}

response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
print(response.json()["message"]["content"])

# Alternative approach using the ollama library
response = ollama.chat(model=MODEL, messages=messages)
print(response["message"]["content"])

# There's actually an alternative approach that some people might prefer
# You can use the OpenAI client python library to call Ollama:
ollama_via_openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = ollama_via_openai.chat.completions.create(model=MODEL, messages=messages)

print(response.choices[0].message.content)

# This may take a few minutes to run! You should then see a fascinating "thinking" trace inside <think> tags, followed by some decent definitions
response = ollama_via_openai.chat.completions.create(
    model="deepseek-r1:1.5b",
    messages=[
        {
            "role": "user",
            "content": "Please give definitions of some core concepts behind LLMs: a neural network, attention and the transformer",
        }
    ],
)

print(response.choices[0].message.content)
