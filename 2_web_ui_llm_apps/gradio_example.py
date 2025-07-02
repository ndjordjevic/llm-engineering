# imports
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
import gradio as gr  # oh yeah!

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

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

# Connect to OpenAI, Anthropic and Google; comment out the Claude or Google lines if you're not using them
openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure()

# A generic system message - no more snarky adversarial AIs!
system_message = "You are a helpful assistant"


# Let's wrap a call to GPT-4o-mini in a simple function
def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content


# This can reveal the "training cut off", or the most recent date in the training data
message_gpt("What is today's date?")


# here's a simple function
def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()


shout("hello")

# The simplicity of gradio. This might appear in "light mode" - I'll show you how to make this in dark mode later.
gr.Interface(fn=shout, inputs="textbox", outputs="textbox").launch()

# Adding share=True means that it can be accessed publicly
# A more permanent hosting is available using a platform called Spaces from HuggingFace, which we will touch on next week
# NOTE: Some Anti-virus software and Corporate Firewalls might not like you using share=True. If you're at work on a work network, I suggest skip this test.
gr.Interface(
    fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never"
).launch(share=True)

# Adding inbrowser=True opens up a new browser window automatically
gr.Interface(
    fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never"
).launch(inbrowser=True)

# Define this variable and then pass js=force_dark_mode when creating the Interface
force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
gr.Interface(
    fn=shout,
    inputs="textbox",
    outputs="textbox",
    flagging_mode="never",
    js=force_dark_mode,
).launch()

# Inputs and Outputs
view = gr.Interface(
    fn=shout,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never",
)
view.launch()

# And now - changing the function from "shout" to "message_gpt"
view = gr.Interface(
    fn=message_gpt,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never",
)
view.launch()

# Let's use Markdown
system_message = "You are a helpful assistant that responds in markdown"

view = gr.Interface(
    fn=message_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never",
)
view.launch()


# Let's create a call that streams back results
def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    stream = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


view = gr.Interface(
    fn=stream_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never",
)
view.launch()


def stream_claude(prompt):
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response


view = gr.Interface(
    fn=stream_claude,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never",
)
view.launch()


# Minor improvement
def stream_model(prompt, model):
    if model == "GPT":
        result = stream_gpt(prompt)
    elif model == "Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


view = gr.Interface(
    fn=stream_model,
    inputs=[
        gr.Textbox(label="Your message:"),
        gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT"),
    ],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never",
)
view.launch()
