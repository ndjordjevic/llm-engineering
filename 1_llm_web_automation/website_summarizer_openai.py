# imports
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

# Load environment variables in a file called .env
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# Check the key
if not api_key:
    print(
        "No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!"
    )
elif not api_key.startswith("sk-proj-"):
    print(
        "An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook"
    )
elif api_key.strip() != api_key:
    print(
        "An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook"
    )
else:
    print("API key found and looks good so far!")

openai = OpenAI()

# A class to represent a Webpage
# Some websites need you to use proper headers when fetching them:
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


class Website:
    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


# Define our system prompt
system_prompt = "You are an assistant that analyzes the contents of a weibsite \\\nand provides a short summary, ignoring text that might be navigation related. \\\nRespond in markdown."


# A function that writes a User Prompt that asks for summaries of websites:
def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \\\nplease provide a short summary of this website in markdown. \\\nIf it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt


# Function to create messages for OpenAI API
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)},
    ]


# Function to summarize a website
def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages_for(website)
    )
    return response.choices[0].message.content


# Function to display summary in markdown
def display_summary(url):   
    summary = summarize(url)
    display(Markdown(summary))


# Example usage
if __name__ == "__main__":
    display_summary("https://example.com")
    display_summary("https://cnn.com")
    display_summary("https://anthropic.com")
