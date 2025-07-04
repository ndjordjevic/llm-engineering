# imports
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Constants
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

# A class to represent a Webpage
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


class Website:
    """
    A utility class to represent a Website that we have scraped
    """

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
system_prompt = "You are an assistant that analyzes the contents of a website \\\nand provides a short summary, ignoring text that might be navigation related. \\\nRespond in markdown."


# A function that writes a User Prompt that asks for summaries of websites:
def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \\\nplease provide a short summary of this website in markdown. \\\nIf it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt


# Function to create messages for Ollama API
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)},
    ]


# Function to summarize a website
def summarize(url):
    website = Website(url)
    payload = {"model": MODEL, "messages": messages_for(website), "stream": False}
    response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
    return response.json()["message"]["content"]


# Function to display summary in markdown
def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))


# Example usage
if __name__ == "__main__":
    display_summary("https://example.com")
    display_summary("https://cnn.com")
    display_summary("https://anthropic.com")
