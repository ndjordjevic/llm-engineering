# 🤖 LLM Web Automation Tools

## 📋 Overview

This folder contains a collection of AI-powered web automation tools that demonstrate different approaches to LLM integration. These tools showcase both cloud-based (OpenAI) and local (Ollama) LLM usage for web content processing, summarization, and generation.

## 🎯 Purpose

These tools demonstrate:
- **Web Scraping & Content Extraction**: Automated gathering of web content
- **LLM Integration**: Both cloud (OpenAI) and local (Ollama) approaches
- **Content Generation**: AI-powered summarization and document creation
- **Multi-step AI Workflows**: Complex automation chains
- **Educational Applications**: Code explanation and learning tools

## 📁 File Structure

### **Core Tools**

#### `website_summarizer_openai.py` (3.1KB, 92 lines)
**Purpose**: Webpage summarization using OpenAI's GPT-4o-mini model

**Key Features:**
- **Web Scraping**: Extracts content from any webpage using BeautifulSoup
- **Content Processing**: Removes irrelevant elements (scripts, styles, images)
- **OpenAI Integration**: Uses GPT-4o-mini for intelligent summarization
- **Markdown Output**: Displays summaries in formatted markdown
- **Error Handling**: Validates API keys and provides troubleshooting guidance

**Usage:**
```python
from website_summarizer_openai import display_summary

# Summarize any website
display_summary("https://example.com")
```

**Sample Output:**
- Summarizes CNN, Anthropic, and personal websites
- Provides concise, well-formatted summaries
- Ignores navigation-related content

#### `website_summarizer_ollama.py` (2.4KB, 74 lines)
**Purpose**: Webpage summarization using local Ollama LLM (Llama 3.2)

**Key Features:**
- **Local LLM**: Uses Ollama with Llama 3.2 model
- **No API Costs**: Runs entirely on your local machine
- **Privacy-Focused**: No data sent to external services
- **Same Interface**: Identical API to OpenAI version
- **HTTP API**: Direct communication with Ollama's REST API

**Usage:**
```python
from website_summarizer_ollama import display_summary

# Requires Ollama running locally with llama3.2 model
display_summary("https://example.com")
```

**Setup Requirements:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull llama3.2

# Start Ollama service
ollama serve
```

#### `company_brochure_generator.py` (5.3KB, 153 lines)
**Purpose**: Advanced multi-step AI workflow that generates company brochures from websites

**Key Features:**
- **Multi-Page Analysis**: Scrapes multiple relevant pages from a company website
- **Intelligent Link Selection**: Uses LLM to identify relevant pages (About, Careers, etc.)
- **Content Aggregation**: Combines information from multiple sources
- **Professional Output**: Generates markdown brochures for customers, investors, and recruits
- **Streaming Support**: Real-time brochure generation with typewriter effect
- **JSON Parsing**: Structured output for link analysis

**Workflow:**
1. **Link Analysis**: LLM identifies relevant pages from website links
2. **Content Extraction**: Scrapes content from selected pages
3. **Information Aggregation**: Combines all relevant content
4. **Brochure Generation**: Creates professional company brochure
5. **Streaming Display**: Shows results in real-time

**Usage:**
```python
from company_brochure_generator import create_brochure, stream_brochure

# Generate brochure for a company
create_brochure("HuggingFace", "https://huggingface.co")

# Stream the brochure generation
stream_brochure("HuggingFace", "https://huggingface.co")
```

**Advanced Features:**
- **Content Truncation**: Handles large websites (5,000 character limit)
- **Error Handling**: Graceful handling of missing content
- **Flexible Prompts**: Customizable brochure tone and style

#### `code_explanation_tool.py` (1.3KB, 45 lines)
**Purpose**: Educational tool for explaining Python code using multiple LLM providers

**Key Features:**
- **Multi-Model Comparison**: Compares GPT-4o-mini vs Llama 3.2 explanations
- **Streaming Output**: Real-time code explanation display
- **Educational Focus**: Designed for learning Python concepts
- **Interactive**: Easy to modify questions for different code examples

**Usage:**
```python
from code_explanation_tool import *

# Modify the question variable to ask about different code
question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""
```

**Educational Value:**
- Demonstrates differences between cloud and local LLMs
- Shows streaming vs non-streaming responses
- Provides detailed technical explanations

## 🛠️ Setup Instructions

### Prerequisites
```bash
pip install requests beautifulsoup4 python-dotenv openai ollama ipython
```

### Environment Variables
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### API Keys Required
1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)
2. **Ollama**: Local installation (no API key needed)

### Ollama Setup (for local LLM usage)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull deepseek

# Start Ollama service
ollama serve
```

## 📖 Usage Examples

### Basic Web Summarization
```python
# Using OpenAI
from website_summarizer_openai import display_summary
display_summary("https://openai.com")

# Using Ollama (local)
from website_summarizer_ollama import display_summary
display_summary("https://openai.com")
```

### Company Brochure Generation
```python
from company_brochure_generator import create_brochure

# Generate brochure for any company
create_brochure("Anthropic", "https://anthropic.com")
create_brochure("Google", "https://google.com")
```

### Code Explanation
```python
from code_explanation_tool import *

# Ask about any Python code
question = """
Explain this list comprehension:
[x for x in range(10) if x % 2 == 0]
"""
```

## 🔧 Technical Architecture

### Web Scraping Components
```python
class Website:
    def __init__(self, url):
        # Fetches webpage content
        # Removes irrelevant elements (scripts, styles, images)
        # Extracts clean text content
        # Handles missing titles gracefully
```

### LLM Integration Patterns
```python
# OpenAI Pattern
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}]
)

# Ollama Pattern
response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
```

### Streaming Implementation
```python
# Real-time output display
stream = openai.chat.completions.create(stream=True)
for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    # Update display in real-time
```

## 📊 Comparison: OpenAI vs Ollama

| Feature | OpenAI (GPT-4o-mini) | Ollama (Llama 3.2) |
|---------|----------------------|-------------------|
| **Cost** | Pay per request | Free (local) |
| **Privacy** | Data sent to OpenAI | Local processing |
| **Performance** | High quality | Good quality |
| **Setup** | API key only | Local installation |
| **Speed** | Fast (cloud) | Variable (local) |
| **Reliability** | High uptime | Depends on local setup |

## 🚀 Advanced Features

### Multi-Step AI Workflows
The `company_brochure_generator.py` demonstrates advanced AI workflows:
1. **Link Analysis**: LLM identifies relevant pages
2. **Content Extraction**: Scrapes multiple pages
3. **Information Synthesis**: Combines data from multiple sources
4. **Content Generation**: Creates professional output

### Streaming Responses
Real-time output display for better user experience:
- Typewriter effect for brochure generation
- Immediate feedback for long-running operations
- Better user engagement

### Error Handling
Robust error handling throughout:
- API key validation
- Network error handling
- Content parsing errors
- Graceful degradation

## 🔮 Potential Extensions

### Content Enhancement
- Add image analysis for visual content
- Include sentiment analysis
- Add language translation capabilities

### Performance Optimization
- Implement caching for repeated requests
- Add parallel processing for multiple pages
- Optimize token usage

### Business Applications
- Automated content creation
- Market research automation
- Competitive analysis tools
- SEO content generation

## 📚 Learning Resources

### Web Automation Concepts
- **Web Scraping**: BeautifulSoup and requests libraries
- **Content Processing**: Text extraction and cleaning
- **API Integration**: OpenAI and Ollama documentation
- **Streaming**: Real-time data processing

### Related Topics
- **Agentic AI**: Multi-step AI workflows
- **Content Generation**: Automated writing systems
- **Local LLMs**: Privacy and cost considerations
- **Web Automation**: Scraping and data extraction

## 🤝 Contributing

To contribute to these automation tools:
1. Fork the repository
2. Add new features or improvements
3. Test with different websites and use cases
4. Submit a pull request

## ⚠️ Important Notes

### Rate Limits
- OpenAI has rate limits on API calls
- Ollama performance depends on local hardware
- Consider implementing request throttling

### Content Quality
- Generated content should be reviewed before use
- Web scraping may not work on all sites
- Some sites block automated access

### Legal Considerations
- Respect robots.txt files
- Check website terms of service
- Ensure compliance with data usage policies

## 🎯 Use Cases

### Business Applications
- **Market Research**: Automated competitor analysis
- **Content Creation**: Generate summaries and reports
- **Lead Generation**: Extract company information
- **SEO Analysis**: Content optimization insights

### Educational Applications
- **Code Learning**: Interactive code explanations
- **Research**: Automated literature reviews
- **Documentation**: Generate technical documentation
- **Training**: Create educational content

### Personal Projects
- **News Aggregation**: Summarize multiple news sources
- **Research Assistant**: Extract key information from websites
- **Content Curation**: Organize and summarize web content
- **Learning Tools**: Create personalized study materials

---

**Disclaimer**: These tools are for educational purposes. Always respect website terms of service and ensure compliance with relevant regulations when scraping web content. 