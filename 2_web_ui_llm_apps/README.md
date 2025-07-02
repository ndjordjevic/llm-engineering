# 🌐 Web UI LLM Applications

## 📋 Overview

This folder contains a collection of interactive web applications that demonstrate advanced LLM integration with user interfaces. These applications showcase multi-provider LLM usage, web-based chatbots, tool integration, and real-time streaming capabilities.

## 🎯 Purpose

These applications demonstrate:
- **Multi-LLM Integration**: Working with OpenAI, Anthropic, and Google models
- **Web Interface Development**: Gradio-based user interfaces
- **Tool Integration**: Function calling and external tool usage
- **Conversational AI**: Interactive chatbots and conversation simulators
- **Streaming & Real-time**: Live response generation
- **Multi-modal Applications**: Voice, image, and text processing

## 📁 File Structure

### **Core Applications**

#### `gradio_example.py` (5.4KB, 199 lines)
**Purpose**: Comprehensive Gradio tutorial demonstrating web interface development

**Key Features:**
- **Basic Gradio Setup**: Simple function-to-interface conversion
- **Multiple LLM Providers**: OpenAI GPT, Anthropic Claude, Google Gemini
- **Streaming Support**: Real-time response generation
- **UI Customization**: Dark mode, custom inputs/outputs, markdown rendering
- **Model Selection**: Dropdown for choosing different LLM providers
- **Public Sharing**: Options for hosting and sharing applications

**Usage:**
```python
# Basic function interface
gr.Interface(fn=shout, inputs="textbox", outputs="textbox").launch()

# Streaming LLM interface
gr.Interface(
    fn=stream_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
).launch()
```

**Advanced Features:**
- **Multi-Model Streaming**: Compare GPT, Claude, and Gemini responses
- **Custom Styling**: Dark mode and custom CSS
- **Input/Output Types**: Text, markdown, dropdowns
- **Public Hosting**: Share applications with `share=True`

#### `airline_assistant_with_tools.py` (3.0KB, 102 lines)
**Purpose**: Interactive airline booking assistant with function calling capabilities

**Key Features:**
- **Function Calling**: LLM can call external tools (ticket pricing)
- **Conversation Memory**: Maintains chat history
- **Tool Integration**: Seamless integration of external data sources
- **Gradio Interface**: Web-based chat interface
- **Business Logic**: Real-world application with practical use case

**Workflow:**
1. **User Query**: Customer asks about flight prices
2. **Tool Detection**: LLM identifies need for pricing information
3. **Function Call**: Calls `get_ticket_price()` function
4. **Response Generation**: Provides complete answer with pricing

**Usage:**
```python
# Ask about ticket prices
"What's the price of a ticket to London?"
# Assistant calls get_ticket_price("london") and returns "$799"
```

**Tool Definition:**
```python
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to"
            }
        },
        "required": ["destination_city"]
    }
}
```

#### `airline_assistant_with_tools_image_voice_gen.py` (4.2KB, 141 lines)
**Purpose**: Advanced multi-modal airline assistant with image and voice capabilities

**Key Features:**
- **Multi-modal Input**: Text, voice, and image processing
- **Voice Generation**: Text-to-speech capabilities
- **Image Analysis**: Visual content understanding
- **Enhanced Tool Integration**: More sophisticated function calling
- **Real-time Processing**: Live voice and image analysis

**Advanced Capabilities:**
- **Voice Recognition**: Convert speech to text
- **Image Understanding**: Analyze uploaded images
- **Voice Synthesis**: Generate spoken responses
- **Multi-modal Context**: Combine text, voice, and visual information

#### `gradio_chatbot_assistant.py` (2.8KB, 96 lines)
**Purpose**: Simple but effective chatbot interface with conversation memory

**Key Features:**
- **Conversation History**: Maintains chat context across interactions
- **System Prompts**: Customizable assistant personality
- **Real-time Chat**: Live conversation interface
- **Message Formatting**: Proper chat message display
- **Easy Customization**: Simple to modify for different use cases

**Usage:**
```python
# Chat interface with memory
gr.ChatInterface(fn=chat, type="messages").launch()
```

#### `gradio_broshure.py` (3.6KB, 126 lines)
**Purpose**: Web interface for the company brochure generator from Week 1

**Key Features:**
- **Web Form Interface**: User-friendly input forms
- **Company Brochure Generation**: Automated brochure creation
- **Streaming Output**: Real-time brochure generation display
- **Input Validation**: Proper form handling and validation
- **Professional Output**: Markdown-formatted brochures

**Interface Components:**
- **Company Name Input**: Text field for company name
- **Website URL Input**: URL field for company website
- **Generate Button**: Trigger brochure generation
- **Streaming Output**: Real-time markdown display

#### `chatbot_conversation_simulator.py` (3.9KB, 112 lines)
**Purpose**: Multi-LLM conversation simulator with different AI personalities

**Key Features:**
- **Multi-LLM Integration**: GPT, Claude, and Gemini in conversation
- **Personality Simulation**: Different AI personalities (argumentative, polite, humorous)
- **Conversation Flow**: Automated multi-turn conversations
- **Personality Comparison**: Demonstrates different AI behaviors
- **Round-based Simulation**: Structured conversation progression

**AI Personalities:**
- **GPT**: Argumentative and challenging
- **Claude**: Polite and agreeable
- **Gemini**: Creative and humorous

**Usage:**
```python
# Simulate conversation between 3 AI models
simulate_conversation(rounds=5)
```

#### `joke_generator_showcase.py` (5.2KB, 158 lines)
**Purpose**: Comprehensive comparison of different LLM providers for content generation

**Key Features:**
- **Multi-Provider Comparison**: GPT-3.5, GPT-4o-mini, GPT-4o, Claude, Gemini
- **Streaming vs Non-streaming**: Compare response generation methods
- **Model Performance**: Side-by-side comparison of different models
- **Content Generation**: Focused on joke generation as a use case
- **API Integration**: Multiple API patterns and approaches

**Models Tested:**
- **OpenAI**: GPT-3.5-turbo, GPT-4o-mini, GPT-4o
- **Anthropic**: Claude-3-5-sonnet-latest (with and without streaming)
- **Google**: Gemini-2.0-flash-exp (direct and via OpenAI client)

**Usage:**
```python
# Generate jokes from all models
display_jokes()
```

## 🛠️ Setup Instructions

### Prerequisites
```bash
pip install gradio openai anthropic google-generativeai python-dotenv
```

### Environment Variables
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### API Keys Required
1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)
2. **Anthropic API Key**: Get from [Anthropic Console](https://console.anthropic.com/)
3. **Google API Key**: Get from [Google AI Studio](https://aistudio.google.com/)

## 📖 Usage Examples

### Basic Gradio Interface
```python
# Simple function interface
import gradio as gr

def greet(name):
    return f"Hello {name}!"

gr.Interface(fn=greet, inputs="text", outputs="text").launch()
```

### Multi-LLM Chat Interface
```python
# Compare different LLM responses
from gradio_example import stream_model

# Use the interface with model selection
# Select GPT, Claude, or Gemini from dropdown
```

### Airline Assistant
```python
# Ask about flight prices
"What's the cost of a ticket to Paris?"
# Assistant will call pricing function and respond
```

### Conversation Simulator
```python
# Watch AI models interact
from chatbot_conversation_simulator import simulate_conversation
simulate_conversation(rounds=3)
```

## 🔧 Technical Architecture

### Gradio Interface Patterns
```python
# Basic interface
gr.Interface(fn=function, inputs=inputs, outputs=outputs).launch()

# Chat interface
gr.ChatInterface(fn=chat_function, type="messages").launch()

# Streaming interface
gr.Interface(fn=streaming_function, inputs=inputs, outputs=outputs).launch()
```

### Multi-LLM Integration
```python
# OpenAI
openai = OpenAI()
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)

# Anthropic
claude = anthropic.Anthropic()
response = claude.messages.create(model="claude-3-haiku-20240307", messages=messages)

# Google
gemini = google.generativeai.GenerativeModel(model_name="gemini-2.0-flash-exp")
response = gemini.generate_content(prompt)
```

### Function Calling Pattern
```python
# Define tool
tools = [{"type": "function", "function": function_definition}]

# Call with tools
response = openai.chat.completions.create(
    model=MODEL, 
    messages=messages, 
    tools=tools
)

# Handle tool calls
if response.choices[0].finish_reason == "tool_calls":
    # Process tool call and continue conversation
```

### Streaming Implementation
```python
# OpenAI streaming
stream = openai.chat.completions.create(stream=True)
for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    yield accumulated_content

# Anthropic streaming
result = claude.messages.stream(model=MODEL, messages=messages)
with result as stream:
    for text in stream.text_stream:
        yield accumulated_text
```

## 📊 Comparison: LLM Providers

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| **Models** | GPT-4o, GPT-4o-mini | Claude-3-5-sonnet | Gemini-2.0-flash |
| **Streaming** | ✅ | ✅ | ✅ |
| **Function Calling** | ✅ | ✅ | ✅ |
| **Multi-modal** | ✅ | ✅ | ✅ |
| **Cost** | Pay per token | Pay per token | Pay per token |
| **API Ease** | Excellent | Good | Good |

## 🚀 Advanced Features

### Multi-modal Processing
- **Text Input**: Standard chat interfaces
- **Voice Input**: Speech-to-text conversion
- **Image Input**: Visual content analysis
- **Voice Output**: Text-to-speech generation

### Tool Integration
- **Function Calling**: LLM can call external functions
- **Data Retrieval**: Access to databases and APIs
- **Business Logic**: Integration with existing systems
- **Real-time Updates**: Live data integration

### Streaming Responses
- **Real-time Output**: Immediate response generation
- **Better UX**: Users see responses as they're generated
- **Progress Indication**: Visual feedback during processing
- **Interruption Support**: Can stop long responses

## 🔮 Potential Extensions

### Enhanced UI Features
- **Custom Themes**: Branded interfaces
- **Advanced Components**: Charts, tables, file uploads
- **Mobile Optimization**: Responsive design
- **Accessibility**: Screen reader support

### Integration Possibilities
- **Database Integration**: Persistent conversation storage
- **Authentication**: User management and access control
- **Analytics**: Usage tracking and performance monitoring
- **Deployment**: Cloud hosting and scaling

### Business Applications
- **Customer Support**: Automated help desks
- **Sales Assistants**: Product recommendations
- **Training Tools**: Interactive learning platforms
- **Data Analysis**: Natural language data exploration

## 📚 Learning Resources

### Gradio Development
- [Gradio Documentation](https://gradio.app/docs/)
- [Gradio Examples](https://gradio.app/guides/)
- [Gradio Components](https://gradio.app/docs/components)

### Multi-LLM Integration
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Google AI API Guide](https://ai.google.dev/docs)

### Web Development
- [HTML/CSS Basics](https://developer.mozilla.org/en-US/docs/Web)
- [JavaScript for Interactivity](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
- [Web Accessibility](https://www.w3.org/WAI/)

## 🤝 Contributing

To contribute to these web applications:
1. Fork the repository
2. Add new features or improve existing ones
3. Test with different LLM providers
4. Submit a pull request

## ⚠️ Important Notes

### Rate Limits
- All LLM providers have rate limits
- Consider implementing request throttling
- Monitor API usage and costs

### Security Considerations
- Never expose API keys in client-side code
- Implement proper authentication for production apps
- Validate all user inputs

### Performance Optimization
- Cache responses when appropriate
- Implement proper error handling
- Consider async processing for long operations

## 🎯 Use Cases

### Business Applications
- **Customer Service**: Automated support chatbots
- **Sales Support**: Product recommendation assistants
- **Training**: Interactive learning platforms
- **Data Analysis**: Natural language data exploration

### Educational Applications
- **Tutoring**: Interactive learning assistants
- **Language Learning**: Conversation practice tools
- **Code Review**: Programming assistance
- **Research**: Literature analysis tools

### Personal Projects
- **Content Creation**: Writing assistants
- **Personal Assistant**: Task management and reminders
- **Entertainment**: Interactive games and stories
- **Productivity**: Workflow automation tools

---

**Disclaimer**: These applications are for educational purposes. Always respect API usage limits and ensure compliance with relevant regulations when deploying to production. 