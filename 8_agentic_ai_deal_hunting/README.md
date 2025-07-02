# 🤖 Agentic AI Deal Hunting System

## 📋 Overview

This folder contains a comprehensive **agentic AI system** for autonomous deal hunting and price optimization. This represents the culmination of the LLM engineering course, combining cloud deployment, RAG systems, multi-agent architectures, and interactive web interfaces to create a sophisticated autonomous system.

## 🎯 Purpose

This system demonstrates:
- **Agentic AI**: Autonomous multi-agent systems working together
- **Cloud Deployment**: Serverless functions with Modal
- **Advanced RAG**: 400K product vector database with semantic search
- **Ensemble Pricing**: Multiple AI models collaborating on price estimation
- **Real-time Deal Hunting**: Automated deal discovery and evaluation
- **Interactive UI**: Gradio-based web interface for user interaction
- **Production Systems**: Deployed services and APIs

## 📁 File Structure

### **Core Tutorials**

#### `day1.ipynb` (12KB, 386 lines)
**Purpose**: Introduction to Modal cloud deployment and serverless functions

**Key Features:**
- **Modal Setup**: Cloud infrastructure configuration
- **Serverless Functions**: Ephemeral and deployed functions
- **Cloud Services**: Regional deployment and scaling
- **API Integration**: HuggingFace and OpenAI integration
- **Production Deployment**: Deploying services to production

**Learning Objectives:**
- Set up Modal cloud infrastructure
- Create serverless functions
- Deploy services to production
- Integrate with external APIs
- Scale applications globally

**Key Concepts:**
```python
import modal

# Create Modal app
app = modal.App("pricing-service")

# Define serverless function
@app.function(image=image)
def price(product_description):
    # Pricing logic here
    return estimated_price

# Deploy to production
# modal deploy pricer_service
```

#### `day2.0.ipynb` (8.4KB, 277 lines)
**Purpose**: Building massive RAG database with 400K products

**Key Features:**
- **Vector Database**: Chroma with 400K product embeddings
- **Sentence Transformers**: Local embedding generation
- **Semantic Search**: Advanced similarity matching
- **Data Processing**: Large-scale data curation
- **Performance Optimization**: Efficient vector operations

**Learning Objectives:**
- Build large-scale vector databases
- Generate embeddings locally
- Implement semantic search
- Process massive datasets
- Optimize vector operations

#### `day2.1.ipynb` (5.1KB, 183 lines)
**Purpose**: 2D visualization of product embeddings

**Key Features:**
- **t-SNE Visualization**: Dimensionality reduction
- **Interactive Plots**: Plotly-based visualizations
- **Category Analysis**: Product category clustering
- **Price Distribution**: Visual price analysis
- **Data Insights**: Exploratory data analysis

#### `day2.2.ipynb` (4.9KB, 175 lines)
**Purpose**: 3D visualization of product embeddings

**Key Features:**
- **3D t-SNE**: Three-dimensional visualization
- **Interactive 3D Plots**: Rotatable 3D visualizations
- **Spatial Analysis**: Product space exploration
- **Category Mapping**: Visual category relationships
- **Price Clustering**: 3D price distribution analysis

#### `day2.3.ipynb` (13KB, 504 lines)
**Purpose**: RAG pipeline with GPT-4o-mini integration

**Key Features:**
- **RAG Pipeline**: Retrieval-augmented generation
- **Frontier Model Integration**: GPT-4o-mini for analysis
- **Context Enhancement**: Rich context for LLM responses
- **Performance Evaluation**: RAG vs. direct LLM comparison
- **Prompt Engineering**: Optimized prompts for RAG

**Learning Objectives:**
- Build RAG pipelines
- Integrate frontier models
- Enhance context with retrieval
- Evaluate RAG performance
- Engineer effective prompts

#### `day2.4.ipynb` (11KB, 409 lines)
**Purpose**: Ensemble pricing with multiple AI models

**Key Features:**
- **Random Forest Integration**: Traditional ML pricing
- **Ensemble Methods**: Combining multiple models
- **Model Comparison**: Performance benchmarking
- **Weighted Averaging**: Intelligent model combination
- **Confidence Scoring**: Model confidence assessment

#### `day3.ipynb` (6.7KB, 236 lines)
**Purpose**: Agentic AI framework introduction

**Key Features:**
- **Multi-Agent Architecture**: Agent coordination
- **Planning Agent**: Strategic orchestration
- **Specialist Agents**: Domain-specific expertise
- **Communication Protocols**: Inter-agent messaging
- **Memory Management**: Persistent agent memory

#### `day4.ipynb` (4.2KB, 153 lines)
**Purpose**: Advanced agent capabilities

**Key Features:**
- **Agent Specialization**: Role-based agent design
- **Task Delegation**: Intelligent task distribution
- **Collaborative Decision Making**: Multi-agent consensus
- **Learning Agents**: Adaptive agent behavior
- **Performance Monitoring**: Agent performance tracking

#### `day5.ipynb` (7.8KB, 210 lines)
**Purpose**: Interactive web interface with Gradio

**Key Features:**
- **Gradio UI**: User-friendly web interface
- **Real-time Updates**: Live deal monitoring
- **Interactive Tables**: Dynamic data display
- **User Interactions**: Click-based deal selection
- **Visual Analytics**: Data visualization components

### **Core System Files**

#### `deal_agent_framework.py` (3.3KB, 99 lines)
**Purpose**: Main orchestrator for the agentic AI system

**Key Features:**
- **Agent Management**: Centralized agent coordination
- **Memory System**: Persistent opportunity tracking
- **Database Integration**: Chroma vector store access
- **Logging System**: Comprehensive system logging
- **Visualization Support**: Data plotting capabilities

#### `price_is_right.py` (2.5KB, 62 lines)
**Purpose**: Basic deal hunting implementation

#### `price_is_right_final.py` (6.3KB, 166 lines)
**Purpose**: Complete deal hunting system with UI

### **Cloud Deployment Files**

#### `hello.py` (766B, 30 lines)
**Purpose**: Basic Modal function demonstration

#### `llama.py` (1.3KB, 45 lines)
**Purpose**: Llama model integration with Modal

#### `pricer_ephemeral.py` (2.1KB, 67 lines)
**Purpose**: Ephemeral pricing service

#### `pricer_service.py` (2.2KB, 67 lines)
**Purpose**: Deployed pricing service

#### `pricer_service2.py` (3.1KB, 90 lines)
**Purpose**: Enhanced pricing service with class-based approach

### **Supporting Files**

#### `items.py` (3.4KB, 101 lines)
**Purpose**: Product item data structures

#### `testing.py` (2.5KB, 75 lines)
**Purpose**: Model evaluation framework

#### `log_utils.py` (745B, 35 lines)
**Purpose**: Logging utilities

#### `keep_warm.py` (234B, 10 lines)
**Purpose**: Service keep-alive functionality

#### `memory.json` (1.5KB, 20 lines)
**Purpose**: Persistent agent memory storage

## 🛠️ Setup Instructions

### Prerequisites
```bash
# Core dependencies
pip install modal gradio chromadb sentence-transformers

# Machine learning
pip install scikit-learn pandas numpy plotly

# Cloud and APIs
pip install openai anthropic twilio

# Visualization
pip install matplotlib seaborn

# Additional utilities
pip install tqdm python-dotenv
```

### Modal Setup
```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup

# Verify installation
modal token new
```

### Environment Variables
Create a `.env` file:
```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
HF_TOKEN=your_huggingface_token_here

# Twilio (for notifications)
TWILIO_ACCOUNT_SID=your_twilio_sid_here
TWILIO_AUTH_TOKEN=your_twilio_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_here

# Modal configuration
MODAL_TOKEN_ID=your_modal_token_here
```

### Database Setup
```bash
# The Chroma database will be created automatically
# Ensure you have sufficient disk space for 400K product embeddings

# For Windows users with Chroma issues:
pip install chromadb==0.5.0
```

## 📖 Usage Examples

### Basic Modal Function
```python
import modal

# Create app
app = modal.App("example")

# Define function
@app.function(image=modal.Image.debian_slim())
def hello():
    return "Hello from Modal!"

# Run locally
with app.run():
    result = hello.local()
    print(result)

# Run remotely
with app.run():
    result = hello.remote()
    print(result)
```

### Deploy Service
```python
# Deploy to production
# modal deploy pricer_service

# Use deployed service
pricer = modal.Function.lookup("pricer-service", "price")
result = pricer.remote("Product description here")
```

### RAG Pipeline
```python
import chromadb
from sentence_transformers import SentenceTransformer

# Setup Chroma
client = chromadb.PersistentClient(path="products_vectorstore")
collection = client.get_collection("products")

# Setup embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Search for similar products
query = "wireless bluetooth headphones"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)
```

### Agent Framework
```python
from deal_agent_framework import DealAgentFramework

# Initialize framework
framework = DealAgentFramework()
framework.init_agents_as_needed()

# Run deal hunting
opportunities = framework.run()

# Access memory
memory = framework.memory
```

### Gradio Interface
```python
import gradio as gr
from deal_agent_framework import DealAgentFramework

# Create interface
with gr.Blocks(title="Deal Hunting AI") as ui:
    gr.Markdown("# Deal Hunting Agentic AI")
    
    # Add components
    product_input = gr.Textbox(label="Product Description")
    price_output = gr.Number(label="Estimated Price")
    
    def estimate_price(description):
        # Pricing logic here
        return estimated_price
    
    product_input.change(estimate_price, inputs=[product_input], outputs=[price_output])

# Launch interface
ui.launch()
```

## 🔧 Technical Architecture

### Multi-Agent System
```python
class DealAgentFramework:
    def __init__(self):
        self.planner = PlanningAgent()
        self.frontier = FrontierAgent()
        self.scanner = ScannerAgent()
        self.specialist = SpecialistAgent()
        self.ensemble = EnsembleAgent()
    
    def run(self):
        # Orchestrate agent collaboration
        plan = self.planner.create_plan()
        deals = self.scanner.find_deals()
        analysis = self.frontier.analyze(deals)
        expertise = self.specialist.apply_expertise(analysis)
        final_estimate = self.ensemble.combine_estimates(expertise)
        return final_estimate
```

### Cloud Deployment Pipeline
```python
# Ephemeral function (temporary)
@app.function(image=image)
def ephemeral_pricing(description):
    return estimate_price(description)

# Deployed service (persistent)
@app.function(image=image)
def deployed_pricing(description):
    return estimate_price(description)

# Deploy to production
# modal deploy pricer_service
```

### RAG System Architecture
```python
class RAGSystem:
    def __init__(self):
        self.vector_db = ChromaDB()
        self.embedding_model = SentenceTransformer()
        self.llm = GPT4oMini()
    
    def retrieve_context(self, query):
        embedding = self.embedding_model.encode(query)
        similar_docs = self.vector_db.query(embedding, n_results=5)
        return similar_docs
    
    def generate_response(self, query, context):
        prompt = f"Context: {context}\nQuery: {query}"
        response = self.llm.generate(prompt)
        return response
```

## 📊 Performance Comparison

| Component | Performance | Cost | Scalability | Privacy |
|-----------|-------------|------|-------------|---------|
| **Modal Functions** | Fast | Pay-per-use | Auto-scaling | ✅ |
| **Chroma Vector DB** | High-speed | Free | 400K+ items | ✅ |
| **Sentence Transformers** | Local | Free | Unlimited | ✅ |
| **GPT-4o-mini** | Fast | $0.15/1M tokens | High | ❌ |
| **Ensemble Models** | Accurate | Free | Scalable | ✅ |
| **Agent Framework** | Autonomous | Free | Multi-agent | ✅ |

## 🚀 Advanced Features

### Autonomous Deal Hunting
```python
class AutonomousDealHunter:
    def __init__(self):
        self.agents = self.initialize_agents()
        self.memory = self.load_memory()
    
    def hunt_deals(self):
        while True:
            # Scan for new deals
            new_deals = self.scan_for_deals()
            
            # Analyze each deal
            for deal in new_deals:
                analysis = self.analyze_deal(deal)
                if analysis.is_good_deal():
                    self.notify_user(analysis)
            
            # Update memory
            self.update_memory()
            
            # Wait before next scan
            time.sleep(300)  # 5 minutes
```

### Real-time Notifications
```python
from twilio.rest import Client

class NotificationSystem:
    def __init__(self):
        self.client = Client(account_sid, auth_token)
    
    def send_deal_alert(self, deal):
        message = f"Great deal found! {deal.description} for ${deal.price}"
        self.client.messages.create(
            body=message,
            from_=twilio_number,
            to=user_number
        )
```

### Performance Monitoring
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def track_prediction(self, model, prediction, actual):
        error = abs(prediction - actual)
        if model not in self.metrics:
            self.metrics[model] = []
        self.metrics[model].append(error)
    
    def get_performance_report(self):
        report = {}
        for model, errors in self.metrics.items():
            report[model] = {
                'mean_error': np.mean(errors),
                'total_predictions': len(errors)
            }
        return report
```

## 🔮 Potential Extensions

### Enhanced Agent Capabilities
- **Learning Agents**: Agents that improve over time
- **Specialized Agents**: Domain-specific expertise
- **Collaborative Agents**: Multi-agent coordination
- **Adaptive Agents**: Dynamic behavior adjustment

### Advanced Analytics
- **Market Analysis**: Trend prediction and analysis
- **Price Forecasting**: Future price predictions
- **Competitive Intelligence**: Competitor analysis
- **Risk Assessment**: Deal risk evaluation

### Production Features
- **Load Balancing**: Distributed agent deployment
- **Fault Tolerance**: Error handling and recovery
- **Monitoring**: Real-time system monitoring
- **Scaling**: Auto-scaling capabilities

### Integration Capabilities
- **E-commerce APIs**: Direct integration with stores
- **Payment Systems**: Automated purchasing
- **Inventory Management**: Stock tracking
- **Customer Management**: User preference learning

## 📚 Learning Resources

### Agentic AI
- [Multi-Agent Systems](https://www.masfoundations.org/)
- [Agent Communication](https://fipa.org/)
- [Autonomous Agents](https://www.autonomousagents.org/)

### Cloud Deployment
- [Modal Documentation](https://modal.com/docs/)
- [Serverless Architecture](https://aws.amazon.com/serverless/)
- [Cloud Functions](https://cloud.google.com/functions)

### RAG Systems
- [Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [Semantic Search](https://www.sbert.net/)

### Production Systems
- [System Design](https://github.com/donnemartin/system-design-primer)
- [Microservices](https://microservices.io/)
- [API Design](https://restfulapi.net/)

## 🤝 Contributing

To contribute to this agentic AI system:
1. Fork the repository
2. Add new agent types
3. Improve the RAG pipeline
4. Enhance the UI
5. Submit a pull request

## ⚠️ Important Notes

### System Requirements
- **Memory**: 16GB+ RAM for large vector operations
- **Storage**: 10GB+ for vector database
- **Network**: Stable internet for cloud services
- **GPU**: Optional but recommended for embeddings

### Cost Management
- **API Usage**: Monitor OpenAI/Anthropic usage
- **Cloud Costs**: Track Modal usage
- **Storage Costs**: Monitor vector database size
- **Network Costs**: Consider data transfer costs

### Privacy Considerations
- **Data Localization**: Keep sensitive data local
- **API Security**: Secure API key management
- **User Privacy**: Protect user data
- **Compliance**: Follow data protection regulations

## 🎯 Use Cases

### Business Applications
- **E-commerce Optimization**: Automated deal hunting
- **Price Intelligence**: Competitive price monitoring
- **Inventory Management**: Smart inventory decisions
- **Customer Service**: Automated deal recommendations

### Personal Applications
- **Personal Shopping**: Automated deal discovery
- **Price Tracking**: Monitor price changes
- **Budget Optimization**: Smart spending decisions
- **Investment Research**: Market analysis

### Research Applications
- **Market Research**: Automated market analysis
- **Price Modeling**: Advanced pricing algorithms
- **Agent Behavior**: Multi-agent system research
- **AI Ethics**: Responsible AI development

---

**Disclaimer**: This system is for educational purposes. Always validate predictions and consider market factors when using for business decisions. 