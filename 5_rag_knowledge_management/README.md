# 🔍 RAG Knowledge Management Systems

## 📋 Overview

This folder contains tutorials and projects focused on **Retrieval Augmented Generation (RAG)** systems for knowledge management. These notebooks demonstrate how to build intelligent question-answering agents that can access and query company knowledge bases using vector databases, embeddings, and conversational AI.

## 🎯 Purpose

These tutorials demonstrate:
- **RAG Implementation**: Building retrieval-augmented generation systems
- **Vector Databases**: Working with Chroma vector store for document storage
- **Document Processing**: Loading, chunking, and embedding company documents
- **Knowledge Management**: Organizing and querying structured company data
- **Embedding Models**: Using OpenAI and HuggingFace embeddings
- **Conversational AI**: Building chat interfaces with conversation memory
- **Business Applications**: Real-world RAG use cases for enterprise

## 📁 File Structure

### **Core Tutorials**

#### `day1.ipynb` (7.4KB, 265 lines)
**Purpose**: Basic RAG implementation with simple keyword-based retrieval

**Key Features:**
- **Simple RAG**: Brute-force keyword matching for document retrieval
- **Knowledge Base Loading**: Loading employee and product documents
- **Context Injection**: Adding relevant context to LLM prompts
- **Gradio Interface**: Web-based chat interface
- **Cost Optimization**: Using GPT-4o-mini for low-cost implementation

**Learning Objectives:**
- Understand basic RAG concepts
- Load and process company documents
- Implement simple keyword-based retrieval
- Build a basic chat interface
- Manage API costs effectively

**Key Concepts:**
```python
# Simple keyword-based context retrieval
def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context

# Context injection into prompts
def add_context(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message
```

**RAG Workflow:**
1. **Document Loading**: Load company documents (employees, products)
2. **Keyword Matching**: Find relevant documents based on user query
3. **Context Injection**: Add relevant context to LLM prompt
4. **Response Generation**: Generate accurate answers using context
5. **Chat Interface**: Provide user-friendly web interface

#### `day2.ipynb` (5.0KB, 196 lines)
**Purpose**: Advanced document processing with LangChain

**Key Features:**
- **LangChain Integration**: Using LangChain document loaders
- **Directory Loading**: Loading documents from multiple folders
- **Text Chunking**: Splitting documents into searchable chunks
- **Metadata Management**: Adding document type metadata
- **Structured Processing**: Organized document processing pipeline

**Learning Objectives:**
- Use LangChain for document processing
- Implement text chunking strategies
- Manage document metadata
- Process structured knowledge bases
- Handle different document formats

**Document Processing Pipeline:**
```python
# Load documents from multiple folders
folders = glob.glob("knowledge-base/*")
documents = []

for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
```

#### `day3.ipynb` (12KB, 367 lines)
**Purpose**: Vector database implementation with Chroma and embeddings

**Key Features:**
- **Vector Database**: Chroma vector store implementation
- **Embeddings**: OpenAI embeddings for text-to-vector conversion
- **Vector Visualization**: t-SNE visualization of document embeddings
- **Similarity Search**: Finding relevant documents using vector similarity
- **Dimensionality Analysis**: Understanding embedding dimensions

**Learning Objectives:**
- Set up and use Chroma vector database
- Generate and store document embeddings
- Visualize vector spaces
- Implement similarity search
- Understand embedding dimensions

**Vector Database Setup:**
```python
# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=db_name
)

# Visualize embeddings
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])
```

#### `day4.ipynb` (15KB, 443 lines)
**Purpose**: Advanced RAG with conversation memory and retrieval chains

**Key Features:**
- **Conversation Memory**: Maintaining chat context across interactions
- **Retrieval Chains**: LangChain conversational retrieval chains
- **Advanced Visualization**: 3D vector space visualization
- **Memory Management**: Conversation buffer memory implementation
- **Enhanced Retrieval**: Improved document retrieval strategies

**Learning Objectives:**
- Implement conversation memory
- Use LangChain retrieval chains
- Create advanced visualizations
- Manage conversation context
- Build sophisticated RAG systems

**Conversation Memory Implementation:**
```python
# Set up conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model=MODEL),
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Chat function with memory
def chat_with_memory(message, history):
    result = qa_chain({"question": message})
    return result["answer"]
```

#### `day4.5.ipynb` (13KB, 409 lines)
**Purpose**: Enhanced RAG with advanced features and optimizations

**Key Features:**
- **Advanced Retrieval**: Improved document retrieval algorithms
- **Performance Optimization**: Optimized RAG performance
- **Enhanced Visualization**: Advanced vector space analysis
- **Query Optimization**: Better query processing
- **System Improvements**: Overall RAG system enhancements

**Learning Objectives:**
- Optimize RAG system performance
- Implement advanced retrieval strategies
- Create sophisticated visualizations
- Improve query processing
- Enhance overall system capabilities

#### `day5.ipynb` (16KB, 484 lines)
**Purpose**: Production-ready RAG system with comprehensive features

**Key Features:**
- **Production Features**: Enterprise-ready RAG implementation
- **HuggingFace Integration**: Alternative embedding models
- **Comprehensive Testing**: Full system testing and validation
- **Advanced Features**: All advanced RAG capabilities
- **Business Integration**: Real-world business application

**Learning Objectives:**
- Build production-ready RAG systems
- Integrate alternative embedding models
- Test and validate RAG systems
- Deploy enterprise solutions
- Apply RAG to business problems

**Production RAG System:**
```python
# Alternative embedding models
from langchain.embeddings import HuggingFaceEmbeddings

# Use HuggingFace embeddings for privacy
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Production-ready vector store
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=db_name
)

# Enterprise chat interface
def enterprise_chat(message, history):
    # Advanced retrieval with filtering
    docs = vectorstore.similarity_search(
        message, 
        k=5,
        filter={"doc_type": "employees"}  # Filter by document type
    )
    
    # Enhanced context processing
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate response with context
    response = generate_response(message, context, history)
    return response
```

### **Knowledge Base Structure**

#### `knowledge-base/` Directory
**Purpose**: Structured company data for RAG system training

**Contents:**
- **`company/`**: Company information (about, careers, overview)
- **`employees/`**: Employee profiles and information
- **`products/`**: Product descriptions and specifications
- **`contracts/`**: Contract documents and agreements

**Document Types:**
- **Markdown Files**: Structured text documents
- **Metadata**: Document type and source information
- **Chunking**: Automatically split into searchable chunks
- **Embeddings**: Vector representations for similarity search

## 🛠️ Setup Instructions

### Prerequisites
```bash
# Core dependencies
pip install langchain langchain-openai langchain-chroma chromadb

# Visualization and analysis
pip install plotly scikit-learn numpy matplotlib

# Web interface
pip install gradio

# Environment management
pip install python-dotenv

# Alternative embeddings
pip install sentence-transformers
```

### Environment Variables
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### API Keys Required
1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)
2. **HuggingFace Token**: Get from [HuggingFace Settings](https://huggingface.co/settings/tokens) (optional)

### Knowledge Base Setup
```bash
# Create knowledge base structure
mkdir -p knowledge-base/{company,employees,products,contracts}

# Add your documents to appropriate folders
# Supported formats: .md, .txt, .pdf (with appropriate loaders)
```

## 📖 Usage Examples

### Basic RAG Implementation
```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load documents
loader = DirectoryLoader("knowledge-base", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Query the knowledge base
query = "Who is the CEO of the company?"
docs = vectorstore.similarity_search(query, k=3)
```

### Conversational RAG
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Chat function
def chat(message):
    result = qa_chain({"question": message})
    return result["answer"]
```

### Vector Visualization
```python
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np

# Get vectors from database
result = vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create visualization
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=documents,
    hoverinfo='text'
)])

fig.show()
```

## 🔧 Technical Architecture

### RAG Pipeline Architecture
```python
class RAGSystem:
    def __init__(self, knowledge_base_path, embedding_model="openai"):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.memory = ConversationBufferMemory()
    
    def load_documents(self):
        """Load and process documents from knowledge base"""
        loader = DirectoryLoader(self.knowledge_base_path, glob="**/*.md")
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = doc.metadata.get("source", "unknown")
            doc.metadata["doc_type"] = self._get_doc_type(doc.metadata["source"])
        
        return documents
    
    def create_embeddings(self, documents):
        """Create embeddings and store in vector database"""
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        if self.embedding_model == "openai":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings()
        
        self.vectorstore = Chroma.from_documents(chunks, embeddings)
        return self.vectorstore
    
    def query(self, question, k=5):
        """Query the knowledge base"""
        docs = self.vectorstore.similarity_search(question, k=k)
        return docs
    
    def generate_response(self, question, context, history=None):
        """Generate response using LLM"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to company knowledge."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        
        if history:
            messages.extend(history)
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        return response.choices[0].message.content
```

### Document Processing Pipeline
```python
def process_documents(directory_path):
    """Complete document processing pipeline"""
    
    # 1. Load documents
    documents = load_documents(directory_path)
    
    # 2. Add metadata
    documents = add_metadata(documents)
    
    # 3. Split into chunks
    chunks = split_documents(documents)
    
    # 4. Create embeddings
    embeddings = create_embeddings(chunks)
    
    # 5. Store in vector database
    vectorstore = store_embeddings(embeddings)
    
    return vectorstore

def load_documents(path):
    """Load documents from directory"""
    loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader)
    return loader.load()

def add_metadata(documents):
    """Add metadata to documents"""
    for doc in documents:
        doc.metadata["doc_type"] = get_doc_type(doc.metadata["source"])
        doc.metadata["timestamp"] = datetime.now().isoformat()
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separator="\n"
    )
    return text_splitter.split_documents(documents)
```

### Vector Database Management
```python
class VectorDatabaseManager:
    def __init__(self, db_path, embedding_function):
        self.db_path = db_path
        self.embedding_function = embedding_function
        self.vectorstore = None
    
    def create_database(self, documents):
        """Create new vector database"""
        if os.path.exists(self.db_path):
            self.delete_database()
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.db_path
        )
        return self.vectorstore
    
    def load_database(self):
        """Load existing vector database"""
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )
        return self.vectorstore
    
    def delete_database(self):
        """Delete existing database"""
        if os.path.exists(self.db_path):
            Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_function
            ).delete_collection()
    
    def add_documents(self, documents):
        """Add new documents to database"""
        if self.vectorstore is None:
            self.load_database()
        
        self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query, k=5, filter_dict=None):
        """Search for similar documents"""
        if self.vectorstore is None:
            self.load_database()
        
        return self.vectorstore.similarity_search(
            query, 
            k=k, 
            filter=filter_dict
        )
```

## 📊 Performance Comparison

| Feature | Simple RAG | Vector RAG | Advanced RAG |
|---------|------------|------------|--------------|
| **Retrieval Method** | Keyword matching | Vector similarity | Hybrid search |
| **Accuracy** | Low | Medium | High |
| **Speed** | Fast | Medium | Fast |
| **Scalability** | Limited | Good | Excellent |
| **Memory Usage** | Low | Medium | High |
| **Cost** | Low | Medium | Medium |

## 🚀 Advanced Features

### Hybrid Search
```python
def hybrid_search(query, vectorstore, k=5):
    """Combine keyword and vector search"""
    
    # Vector similarity search
    vector_results = vectorstore.similarity_search(query, k=k)
    
    # Keyword search
    keyword_results = keyword_search(query, vectorstore, k=k)
    
    # Combine and rank results
    combined_results = combine_results(vector_results, keyword_results)
    
    return combined_results[:k]
```

### Document Filtering
```python
def filtered_search(query, vectorstore, doc_type=None, date_range=None):
    """Search with document type and date filters"""
    
    filter_dict = {}
    
    if doc_type:
        filter_dict["doc_type"] = doc_type
    
    if date_range:
        filter_dict["timestamp"] = {
            "$gte": date_range[0],
            "$lte": date_range[1]
        }
    
    return vectorstore.similarity_search(
        query, 
        k=5, 
        filter=filter_dict
    )
```

### Conversation Memory
```python
class ConversationManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.max_history = 10
    
    def add_message(self, role, content):
        """Add message to conversation history"""
        self.memory.chat_memory.add_user_message(content)
        if len(self.memory.chat_memory.messages) > self.max_history * 2:
            # Keep only recent messages
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_history * 2:]
    
    def get_context(self):
        """Get conversation context"""
        return self.memory.load_memory_variables({})
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
```

## 🔮 Potential Extensions

### Enhanced Retrieval
- **Multi-modal RAG**: Support for images, audio, and video
- **Temporal RAG**: Time-aware document retrieval
- **Hierarchical RAG**: Multi-level document organization
- **Federated RAG**: Distributed knowledge bases

### Advanced Analytics
- **Query Analytics**: Analyze user query patterns
- **Document Analytics**: Track document usage and relevance
- **Performance Metrics**: Monitor RAG system performance
- **A/B Testing**: Test different retrieval strategies

### Integration Possibilities
- **Enterprise Systems**: Integration with existing business systems
- **API Development**: RESTful APIs for RAG services
- **Mobile Applications**: Mobile-friendly RAG interfaces
- **Real-time Updates**: Live document updates and indexing

### Business Applications
- **Customer Support**: Automated customer service systems
- **Employee Onboarding**: Knowledge base for new employees
- **Product Documentation**: Intelligent product documentation
- **Legal Document Search**: Legal document retrieval and analysis

## 📚 Learning Resources

### RAG Fundamentals
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Chroma Vector Database](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

### Vector Databases
- [Vector Database Comparison](https://zilliz.com/comparison)
- [Chroma Tutorials](https://docs.trychroma.com/getting-started)
- [Pinecone Documentation](https://docs.pinecone.io/)

### Advanced RAG
- [Advanced RAG Techniques](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAG Evaluation](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/evaluation)
- [RAG Optimization](https://python.langchain.com/docs/use_cases/question_answering/)

### Business Applications
- [Enterprise RAG Solutions](https://www.anthropic.com/claude)
- [RAG Case Studies](https://openai.com/research)
- [Knowledge Management](https://www.notion.so/)

## 🤝 Contributing

To contribute to these RAG tutorials:
1. Fork the repository
2. Add new RAG examples and techniques
3. Improve existing implementations
4. Test with different knowledge bases
5. Submit a pull request

## ⚠️ Important Notes

### Data Privacy
- **Local Processing**: Consider using local embedding models for sensitive data
- **Data Encryption**: Encrypt sensitive documents before processing
- **Access Control**: Implement proper access controls for knowledge bases
- **Audit Logging**: Log all queries and access for compliance

### Performance Optimization
- **Chunk Size**: Optimize chunk size for your use case
- **Embedding Models**: Choose appropriate embedding models
- **Vector Database**: Select suitable vector database for scale
- **Caching**: Implement caching for frequently accessed documents

### Cost Management
- **API Usage**: Monitor embedding API usage and costs
- **Model Selection**: Choose cost-effective models for your needs
- **Batch Processing**: Process documents in batches to reduce costs
- **Local Alternatives**: Use local models when possible

## 🎯 Use Cases

### Business Applications
- **Customer Support**: Automated customer service with company knowledge
- **Employee Training**: Onboarding and training knowledge bases
- **Product Documentation**: Intelligent product documentation search
- **Legal Research**: Legal document retrieval and analysis

### Educational Applications
- **Research Assistance**: Academic research and literature review
- **Student Support**: Educational content and FAQ systems
- **Library Systems**: Intelligent library catalog search
- **Knowledge Discovery**: Discovering connections in large document collections

### Personal Projects
- **Personal Knowledge Base**: Organizing personal notes and documents
- **Research Tools**: Academic and professional research assistance
- **Content Creation**: Research and fact-checking for content creation
- **Learning Systems**: Personalized learning and study assistance

---

**Disclaimer**: These RAG systems are for educational purposes. Always ensure proper data handling and privacy protection when deploying to production environments. 