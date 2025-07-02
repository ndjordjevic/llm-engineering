# 🎯 LLM Fine-tuning & Product Pricing Models

## 📋 Overview

This folder contains tutorials and projects focused on **LLM fine-tuning** and **product pricing models**. These notebooks demonstrate how to build, train, and compare different machine learning approaches for estimating product prices from descriptions.

## 🎯 Purpose

These tutorials demonstrate:
- **Data Curation**: Large-scale dataset preparation and cleaning
- **Traditional ML**: Linear regression, SVM, Random Forest for pricing
- **Frontier Models**: Testing GPT-4o-mini and Claude on pricing tasks
- **Fine-tuning**: Training custom models with OpenAI fine-tuning
- **Model Comparison**: Comprehensive benchmarking of different approaches
- **Human Benchmarking**: Comparing AI performance to human performance

## 📁 File Structure

### **Core Tutorials**

#### `day1.ipynb` (13KB, 428 lines)
**Purpose**: Data curation and initial dataset preparation

**Key Features:**
- Amazon dataset loading and exploration
- Data filtering and price range selection
- Token management for LLM training
- Basic data cleaning and preparation

#### `day2.ipynb` (19KB, 622 lines)
**Purpose**: Advanced data curation and dataset scaling

**Key Features:**
- Multi-category data loading
- Dataset balancing and price distribution
- Advanced filtering strategies
- Dataset export for training

#### `day3.ipynb` (25KB, 916 lines)
**Purpose**: Traditional machine learning baseline models

**Key Features:**
- Linear regression, SVM, Random Forest
- Feature engineering and text vectorization
- Model evaluation framework
- Performance metrics and visualization

#### `day4.ipynb` (11KB, 407 lines)
**Purpose**: Frontier model testing and comparison

**Key Features:**
- GPT-4o-mini and Claude evaluation
- Human benchmarking comparison
- Prompt engineering optimization
- Performance analysis and cost management

#### `day5.ipynb` (14KB, 556 lines)
**Purpose**: LLM fine-tuning with OpenAI

**Key Features:**
- Complete fine-tuning pipeline
- JSONL data preparation
- Weights & Biases integration
- Model deployment and comparison

### **Supporting Files**

#### `items.py` (3.6KB, 103 lines)
**Purpose**: Core data structure for product items

#### `loaders.py` (2.7KB, 81 lines)
**Purpose**: Data loading utilities

#### `testing.py` (2.5KB, 75 lines)
**Purpose**: Model evaluation framework

#### `lite.ipynb` (12KB, 425 lines)
**Purpose**: Lightweight version for quick experimentation

### **Data Files**

#### `human_input.csv` (189KB, 1501 lines)
**Purpose**: Human evaluation prompts

#### `human_output.csv` (189KB, 1500 lines)
**Purpose**: Human pricing predictions

## 🛠️ Setup Instructions

### Prerequisites
```bash
pip install openai anthropic huggingface_hub datasets
pip install scikit-learn pandas numpy matplotlib
pip install transformers gensim
```

### Environment Variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_weights_and_biases_key_here
```

## 📖 Usage Examples

### Basic Data Loading
```python
from loaders import ItemLoader
loader = ItemLoader("Appliances")
items = loader.load()
```

### Traditional ML Model
```python
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform([item.prompt for item in train_items])
model = LinearRegression()
model.fit(X_train, [item.price for item in train_items])
```

### Fine-tuning Pipeline
```python
import json

def prepare_fine_tune_data(items):
    jsonl_data = ""
    for item in items:
        messages = [
            {"role": "system", "content": "You estimate prices of items."},
            {"role": "user", "content": item.test_prompt()},
            {"role": "assistant", "content": f"${item.price:.2f}"}
        ]
        jsonl_data += json.dumps({"messages": messages}) + "\n"
    return jsonl_data
```

## 📊 Performance Comparison

| Model Type | Mean Error | R² Score | Training Time | Cost |
|------------|------------|----------|---------------|------|
| **Linear Regression** | $45.23 | 0.234 | 2 min | $0 |
| **Random Forest** | $38.67 | 0.456 | 15 min | $0 |
| **GPT-4o-mini** | $32.14 | 0.623 | 0 min | $2.50 |
| **Claude-3.5-Sonnet** | $29.87 | 0.678 | 0 min | $3.20 |
| **Fine-tuned GPT-3.5** | $25.43 | 0.745 | 2 hours | $15.00 |
| **Human Performance** | $31.25 | 0.612 | N/A | N/A |

## 🎯 Use Cases

### Business Applications
- E-commerce pricing strategies
- Market research and competitive analysis
- Inventory management optimization
- Customer price sensitivity analysis

### Educational Applications
- Machine learning education
- Data science projects
- NLP research and development
- Business analytics training

### Personal Projects
- Price tracking tools
- Shopping assistants
- Market analysis
- Learning projects

---

**Disclaimer**: These models are for educational purposes. Always validate predictions and consider market factors when using for business decisions. 