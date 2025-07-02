# 🤗 HuggingFace Local Models & Transformers

## 📋 Overview

This folder contains tutorials and projects focused on the **HuggingFace ecosystem** and **local model deployment**. These notebooks are designed to run on **Google Colab** with GPU acceleration, teaching you how to work with transformer models locally without relying on external APIs.

## 🎯 Purpose

These tutorials demonstrate:
- **Local Model Deployment**: Running models on your own hardware/cloud
- **HuggingFace Ecosystem**: Transformers, tokenizers, and pipelines
- **GPU Acceleration**: Optimized for T4 GPU runtime
- **Audio Processing**: Speech-to-text and meeting transcription
- **Token Management**: HuggingFace authentication and token handling
- **Colab Integration**: Google Colab environment setup and usage

## 📁 File Structure

### **Core Tutorials**

#### `day1.ipynb` (1.3KB, 50 lines)
**Purpose**: Google Colab introduction and environment setup

**Key Features:**
- **Colab Setup**: Introduction to Google Colab environment
- **Account Configuration**: Setting up Google account for Colab
- **Free Tier Usage**: Instructions for minimal cost/free usage
- **Pro+ Recommendations**: Optional premium features
- **Environment Overview**: Understanding Colab capabilities

**Learning Objectives:**
- Set up Google Colab account
- Understand Colab's free vs. paid tiers
- Navigate Colab interface
- Prepare for GPU-accelerated development

**Colab Link**: [Day 1 - Colab Introduction](https://colab.research.google.com/drive/1DjcrYDZldAXKJ08x1uYIVCtItoLPk1Wr?usp=sharing)

#### `day2.ipynb` (1.2KB, 50 lines)
**Purpose**: HuggingFace pipelines - high-level API for model inference

**Key Features:**
- **HuggingFace Pipelines**: High-level API exploration
- **Token Setup**: HuggingFace token configuration
- **GPU Optimization**: T4 GPU runtime instructions
- **Model Inference**: Easy-to-use pipeline interface
- **Secret Management**: Secure token handling in Colab

**Learning Objectives:**
- Understand HuggingFace pipelines
- Set up HuggingFace authentication
- Use high-level API for model inference
- Optimize for GPU performance

**Colab Link**: [Day 2 - HuggingFace Pipelines](https://colab.research.google.com/drive/1aMaEw8A56xs0bRM4lu8z7ou18jqyybGm?usp=sharing)

**Key Concepts:**
```python
# Example pipeline usage
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification")
result = classifier("I love this movie!")

# Translation
translator = pipeline("translation_en_to_fr")
translated = translator("Hello, how are you?")
```

#### `day3.ipynb` (823B, 38 lines)
**Purpose**: Tokenizers - understanding text tokenization and processing

**Key Features:**
- **Tokenization Process**: How text becomes model inputs
- **Different Tokenizers**: BERT, GPT, T5 tokenizer comparison
- **Token Analysis**: Understanding token IDs and vocabulary
- **Text Processing**: Preprocessing text for models
- **Vocabulary Management**: Token-to-ID mappings

**Learning Objectives:**
- Understand tokenization fundamentals
- Compare different tokenizer types
- Analyze tokenization results
- Prepare text for model input

**Colab Link**: [Day 3 - Tokenizers](https://colab.research.google.com/drive/1WD6Y2N7ctQi1X9wa6rpkg8UfyA4iSVuz?usp=sharing)

**Key Concepts:**
```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
tokens = tokenizer("Hello, world!")
print(tokens)  # {'input_ids': [101, 7592, 1010, 2088, 999, 102], ...}

# Decode tokens
decoded = tokenizer.decode(tokens['input_ids'])
print(decoded)  # "[CLS] hello, world! [SEP]"
```

#### `day4.ipynb` (889B, 40 lines)
**Purpose**: Models - working with transformer models directly

**Key Features:**
- **Model Architecture**: Understanding transformer models
- **Model Loading**: Loading pre-trained models locally
- **Inference**: Running models for predictions
- **Model Types**: Different model architectures and use cases
- **Performance Optimization**: T4 GPU utilization

**Learning Objectives:**
- Load and use transformer models
- Understand model architecture
- Perform inference with custom inputs
- Optimize model performance

**Colab Link**: [Day 4 - Models](https://colab.research.google.com/drive/1hhR9Z-yiqjUe7pJjVQw4c74z_V3VchLy?usp=sharing)

**Key Concepts:**
```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare input
inputs = tokenizer("Hello, world!", return_tensors="pt")

# Run inference
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # torch.Size([1, 4, 768])
```

#### `day5.ipynb` (1.1KB, 52 lines)
**Purpose**: Meeting minutes creator - practical audio processing application

**Key Features:**
- **Audio Processing**: Speech-to-text conversion
- **Google Drive Integration**: Connecting Colab to Google Drive
- **Meeting Transcription**: Converting audio to text
- **Minutes Generation**: Creating structured meeting summaries
- **File Upload**: Handling audio file uploads

**Learning Objectives:**
- Process audio files with HuggingFace
- Integrate Google Drive with Colab
- Create practical audio applications
- Generate meeting summaries

**Colab Link**: [Day 5 - Meeting Minutes Creator](https://colab.research.google.com/drive/1KSMxOCprsl1QRpt_Rq0UqCAyMtPqDQYy?usp=sharing)

**Key Concepts:**
```python
from transformers import pipeline

# Speech recognition pipeline
transcriber = pipeline("automatic-speech-recognition")

# Transcribe audio file
result = transcriber("audio_file.wav")
transcript = result["text"]

# Generate meeting minutes
minutes_generator = pipeline("text-generation")
minutes = minutes_generator(f"Create meeting minutes from: {transcript}")
```

## 🛠️ Setup Instructions

### Prerequisites
```bash
# Core HuggingFace libraries
pip install transformers torch torchaudio

# Audio processing
pip install librosa soundfile

# Google Drive integration
pip install google-colab
```

### Environment Setup

#### 1. Google Colab Account
- Visit [Google Colab](https://colab.research.google.com/)
- Sign in with your Google account
- Enable GPU runtime (T4 recommended)

#### 2. HuggingFace Token
- Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Create a new token with read permissions
- Copy the token for use in notebooks

#### 3. Colab Secrets
```python
# Add HuggingFace token as secret
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')  # Set this in Colab secrets
```

### GPU Configuration
```python
# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 📖 Usage Examples

### Basic Pipeline Usage
```python
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Question answering
qa_pipeline = pipeline("question-answering")
context = "The Eiffel Tower is located in Paris, France."
question = "Where is the Eiffel Tower?"
answer = qa_pipeline(question=question, context=context)
print(answer)  # {'answer': 'Paris, France', 'score': 0.9999}
```

### Custom Model Loading
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
text = "This is a sample text for classification."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(f"Predictions: {predictions}")
```

### Audio Processing
```python
from transformers import pipeline
import librosa

# Load audio file
audio_path = "meeting_recording.wav"
audio, sr = librosa.load(audio_path, sr=16000)

# Transcribe audio
transcriber = pipeline("automatic-speech-recognition")
result = transcriber(audio)
transcript = result["text"]

print(f"Transcript: {transcript}")
```

## 🔧 Technical Architecture

### HuggingFace Pipeline Architecture
```python
# Pipeline components
pipeline = Pipeline(
    model=AutoModel.from_pretrained("model_name"),
    tokenizer=AutoTokenizer.from_pretrained("model_name"),
    feature_extractor=AutoFeatureExtractor.from_pretrained("model_name")
)
```

### Model Loading Patterns
```python
# Auto-classes for automatic model selection
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Load with specific model type
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load with custom configuration
config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 3  # Custom number of labels
model = AutoModel.from_pretrained("bert-base-uncased", config=config)
```

### Tokenization Process
```python
# Tokenization workflow
text = "Hello, world!"

# 1. Tokenize
tokens = tokenizer.tokenize(text)
print(tokens)  # ['hello', ',', 'world', '!']

# 2. Convert to IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)  # [7592, 1010, 2088, 999]

# 3. Add special tokens
inputs = tokenizer(text, return_tensors="pt")
print(inputs)  # {'input_ids': tensor([[101, 7592, 1010, 2088, 999, 102]]), ...}
```

### GPU Optimization
```python
# Move model to GPU
model = model.to(device)

# Batch processing for efficiency
batch_texts = ["Text 1", "Text 2", "Text 3"]
inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
```

## 📊 Model Comparison

| Model Type | Use Case | Example Models | GPU Memory |
|------------|----------|----------------|------------|
| **BERT** | Text classification, QA | bert-base-uncased | ~500MB |
| **GPT** | Text generation | gpt2 | ~500MB |
| **T5** | Text-to-text | t5-small | ~300MB |
| **Whisper** | Speech recognition | openai/whisper-base | ~1GB |
| **DistilBERT** | Lightweight BERT | distilbert-base-uncased | ~250MB |

## 🚀 Advanced Features

### Custom Training
```python
from transformers import TrainingArguments, Trainer

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train model
trainer.train()
```

### Model Quantization
```python
from transformers import AutoModelForSequenceClassification

# Load quantized model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto"
)
```

### Pipeline Customization
```python
# Custom pipeline
from transformers import pipeline

class CustomPipeline:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return self.process_outputs(outputs)
    
    def process_outputs(self, outputs):
        # Custom output processing
        return outputs
```

## 🔮 Potential Extensions

### Enhanced Audio Processing
- **Real-time Transcription**: Live audio processing
- **Speaker Diarization**: Identify different speakers
- **Emotion Detection**: Analyze speech emotions
- **Language Detection**: Automatic language identification

### Advanced Text Processing
- **Document Classification**: Multi-label classification
- **Named Entity Recognition**: Extract entities from text
- **Text Summarization**: Generate document summaries
- **Translation**: Multi-language translation

### Model Optimization
- **Model Pruning**: Reduce model size
- **Knowledge Distillation**: Transfer knowledge to smaller models
- **Model Compression**: Optimize for deployment
- **Quantization**: Reduce precision for efficiency

### Integration Possibilities
- **Web Applications**: Deploy models as APIs
- **Mobile Apps**: On-device model inference
- **Edge Computing**: IoT device deployment
- **Batch Processing**: Large-scale inference pipelines

## 📚 Learning Resources

### HuggingFace Documentation
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Model Hub](https://huggingface.co/models)
- [Datasets Hub](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)

### Google Colab
- [Colab Documentation](https://colab.research.google.com/notebooks/)
- [GPU Runtime Guide](https://colab.research.google.com/notebooks/gpu.ipynb)
- [TPU Runtime Guide](https://colab.research.google.com/notebooks/tpu.ipynb)

### Audio Processing
- [Librosa Documentation](https://librosa.org/doc/latest/)
- [SoundFile Documentation](https://pysoundfile.readthedocs.io/)
- [Audio Processing Tutorials](https://huggingface.co/docs/transformers/tasks/automatic_speech_recognition)

### Model Optimization
- [Model Compression Guide](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.half)
- [Quantization Tutorial](https://huggingface.co/docs/transformers/main_classes/quantization)
- [Performance Optimization](https://huggingface.co/docs/transformers/performance)

## 🤝 Contributing

To contribute to these tutorials:
1. Fork the repository
2. Add new HuggingFace examples
3. Improve existing tutorials
4. Test with different models
5. Submit a pull request

## ⚠️ Important Notes

### Resource Management
- **GPU Memory**: Monitor memory usage with large models
- **Model Caching**: HuggingFace caches models locally
- **Token Limits**: Be aware of model input/output limits
- **API Rate Limits**: Some models have usage restrictions

### Best Practices
- **Model Selection**: Choose appropriate model size for your use case
- **Error Handling**: Implement proper error handling for model loading
- **Security**: Be careful with model weights and tokens
- **Performance**: Use batching for efficient inference

### Cost Considerations
- **Colab Pro**: Consider for longer training sessions
- **GPU Usage**: Monitor GPU time usage
- **Model Storage**: Large models consume storage space
- **API Costs**: Some models may require paid API access

## 🎯 Use Cases

### Business Applications
- **Document Processing**: Automated document classification
- **Customer Support**: Sentiment analysis of support tickets
- **Content Moderation**: Automated content filtering
- **Market Research**: Social media sentiment analysis

### Educational Applications
- **Language Learning**: Grammar correction and translation
- **Content Creation**: Automated text generation
- **Research**: Literature analysis and summarization
- **Assessment**: Automated essay grading

### Personal Projects
- **Chatbots**: Custom conversational agents
- **Content Summarization**: News and article summaries
- **Language Translation**: Multi-language communication
- **Audio Processing**: Meeting transcription and notes

---

**Disclaimer**: These tutorials are for educational purposes. Always respect model licenses and usage terms when deploying to production. 