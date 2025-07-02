# 🤖 LLM Code Generation & Optimization

## 📋 Overview

This folder contains tutorials and projects focused on **LLM-powered code generation** and **performance optimization**. These notebooks demonstrate how to use Large Language Models to generate high-performance C++ code from Python, compare different LLM providers, and deploy open-source models for code generation tasks.

## 🎯 Purpose

These tutorials demonstrate:
- **Code Generation**: Using LLMs to generate production-ready code
- **Performance Optimization**: Converting Python to high-performance C++
- **Multi-Model Comparison**: Testing different LLM providers (GPT-4o, Claude-3.5-Sonnet)
- **Open Source Model Deployment**: HuggingFace endpoints for model hosting
- **Code Translation**: Converting between programming languages
- **Performance Benchmarking**: Comparing execution times between implementations

## 📁 File Structure

### **Core Tutorials**

#### `day3.ipynb` (20KB, 616 lines)
**Purpose**: Code generation using frontier models (GPT-4o and Claude-3.5-Sonnet)

**Key Features:**
- **Frontier Model Usage**: GPT-4o and Claude-3.5-Sonnet for code generation
- **Python to C++ Translation**: Converting Python algorithms to optimized C++
- **Performance Optimization**: Generating high-performance implementations
- **Streaming Responses**: Real-time code generation output
- **Cost Management**: Options for ultra-low cost model alternatives

**Learning Objectives:**
- Use frontier LLMs for code generation
- Convert Python algorithms to C++
- Optimize code for performance
- Compare different LLM providers
- Manage API costs effectively

**Key Concepts:**
```python
# System message for code generation
system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
system_message += "The C++ response needs to produce an identical output in the fastest possible time."

# User prompt for code translation
def user_prompt_for(python):
    user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
    user_prompt += python
    return user_prompt
```

**Code Generation Workflow:**
1. **Input Python Code**: Algorithm to be optimized
2. **LLM Processing**: Generate C++ equivalent
3. **Code Output**: Write to `optimized.cpp`
4. **Compilation**: Compile with optimization flags
5. **Performance Testing**: Compare execution times

#### `day4.ipynb` (29KB, 849 lines)
**Purpose**: Code generation using open-source models via HuggingFace endpoints

**Key Features:**
- **Open Source Model Deployment**: HuggingFace endpoints for model hosting
- **Local Model Inference**: Using deployed models instead of cloud APIs
- **Cost Optimization**: Pause endpoints when not in use
- **Advanced Algorithms**: Complex mathematical computations
- **Performance Comparison**: Python vs. C++ execution times

**Learning Objectives:**
- Deploy open-source models on HuggingFace endpoints
- Use local model inference for code generation
- Manage endpoint costs and resources
- Compare open-source vs. frontier model performance
- Implement complex algorithms in C++

**HuggingFace Endpoint Setup:**
```python
# Endpoint configuration
endpoint_url = "https://your-endpoint.huggingface.cloud"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    "Content-Type": "application/json"
}

# Model inference
def optimize_with_endpoint(python_code):
    payload = {
        "inputs": user_prompt_for(python_code),
        "parameters": {
            "max_new_tokens": 2000,
            "temperature": 0.1
        }
    }
    response = requests.post(endpoint_url, headers=headers, json=payload)
    return response.json()[0]["generated_text"]
```

**Cost Management:**
- **Pause Endpoints**: Stop endpoints when not in use
- **Resource Monitoring**: Track usage and costs
- **Alternative Models**: Use different model sizes for cost optimization

### **Generated Code Files**

#### `simple.cpp` (77B, 8 lines)
**Purpose**: Basic C++ example for testing compilation

**Content:**
```cpp
#include <iostream>

int main() {
    std::cout << "Hello";
    return 0;
}
```

**Usage:**
- **Compilation Test**: Verify C++ environment setup
- **Basic Example**: Simple C++ program structure
- **Learning Reference**: Basic C++ syntax and structure

#### `optimized.cpp` (1.9KB, 65 lines)
**Purpose**: LLM-generated optimized C++ implementation

**Key Features:**
- **High Performance**: Optimized for speed and efficiency
- **Algorithm Implementation**: Complex mathematical computations
- **Memory Management**: Efficient data structures and algorithms
- **Precision Handling**: Proper number types and overflow prevention
- **Timing Measurement**: High-resolution clock for performance measurement

**Generated Code Example:**
```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include <iomanip>

class LCG {
private:
    uint64_t value;
    const uint64_t a = 1664525;
    const uint64_t c = 1013904223;
    const uint64_t m = 1ULL << 32;

public:
    LCG(uint64_t seed) : value(seed) {}

    uint64_t next() {
        value = (a * value + c) % m;
        return value;
    }
};

int64_t max_subarray_sum(int n, uint64_t seed, int min_val, int max_val) {
    LCG lcg(seed);
    std::vector<int> random_numbers(n);
    for (int i = 0; i < n; ++i) {
        random_numbers[i] = static_cast<int>(lcg.next() % (max_val - min_val + 1) + min_val);
    }

    int64_t max_sum = std::numeric_limits<int64_t>::min();
    int64_t current_sum = 0;
    for (int i = 0; i < n; ++i) {
        current_sum = std::max(static_cast<int64_t>(random_numbers[i]), current_sum + random_numbers[i]);
        max_sum = std::max(max_sum, current_sum);
    }
    return max_sum;
}
```

## 🛠️ Setup Instructions

### Prerequisites
```bash
# Core dependencies
pip install openai anthropic google-generativeai python-dotenv

# For HuggingFace endpoints
pip install requests

# For C++ compilation (macOS)
# Install Xcode Command Line Tools
xcode-select --install

# For C++ compilation (Linux)
sudo apt-get install build-essential
```

### Environment Variables
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### API Keys Required
1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)
2. **Anthropic API Key**: Get from [Anthropic Console](https://console.anthropic.com/)
3. **HuggingFace Token**: Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### C++ Compilation Setup
```bash
# macOS (M1 Mac optimization)
clang++ -O3 -std=c++17 -march=armv8.3-a -o optimized optimized.cpp

# Linux
g++ -O3 -std=c++17 -o optimized optimized.cpp

# Windows
g++ -O3 -std=c++17 -o optimized.exe optimized.cpp
```

## 📖 Usage Examples

### Basic Code Generation
```python
from openai import OpenAI
import anthropic

# Initialize clients
openai = OpenAI()
claude = anthropic.Anthropic()

# Python code to optimize
python_code = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

# Generate C++ with GPT-4o
def optimize_gpt(python):
    stream = openai.chat.completions.create(
        model="gpt-4o", 
        messages=messages_for(python), 
        stream=True
    )
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        print(fragment, end='', flush=True)
    return reply

# Generate C++ with Claude
def optimize_claude(python):
    result = claude.messages.stream(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            print(text, end="", flush=True)
    return reply
```

### HuggingFace Endpoint Usage
```python
import requests
import os

def optimize_with_endpoint(python_code, endpoint_url):
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": user_prompt_for(python_code),
        "parameters": {
            "max_new_tokens": 2000,
            "temperature": 0.1,
            "do_sample": True
        }
    }
    
    response = requests.post(endpoint_url, headers=headers, json=payload)
    return response.json()[0]["generated_text"]
```

### Performance Comparison
```python
import time
import subprocess

def compare_performance():
    # Run Python version
    start_time = time.time()
    exec(python_code)
    python_time = time.time() - start_time
    
    # Compile and run C++ version
    subprocess.run(["clang++", "-O3", "-std=c++17", "-o", "optimized", "optimized.cpp"])
    start_time = time.time()
    subprocess.run(["./optimized"])
    cpp_time = time.time() - start_time
    
    print(f"Python execution time: {python_time:.6f} seconds")
    print(f"C++ execution time: {cpp_time:.6f} seconds")
    print(f"Speedup: {python_time / cpp_time:.2f}x")
```

## 🔧 Technical Architecture

### Code Generation Pipeline
```python
# 1. System message definition
system_message = "You are an assistant that reimplements Python code in high performance C++..."

# 2. User prompt construction
def user_prompt_for(python):
    return f"Rewrite this Python code in C++ with the fastest possible implementation...\n\n{python}"

# 3. Message formatting
def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

# 4. Code generation
def generate_code(python, model="gpt-4o"):
    if model == "gpt-4o":
        return optimize_gpt(python)
    elif model == "claude":
        return optimize_claude(python)
    elif model == "endpoint":
        return optimize_with_endpoint(python, endpoint_url)
```

### Model Comparison Framework
```python
class CodeGenerator:
    def __init__(self):
        self.openai = OpenAI()
        self.claude = anthropic.Anthropic()
        self.models = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
            "claude-3-haiku": "claude-3-haiku-20240307"
        }
    
    def generate(self, python_code, model_name):
        if model_name.startswith("gpt"):
            return self._generate_with_openai(python_code, self.models[model_name])
        elif model_name.startswith("claude"):
            return self._generate_with_claude(python_code, self.models[model_name])
        else:
            return self._generate_with_endpoint(python_code, model_name)
    
    def compare_models(self, python_code, model_names):
        results = {}
        for model in model_names:
            print(f"\nGenerating with {model}...")
            results[model] = self.generate(python_code, model)
        return results
```

### C++ Optimization Techniques
```cpp
// 1. Memory optimization
std::vector<int> numbers;
numbers.reserve(n);  // Pre-allocate memory

// 2. Loop optimization
#pragma omp parallel for  // OpenMP parallelization
for (int i = 0; i < n; ++i) {
    // Optimized loop body
}

// 3. Data type optimization
uint64_t value;  // Use appropriate integer types
double result;   // Use double for precision

// 4. Compiler optimization flags
// -O3: Maximum optimization
// -march=armv8.3-a: ARM-specific optimizations
// -std=c++17: Modern C++ features
```

## 📊 Model Performance Comparison

| Model | Code Quality | Generation Speed | Cost | Use Case |
|-------|--------------|------------------|------|----------|
| **GPT-4o** | Excellent | Fast | High | Production code generation |
| **GPT-4o-mini** | Good | Very Fast | Low | Cost-effective generation |
| **Claude-3-5-Sonnet** | Excellent | Fast | High | Complex algorithms |
| **Claude-3-Haiku** | Good | Very Fast | Low | Simple translations |
| **Open Source** | Variable | Medium | Very Low | Custom deployments |

## 🚀 Advanced Features

### Custom Model Deployment
```python
# Deploy custom model on HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Deploy to HuggingFace endpoint
model.push_to_hub("your-username/your-model")
```

### Code Quality Assessment
```python
def assess_code_quality(cpp_code):
    # Compilation check
    try:
        subprocess.run(["clang++", "-std=c++17", "-o", "test", "-"], 
                      input=cpp_code.encode(), check=True)
        compilation_score = 1.0
    except subprocess.CalledProcessError:
        compilation_score = 0.0
    
    # Performance check
    performance_score = measure_performance(cpp_code)
    
    # Style check
    style_score = check_coding_style(cpp_code)
    
    return {
        "compilation": compilation_score,
        "performance": performance_score,
        "style": style_score,
        "overall": (compilation_score + performance_score + style_score) / 3
    }
```

### Automated Testing
```python
def test_code_generation():
    test_cases = [
        "simple_loop.py",
        "complex_algorithm.py",
        "data_structures.py"
    ]
    
    results = {}
    for test_case in test_cases:
        python_code = load_test_case(test_case)
        cpp_code = generate_code(python_code)
        
        # Test compilation
        compilation_success = compile_cpp(cpp_code)
        
        # Test correctness
        correctness = test_correctness(python_code, cpp_code)
        
        # Test performance
        performance_improvement = measure_performance_improvement(python_code, cpp_code)
        
        results[test_case] = {
            "compilation": compilation_success,
            "correctness": correctness,
            "performance": performance_improvement
        }
    
    return results
```

## 🔮 Potential Extensions

### Enhanced Code Generation
- **Multi-language Support**: Generate code in multiple languages
- **Framework Integration**: Generate code for specific frameworks
- **Testing Code**: Generate unit tests and integration tests
- **Documentation**: Generate code documentation and comments

### Performance Optimization
- **Algorithm Analysis**: Analyze and optimize algorithms
- **Memory Profiling**: Optimize memory usage patterns
- **Parallel Processing**: Generate parallel implementations
- **GPU Acceleration**: Generate CUDA/OpenCL code

### Integration Possibilities
- **IDE Integration**: Code generation within development environments
- **CI/CD Pipeline**: Automated code optimization in build processes
- **Code Review**: Automated code review and suggestions
- **Refactoring**: Automated code refactoring and modernization

### Advanced Applications
- **Domain-Specific Languages**: Generate DSL implementations
- **Compiler Optimization**: Generate optimized compiler passes
- **Embedded Systems**: Generate code for resource-constrained systems
- **Real-time Systems**: Generate real-time capable implementations

## 📚 Learning Resources

### Code Generation
- [OpenAI Code Generation Guide](https://platform.openai.com/docs/guides/code)
- [Anthropic Claude Code Generation](https://docs.anthropic.com/claude/docs/code-generation)
- [HuggingFace Code Models](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)

### C++ Optimization
- [C++ Performance Best Practices](https://en.cppreference.com/w/cpp/language)
- [Compiler Optimization Flags](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- [Modern C++ Features](https://isocpp.org/std/the-standard)

### Model Deployment
- [HuggingFace Endpoints](https://huggingface.co/docs/inference-endpoints)
- [Model Deployment Best Practices](https://huggingface.co/docs/hub/model-repos)
- [Cost Optimization Strategies](https://huggingface.co/docs/inference-endpoints/guides/cost_optimization)

### Performance Analysis
- [Profiling Tools](https://perf.wiki.kernel.org/index.php/Main_Page)
- [Benchmarking Techniques](https://github.com/google/benchmark)
- [Performance Measurement](https://en.cppreference.com/w/cpp/chrono)

## 🤝 Contributing

To contribute to these code generation tutorials:
1. Fork the repository
2. Add new code generation examples
3. Improve existing optimizations
4. Test with different models and languages
5. Submit a pull request

## ⚠️ Important Notes

### Cost Management
- **API Usage**: Monitor API calls and costs
- **Endpoint Management**: Pause HuggingFace endpoints when not in use
- **Model Selection**: Choose appropriate models for your use case
- **Batch Processing**: Group requests to reduce costs

### Code Quality
- **Testing**: Always test generated code thoroughly
- **Security**: Review generated code for security vulnerabilities
- **Maintainability**: Ensure generated code is readable and maintainable
- **Documentation**: Add appropriate comments and documentation

### Performance Considerations
- **Compilation Time**: Consider compilation overhead for large projects
- **Memory Usage**: Monitor memory consumption of generated code
- **Optimization Trade-offs**: Balance performance vs. readability
- **Platform Differences**: Test on different platforms and architectures

## 🎯 Use Cases

### Business Applications
- **Legacy Code Migration**: Convert old Python code to modern C++
- **Performance Critical Systems**: Optimize bottlenecks in applications
- **Algorithm Implementation**: Generate efficient algorithm implementations
- **Prototype to Production**: Convert prototypes to production code

### Educational Applications
- **Learning C++**: Generate examples for C++ learning
- **Algorithm Visualization**: Generate code for algorithm demonstrations
- **Performance Comparison**: Show performance differences between languages
- **Code Review**: Generate code for review and analysis

### Personal Projects
- **Performance Optimization**: Optimize personal project code
- **Language Learning**: Learn new programming languages through translation
- **Code Generation**: Automate repetitive coding tasks
- **Algorithm Implementation**: Implement complex algorithms efficiently

---

**Disclaimer**: Generated code should always be reviewed and tested before use in production. These tutorials are for educational purposes and demonstrate the capabilities of LLM-powered code generation. 