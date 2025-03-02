# DSPy Demo Project

A comprehensive demonstration project for DSPy, showcasing various advanced features and patterns for programming foundation models.

## Features and Examples

1. **Basic DSPy Usage** (`notebooks/01_dspy_basics.ipynb`)

   - Simple question answering
   - Chain of thought reasoning
   - Custom module creation
   - Working with signatures

2. **Teleprompting** (`notebooks/02_teleprompting.ipynb`)

   - Basic classification with Chain-of-Thought
   - Bootstrapped few-shot learning
   - Adaptive teleprompting
   - Learning from feedback

3. **RAG Examples** (`notebooks/03_rag_examples.ipynb`)

   - Basic RAG with single passage retrieval
   - Multi-hop reasoning
   - Self-correcting RAG
   - Progressive query refinement

4. **Advanced Patterns** (`notebooks/04_advanced_patterns.ipynb`)
   - Signature chaining
   - Dynamic few-shot learning
   - Error handling and self-correction
   - Prompt optimization

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your API keys in a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
```

3. Start with basic examples:

```python
from src.config import setup_dspy
from src.basic_qa import BasicQA

# Initialize DSPy
lm = setup_dspy()

# Create a simple QA module
qa = BasicQA()
result = qa("What is DSPy?")
print(result['answer'])
```

## Advanced Usage Examples

### RAG with Multi-hop Retrieval:

```python
from src.advanced_rag import MultiHopRetriever

retriever = MultiHopRetriever()
result = retriever("What are the environmental impacts of electric cars?")
print(f"Answer: {result['answer']}")
print(f"Number of contexts used: {len(result['contexts'])}")
```

### Self-improving Classification:

```python
from src.teleprompter_example import BootstrappedClassifier

# Create and train classifier
classifier = BootstrappedClassifier()
classifier.compile_with_examples([
    {"text": "Amazing product!", "label": "positive"},
    {"text": "Terrible experience", "label": "negative"}
])

# Use the classifier
result = classifier("This exceeded my expectations")
print(f"Label: {result['label']}, Confidence: {result['confidence']}")
```

### Adaptive Prompting:

```python
from src.optimization_patterns import AdaptivePrompting

adaptive = AdaptivePrompting()
result = adaptive("Explain quantum computing")
print(f"Output: {result['output']}")
print(f"Method used: {result['method']}")
```

## Project Structure

```
src/
├── basic_qa.py         # Basic Q&A implementations
├── advanced_rag.py     # Advanced RAG patterns
├── teleprompter_example.py  # Teleprompting examples
├── optimization_patterns.py  # Advanced optimization patterns
└── config.py          # Configuration utilities

notebooks/
├── 01_dspy_basics.ipynb
├── 02_teleprompting.ipynb
├── 03_rag_examples.ipynb
└── 04_advanced_patterns.ipynb
```

## Configuration

The project supports multiple LLM providers and configuration options. See `src/config.py` for details.

Example configuration:

```python
from src.config import DSPyConfig

config = DSPyConfig()
config.initialize(
    model_name="openai/gpt-4",
    temperature=0.7,
    retriever_config={
        "collection_name": "my_docs",
        "docs_dir": "path/to/docs"
    }
)
```

## Extending the Examples

1. Create custom metrics:

```python
config.register_metric("semantic_similarity", your_similarity_function)
```

2. Add callbacks for monitoring:

```python
config.register_callback(lambda x: print(f"Processing: {x}"))
```

3. Create custom DSPy modules:

```python
class CustomModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought("input -> output")

    def forward(self, input_text):
        return self.process(input=input_text).output
```

## Learn More

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Stanford DSPy Research](https://stanford.edu/~dshahara/dspy.pdf)
