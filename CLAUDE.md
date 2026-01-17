# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive DSPy demo project showcasing 30+ tutorials and examples for learning the DSPy framework. DSPy is a framework for building AI applications through "programming rather than prompting" - using signatures, modules, and optimizers to create robust LLM-powered programs.

The project uses:
- **Python 3.10-3.14** (requires >=3.10, <3.15)
- **UV package manager** for fast dependency management
- **DSPy 3.1.0** as the core framework (upgraded from 2.5.28)
- Multiple LLM providers (OpenAI, Anthropic, Google, Groq)

## Common Commands

### Environment Setup

```bash
# Install dependencies with UV (creates .venv automatically)
uv sync

# Install with dev dependencies
uv sync --dev

# Update dependencies
uv lock --upgrade && uv sync

# Activate virtual environment
uv shell
```

### Running Examples

```bash
# Run Jupyter Lab for interactive notebooks
uv run jupyter lab

# Run a specific script directly
uv run python scripts/01_basics/getting_started.py

# Run with custom model argument (if script supports it)
uv run python scripts/02_building/rag_system.py --model gpt-4o
```

### Development Tools

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_specific.py

# Format code with Black
uv run black .

# Sort imports
uv run isort .

# Type checking
uv run mypy .

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## Architecture & Code Structure

### Dual-Format Design

The project maintains **parallel implementations** of each tutorial:
- **Notebooks** (`notebooks/`) - Interactive Jupyter notebooks for learning
- **Scripts** (`scripts/`) - Standalone Python scripts for automation

Both formats cover identical content and are organized into matching subdirectories:
- `01_fundamentals/` - Core DSPy concepts (renamed from `01_basics/`)
- `02_core_modules/` - Built-in DSPy modules (Predict, ChainOfThought, Reasoning, etc.) **NEW**
- `03_building_programs/` - Building AI programs (renamed from `02_building/`)
- `04_optimization/` - Optimization techniques (renamed from `03_optimization/`)
- `05_advanced/` - Advanced features (multimodal, adapters, usage tracking) (renamed from `04_advanced/`)
- `06_deployment/` - Production deployment patterns (renamed from `05_deployment/`)
- `07_real_world/` - Real-world examples (renamed from `06_real_world/`)

**Note**: The old directory structure (`01_basics/`, `02_building/`, etc.) is still present for backward compatibility, but new work should use the updated structure above.

### Utility System

**`utils/` Package**: Centralized utilities for consistent setup across all examples

Key modules:
- **`utils/__init__.py`**: LM configuration helpers and output formatting
  - `setup_openai_lm()`, `setup_anthropic_lm()`, `setup_google_lm()`, `setup_groq_lm()` - Provider-specific LM setup **UPDATED**
  - `setup_default_lm(provider="openai", **kwargs)` - Unified LM configuration
  - `configure_dspy(lm=None, track_usage=False, adapter=None, **kwargs)` - DSPy configuration wrapper **UPDATED**
  - Color-coded terminal output: `print_step()`, `print_result()`, `print_error()`, `print_warning()`
  - **DSPy 3.x helpers**: `get_usage_stats()`, `print_usage_stats()` - Usage tracking **NEW**
  - **Adapter helpers**: `create_chat_adapter()`, `create_json_adapter()`, `create_xml_adapter()` **NEW**

- **`utils/datasets.py`**: Sample dataset loaders
  - `get_sample_qa_data()` - Question-answering examples
  - `get_sample_classification_data()` - Sentiment classification examples
  - `get_sample_entity_data()` - Named entity recognition examples
  - `get_sample_math_data()` - Mathematical reasoning problems
  - `get_sample_rag_documents()` - Documents for RAG examples
  - `save_dataset()` / `load_dataset()` - Dataset persistence

All scripts import from `utils` for consistent LM setup and data loading.

### DSPy Module Patterns

**Common Module Structure**: Most examples follow this pattern:

1. **Signature Definition**: Define input/output specifications
   ```python
   class TaskSignature(dspy.Signature):
       """Docstring describing the task."""
       input_field = dspy.InputField(desc="Description")
       output_field = dspy.OutputField(desc="Description")
   ```

2. **Module Implementation**: Create custom modules inheriting `dspy.Module`
   ```python
   class CustomModule(dspy.Module):
       def __init__(self):
           super().__init__()
           self.predictor = dspy.Predict(TaskSignature)
           # or dspy.ChainOfThought(TaskSignature)

       def forward(self, **inputs):
           result = self.predictor(**inputs)
           return dspy.Prediction(...)
   ```

3. **Retrieval Pattern** (for RAG systems): Custom retriever class + RAG module
   - Retriever provides `retrieve(query, k)` method returning top-k documents
   - RAG module combines retrieval + generation in `forward()`

### Data & Model Directories

- **`data/`**: Sample datasets (JSON files) and knowledge bases
  - Automatically loaded by `utils/datasets.py` functions
  - Contains: `qa_dataset.json`, `classification_dataset.json`, `entity_extraction_dataset.json`

- **`models/`**: Pre-trained/optimized DSPy models and configurations
  - Organized by use case: `basic_qa/`, `classification/`, `rag_systems/`
  - Each model directory includes: `model.json`, `metadata.json`, `config.json`, `README.md`

### Environment Configuration

**`.env` file**: Required API keys and configuration (see `.env.example`)

Critical environment variables:
- `OPENAI_API_KEY` - Required for most examples
- `ANTHROPIC_API_KEY` - For Claude models
- `GOOGLE_API_KEY` - For Gemini models
- `GROQ_API_KEY` - For Groq models
- `DEFAULT_LM_PROVIDER` - Default provider (default: "openai")
- `DEFAULT_LM_MODEL` - Default model (default: "gpt-4o")

The `utils/__init__.py` module automatically loads `.env` via `python-dotenv`.

### Key Implementation Patterns

**Standard Script Structure**:
```python
#!/usr/bin/env python3
from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result
import dspy
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def main():
    load_dotenv('.env')
    lm = setup_default_lm(provider="openai", model="gpt-4o")
    dspy.configure(lm=lm)
    # ... implementation

if __name__ == "__main__":
    main()
```

**RAG System Pattern**:
- Custom retriever class with `retrieve(query, k)` method
- RAG module inheriting `dspy.Module`
- `forward()` method: retrieve documents → combine context → generate answer
- See `scripts/02_building/rag_system.py` for reference implementation

**Optimization Pattern**:
- Define metric function for evaluation
- Create optimizer (e.g., `dspy.BootstrapFewShot()`, `dspy.MIPROv2()`)
- Compile module: `optimized = optimizer.compile(module, trainset=data, metric=metric)`
- See `scripts/03_optimization/` for examples

## Development Guidelines

### Adding New Examples

1. Create parallel implementations in `notebooks/` and `scripts/` under appropriate category
2. Use `utils` functions for LM setup and consistent output formatting
3. Import sample data from `utils/datasets.py` when possible
4. Include docstring explaining the example's purpose
5. Follow existing naming conventions (lowercase with underscores)

### Working with LLM Providers

- All provider setup is centralized in `utils/__init__.py`
- Use `setup_default_lm(provider="...")` for consistent configuration
- Supported providers: "openai", "anthropic", "google", "groq" **UPDATED**
- Provider-specific functions available: `setup_openai_lm()`, `setup_anthropic_lm()`, `setup_google_lm()`, `setup_groq_lm()` **UPDATED**
- Default models updated for DSPy 3.x:
  - OpenAI: gpt-4o (unchanged)
  - Anthropic: claude-3-5-sonnet-20241022 (upgraded from claude-3-haiku)
  - Google: gemini-1.5-pro (upgraded from gemini-pro)
  - Groq: llama-3.3-70b-versatile **NEW**

### Testing Changes

- Test both notebook and script versions if modifying shared utilities
- Verify `.env` configuration is correct (copy from `.env.example`)
- Run `uv run pytest` for automated tests
- Use `uv run python scripts/path/to/script.py` to test individual examples

### Code Style

- Project uses Black (line length 88) and isort for formatting
- Python 3.10+ features are allowed (requires >=3.10, <3.15) **UPDATED**
- Type hints preferred (MyPy configured in pyproject.toml)
- Clear docstrings required for all modules and functions

## DSPy 3.x Features and Updates

### New in DSPy 3.1.0

This project has been updated to DSPy 3.1.0 with the following new features:

#### 1. Reasoning Module (`dspy.Reasoning`)
- Native support for reasoning-capable models (OpenAI o1, Claude with thinking)
- Captures and exposes model reasoning traces
- See: `scripts/02_core_modules/05_reasoning.py`

#### 2. Multimodal Support (`dspy.Image`, `dspy.Audio`)
- First-class support for images and audio in signatures
- Works with vision models (GPT-4o, Claude 3+, Gemini 1.5+)
- See: `scripts/05_advanced/01_multimodal.py`

#### 3. Adapters (`dspy.ChatAdapter`, `dspy.JSONAdapter`, `dspy.XMLAdapter`)
- Control prompt formatting and output structure
- ChatAdapter: For conversational interfaces
- JSONAdapter: For structured JSON outputs
- XMLAdapter: For XML-structured data
- See: `scripts/05_advanced/02_adapters.py`

#### 4. Advanced Optimizers
- **GEPA**: Genetic-Pareto optimizer with reflective improvement
- **SIMBA**: Self-reflective improvement rules
- **MIPROv2**: Updated with Bayesian optimization
- See: `scripts/04_optimization/03_gepa.py`

#### 5. Usage Tracking
- Built-in token and cost monitoring
- Enable with `configure_dspy(track_usage=True)`
- Access stats with `dspy.get_lm_usage()` or `get_usage_stats()`
- See: `scripts/05_advanced/04_usage_tracking.py`

#### 6. Updated APIs
- Package name remains `dspy-ai` on PyPI (imports as `dspy`)
- No breaking changes from 2.x to 3.x
- All existing code continues to work
- New features are additive

### Migration from DSPy 2.x

No code changes required! DSPy 3.x is backward compatible with 2.x.

To take advantage of new features:
1. Update pyproject.toml: `dspy>=3.1.0`
2. Run: `uv sync`
3. Start using new features as needed

### New Module Patterns

**Using Reasoning:**
```python
class ProblemSolver(dspy.Signature):
    problem = dspy.InputField()
    reasoning = dspy.OutputField()  # Captures model's reasoning
    solution = dspy.OutputField()

reasoner = dspy.Reasoning(ProblemSolver)
result = reasoner(problem="Complex problem here")
print(result.reasoning)  # View the reasoning process
```

**Using Multimodal:**
```python
class ImageAnalyzer(dspy.Signature):
    image = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

analyzer = dspy.Predict(ImageAnalyzer)
img = dspy.Image(path="photo.jpg")
result = analyzer(image=img, question="What's in this image?")
```

**Using Adapters:**
```python
# Configure with JSON adapter for structured outputs
json_adapter = dspy.JSONAdapter()
configure_dspy(lm=lm, adapter=json_adapter)

# Your modules now encourage JSON outputs
```

**Using Usage Tracking:**
```python
# Enable tracking
configure_dspy(lm=lm, track_usage=True)

# Make predictions
result = predictor(input="...")

# Check usage
usage = get_usage_stats()
print(f"Tokens used: {usage['total_tokens']}")
print_usage_stats()  # Pretty-printed stats
```

### Updated Optimization Patterns

**GEPA Optimizer:**
```python
optimizer = dspy.GEPA(
    metric=my_metric,
    population_size=15,
    num_generations=7,
    reflection_enabled=True
)
optimized = optimizer.compile(student=MyModule(), trainset=data)
```

See `docs/LEARNING_PATH.md` for a complete guide to learning DSPy 3.x features.
