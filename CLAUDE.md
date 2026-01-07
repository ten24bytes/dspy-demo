# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive DSPy demo project showcasing 30+ tutorials and examples for learning the DSPy framework. DSPy is a framework for building AI applications through "programming rather than prompting" - using signatures, modules, and optimizers to create robust LLM-powered programs.

The project uses:
- **Python 3.13.11** (requires >=3.12.11, <3.14)
- **UV package manager** for fast dependency management
- **DSPy 2.5.28+** as the core framework
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
- `01_basics/` - Core DSPy concepts
- `02_building/` - Building AI programs
- `03_optimization/` - Optimization techniques
- `04_advanced/` - Advanced features (RL optimization)
- `05_deployment/` - Production deployment patterns
- `06_real_world/` - Real-world examples

### Utility System

**`utils/` Package**: Centralized utilities for consistent setup across all examples

Key modules:
- **`utils/__init__.py`**: LM configuration helpers and output formatting
  - `setup_openai_lm()`, `setup_anthropic_lm()`, `setup_google_lm()` - Provider-specific LM setup
  - `setup_default_lm(provider="openai", **kwargs)` - Unified LM configuration
  - `configure_dspy(lm=None, **kwargs)` - DSPy configuration wrapper
  - Color-coded terminal output: `print_step()`, `print_result()`, `print_error()`, `print_warning()`

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
- Supported providers: "openai", "anthropic", "google"
- Provider-specific functions available: `setup_openai_lm()`, `setup_anthropic_lm()`, `setup_google_lm()`

### Testing Changes

- Test both notebook and script versions if modifying shared utilities
- Verify `.env` configuration is correct (copy from `.env.example`)
- Run `uv run pytest` for automated tests
- Use `uv run python scripts/path/to/script.py` to test individual examples

### Code Style

- Project uses Black (line length 88) and isort for formatting
- Python 3.13+ features are allowed (requires >=3.12.11, <3.14)
- Type hints preferred (MyPy configured in pyproject.toml)
- Clear docstrings required for all modules and functions
