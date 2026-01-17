# DSPy Demo Project

A comprehensive collection of DSPy tutorials, examples, and runnable code snippets for learning and reference.

## Overview

This project contains implementations of all major DSPy tutorials and features, organized into notebooks and scripts for easy learning and experimentation. Each example is self-contained and includes detailed explanations.

**✨ Recently Updated**: This project has been upgraded to **DSPy 3.1.0** with support for **Python 3.10-3.14**. The project now includes:
- **New DSPy 3.x features**: Reasoning, Multimodal (Image/Audio), Adapters, GEPA & SIMBA optimizers, Usage Tracking
- **Restructured learning path**: Beginner-friendly organization with dedicated sections for fundamentals and core modules
- **Updated utilities**: Enhanced helpers for usage tracking, adapters, and multiple LLM providers (including Groq)

The project uses [UV](https://docs.astral.sh/uv/) for fast, reliable dependency management and virtual environment handling.

## Key Features

- 🚀 **DSPy 3.1.0** - Latest DSPy with reasoning, multimodal, and advanced optimization
- 🐍 **Python 3.10-3.14** - Broad Python version compatibility
- 📦 **UV Package Manager** - Lightning-fast dependency resolution and installation
- 🔒 **Locked Dependencies** - Reproducible builds with `uv.lock`
- 📚 **40+ Examples** - Comprehensive tutorials covering all DSPy 3.x features
- 🎓 **Beginner-Friendly** - Restructured learning path from fundamentals to advanced
- 🔧 **Development Ready** - Pre-commit hooks, testing, and type checking included
- 📖 **Dual Format** - Both Jupyter notebooks and Python scripts for each example
- 🆕 **New in 3.x**: Reasoning models, multimodal AI, adapters, GEPA/SIMBA optimizers, usage tracking

## Project Structure

```
dspy-demo/
├── notebooks/          # Jupyter notebooks for interactive learning
│   ├── 01_fundamentals/     # Core DSPy concepts (NEW structure)
│   ├── 02_core_modules/     # Built-in modules (Predict, ChainOfThought, Reasoning, etc.) NEW!
│   ├── 03_building_programs/ # Building AI programs
│   ├── 04_optimization/     # Optimization techniques (includes GEPA, SIMBA) NEW!
│   ├── 05_advanced/         # Advanced features (multimodal, adapters, usage tracking) NEW!
│   ├── 06_deployment/       # Production deployment patterns
│   └── 07_real_world/       # Real-world applications
├── scripts/            # Python scripts (mirrors notebooks structure)
│   └── [same structure as notebooks/]
├── data/               # Sample datasets
├── models/             # Saved models and configurations
├── utils/              # Helper utilities (updated for DSPy 3.x)
└── docs/               # Documentation
    └── LEARNING_PATH.md # Recommended learning order NEW!
```

**Note**: Old directory structure (`01_basics/`, `02_building/`, etc.) is preserved for backward compatibility.

## Getting Started

### Prerequisites

- **Python 3.10 or higher** (supported: 3.10, 3.11, 3.12, 3.13, 3.14)
- **UV package manager** - Fast Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Quick Setup

Follow these steps to get the project running on your machine:

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd dspy-demo
```

#### 2. Install Python (if needed)

UV can automatically install Python for you:

```bash
# UV will install a compatible Python version (3.10-3.14)
uv python install 3.13
```

Or verify your Python version:

```bash
python --version  # Should be 3.10 or higher (up to 3.14)
```

#### 3. Install Dependencies

UV will create a virtual environment and install all dependencies:

```bash
# This reads pyproject.toml and uv.lock to install exact versions
uv sync
```

This command will:

- Create a virtual environment with your Python version (3.10-3.14)
- Install all production and development dependencies
- Ensure reproducible builds using the lockfile

#### 4. Verify Installation

```bash
# Check that DSPy and other key packages are installed correctly
uv run python -c "import dspy; print(f'DSPy version: {dspy.__version__}')"
uv run python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

#### 5. Set Up Environment Variables

Create a `.env` file for your API keys:

```bash
# Copy the example file (if it exists)
cp .env.example .env

# Or create a new .env file
touch .env
```

Edit `.env` and add your API keys (see [API Keys Required](#api-keys-required) section below).

### Running Examples

#### Option 1: Using UV Run (Recommended)

UV can run commands in the project environment without activation:

```bash
# Run Jupyter Lab
uv run jupyter lab

# Run a specific script
uv run python scripts/01_basics/getting_started.py

# Run with additional arguments
uv run python scripts/02_building/rag_system.py --model gpt-4o
```

#### Option 2: Activate Environment

```bash
# Activate the virtual environment
uv shell

# Now you can run commands directly
jupyter lab
python scripts/01_basics/getting_started.py
```

### Development Setup

If you plan to contribute or modify the code:

```bash
# Install with development dependencies (already included in uv sync)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Format code
uv run black .

# Type checking
uv run mypy .
```

### Updating Dependencies

To update to the latest compatible versions:

```bash
# Update all dependencies
uv lock --upgrade

# Sync the updated lockfile
uv sync
```

## Tutorial Categories

### 1. Fundamentals (Start Here!)

- **Getting Started** - Introduction to DSPy basics
- **Signatures** - Defining input/output specifications NEW!
- **Modules** - Building reusable components NEW!
- **Predictions** - Working with model outputs NEW!

### 2. Core Modules (Essential Skills)

- **Predict** - Basic predictions
- **ChainOfThought** - Reasoning step-by-step
- **ProgramOfThought** - Code-based reasoning
- **ReAct** - Building agents with ReAct
- **Reasoning** - Using reasoning models (o1, Claude thinking) 🆕 NEW in 3.x!

### 3. Building Programs (Applications)

- **Custom Modules** - Creating custom DSPy modules
- **RAG Systems** - Retrieval-Augmented Generation
- **Classification** - Text classification tasks
- **Entity Extraction** - Named entity recognition
- **Multi-Stage Pipelines** - Complex workflows
- **Customer Service Agent** - Building intelligent agents
- **Image Generation** - Prompt iteration for images
- **Audio Processing** - Speech and audio tasks

### 4. Optimization (Performance)

- **BootstrapFewShot** - Quick optimization
- **MIPROv2** - Bayesian optimization (UPDATED for 3.x)
- **GEPA** - Genetic-Pareto optimization 🆕 NEW in 3.x!
- **SIMBA** - Self-reflective improvement 🆕 NEW in 3.x!
- **Finetuning** - Model fine-tuning

### 5. Advanced Features (Cutting Edge)

- **Multimodal** - Images and audio 🆕 NEW in 3.x!
- **Adapters** - ChatAdapter, JSONAdapter, XMLAdapter 🆕 NEW in 3.x!
- **Async & Batch** - Performance optimization
- **Usage Tracking** - Monitor costs and usage 🆕 NEW in 3.x!
- **Multi-Hop RAG** - Advanced RAG techniques
- **RL Optimization** - Reinforcement learning approaches

### 6. Deployment & Production

- **Saving/Loading** - Model persistence (updated for 3.x)
- **Streaming** - Real-time processing
- **Caching** - Performance optimization
- **Production Tips** - Best practices
- **MCP Integration** - Model Context Protocol
- **Output Refinement** - Best-of-N techniques

### 7. Real-World Applications

- **Financial Analysis** - Yahoo Finance integration
- **Email Extraction** - Email information processing
- **AI Text Game** - Creative text-based games
- **Code Generation** - Automated code creation
- **Memory-Enabled Agents** - Persistent memory systems

## 📚 Learning Path

New to DSPy? Follow the recommended learning path in [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) for:
- **Quick Start**: Get running in 30 minutes
- **Beginner Track**: Fundamentals to building programs (2-4 hours)
- **Intermediate Track**: Optimization and advanced features (4-8 hours)
- **Advanced Track**: Deployment and real-world applications (8+ hours)
- **Specialized Paths**: Conversational AI, Data Extraction, RAG Systems, Research, Multimodal AI

## 🆕 What's New in DSPy 3.x

### Major New Features

1. **dspy.Reasoning** - Native support for reasoning models
   - OpenAI o1, Claude with extended thinking
   - Captures and exposes reasoning traces
   - See: `scripts/02_core_modules/05_reasoning.py`

2. **Multimodal Support** - First-class image and audio
   - `dspy.Image` for vision tasks
   - `dspy.Audio` for speech/audio processing
   - Works with GPT-4o, Claude 3+, Gemini 1.5+
   - See: `scripts/05_advanced/01_multimodal.py`

3. **Adapters** - Control output structure
   - `ChatAdapter` for conversational interfaces
   - `JSONAdapter` for structured JSON outputs
   - `XMLAdapter` for XML-structured data
   - See: `scripts/05_advanced/02_adapters.py`

4. **Advanced Optimizers**
   - **GEPA**: Genetic-Pareto with reflective improvement
   - **SIMBA**: Self-reflective improvement rules
   - **MIPROv2**: Updated Bayesian optimization
   - See: `scripts/04_optimization/`

5. **Usage Tracking** - Built-in cost monitoring
   - Track tokens and costs automatically
   - Per-module usage tracking
   - Cost alerts and budgeting
   - See: `scripts/05_advanced/04_usage_tracking.py`

### Migration from DSPy 2.x

✅ **No breaking changes!** DSPy 3.x is fully backward compatible with 2.x code.

New features are additive - your existing code continues to work, and you can adopt new features incrementally.

## API Keys Required

You'll need API keys for various services depending on which examples you want to run. Add them to your `.env` file:

```env
# OpenAI (required for most examples)
OPENAI_API_KEY=your_openai_key_here

# Anthropic Claude (for Claude examples)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google Gemini (for Google examples)
GOOGLE_API_KEY=your_google_key_here

# Groq (for Groq examples)
GROQ_API_KEY=your_groq_key_here

# Optional: Other services
COHERE_API_KEY=your_cohere_key_here
TOGETHER_API_KEY=your_together_key_here
```

### Getting API Keys

- **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com/)
- **Anthropic**: Sign up at [console.anthropic.com](https://console.anthropic.com/)
- **Google**: Get API key from [Google AI Studio](https://aistudio.google.com/)
- **Groq**: Sign up at [console.groq.com](https://console.groq.com/)

## Troubleshooting

### Common Issues

#### Python Version Issues

```bash
# Check your Python version
python --version

# If you have multiple Python versions, use UV to manage them
uv python list
uv python install 3.12.11
```

#### Dependency Conflicts

```bash
# Clean install - remove existing environment
rm -rf .venv
uv sync

# Or force reinstall
uv sync --reinstall
```

#### Missing API Keys

```bash
# Verify your .env file exists and has the right format
cat .env

# Test API key (replace with your key)
uv run python -c "import openai; print('OpenAI client created successfully')"
```

#### Jupyter Notebook Issues

```bash
# Install Jupyter kernel for the virtual environment
uv run python -m ipykernel install --user --name dspy-demo

# Start Jupyter with the correct kernel
uv run jupyter lab
```

#### Package Import Errors

```bash
# Verify packages are installed
uv run pip list | grep dspy
uv run python -c "import dspy; print('Success!')"

# Reinstall if needed
uv sync --reinstall
```

### Performance Tips

- **GPU Support**: For faster model inference, install PyTorch with CUDA support
- **Memory**: Some examples require significant RAM (8GB+ recommended)
- **API Limits**: Be aware of rate limits when running multiple examples

## Project Information

### Technical Specifications

- **Python Version**: 3.10-3.14 (requires >=3.10, <3.15)
- **Package Manager**: UV (uv.lock ensures reproducible installs)
- **Key Dependencies**:
  - **DSPy: 3.1.0** (upgraded from 2.5.28)
  - OpenAI: 1.51.0+
  - Anthropic: 0.37.0+
  - PyTorch: 2.4.0+
  - Transformers: 4.45.0+
  - NumPy: 2.0.0+
  - Pandas: 2.2.0+

### Recent Updates (v3.0)

This project was upgraded to **DSPy 3.1.0** with:

✨ **New Features**:
- Reasoning module for o1/Claude thinking models
- Multimodal support (Image/Audio)
- Adapters (Chat/JSON/XML)
- GEPA and SIMBA optimizers
- Built-in usage tracking

📁 **Restructured**:
- Beginner-friendly learning progression
- Dedicated core modules section
- Updated fundamentals tutorials

🔧 **Enhanced**:
- Groq provider support
- Updated model defaults (Claude 3.5 Sonnet, Gemini 1.5 Pro)
- Usage tracking utilities
- Adapter helper functions

See [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) and [`CLAUDE.md`](CLAUDE.md) for more details.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DSPy Team](https://github.com/stanfordnlp/dspy) for the amazing framework
- [Astral](https://astral.sh/) for UV package manager
