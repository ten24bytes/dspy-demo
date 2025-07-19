# DSPy Demo Project

A comprehensive collection of DSPy tutorials, examples, and runnable code snippets for learning and reference.

## Overview

This project contains implementations of all major DSPy tutorials and features, organized into notebooks and scripts for easy learning and experimentation. Each example is self-contained and includes detailed explanations.

**✨ Recently Updated**: This project has been upgraded to **Python 3.12.11** with all dependencies updated to their latest compatible versions. The project uses [UV](https://docs.astral.sh/uv/) for fast, reliable dependency management and virtual environment handling.

## Key Features

- 🚀 **Python 3.12.11** - Latest stable Python with improved performance
- 📦 **UV Package Manager** - Lightning-fast dependency resolution and installation
- 🔒 **Locked Dependencies** - Reproducible builds with `uv.lock`
- 📚 **Comprehensive Examples** - 30+ tutorials covering all DSPy features
- 🔧 **Development Ready** - Pre-commit hooks, testing, and type checking included
- 📖 **Dual Format** - Both Jupyter notebooks and Python scripts for each example

## Project Structure

```
dspy-demo/
├── notebooks/          # Jupyter notebooks for interactive learning
│   ├── 01_basics/      # Basic DSPy concepts
│   ├── 02_building/    # Building AI programs
│   ├── 03_optimization/ # Optimization techniques
│   ├── 04_advanced/    # Advanced features
│   └── 05_deployment/  # Deployment and production
├── scripts/            # Python scripts for each tutorial
│   ├── 01_basics/
│   ├── 02_building/
│   ├── 03_optimization/
│   ├── 04_advanced/
│   └── 05_deployment/
├── data/               # Sample datasets
├── models/             # Saved models and configurations
└── utils/              # Helper utilities
```

## Getting Started

### Prerequisites

- **Python 3.12.11 or higher** (but less than 3.14)
- **UV package manager** - Fast Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Quick Setup

Follow these steps to get the project running on your machine:

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd dspy-demo
```

#### 2. Install Python 3.12.11 (if needed)

UV can automatically install Python for you:

```bash
# UV will install Python 3.12.11 if not already available
uv python install 3.12.11
```

Or verify your Python version:

```bash
python --version  # Should be 3.12.11 or higher
```

#### 3. Install Dependencies

UV will create a virtual environment and install all dependencies:

```bash
# This reads pyproject.toml and uv.lock to install exact versions
uv sync
```

This command will:

- Create a virtual environment with Python 3.12.11
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

### 1. Basic DSPy Concepts

- Getting started with DSPy
- Signatures and modules
- Language model configuration
- Basic prediction and evaluation

### 2. Building AI Programs

- **Customer Service Agent** - Building intelligent agents
- **Custom Modules** - Creating custom DSPy modules
- **RAG Systems** - Retrieval-Augmented Generation
- **RAG as Agent** - Advanced RAG with agent capabilities
- **Entity Extraction** - Named entity recognition
- **Classification** - Text classification tasks
- **Multi-Hop RAG** - Complex multi-step reasoning
- **Privacy-Conscious Delegation** - Secure AI workflows
- **Program of Thought** - Mathematical reasoning
- **Image Generation** - Prompt iteration for images
- **Audio Processing** - Speech and audio tasks

### 3. Optimization Techniques

- **Math Reasoning** - Optimizing mathematical problem solving
- **Classification Finetuning** - Model finetuning for classification
- **Advanced Tool Use** - Complex tool integration
- **Finetuning Agents** - Agent optimization

### 4. Advanced Features

- **RL Optimization** - Reinforcement learning approaches
- **RL Privacy Delegation** - RL for privacy-conscious systems
- **RL Multi-Hop** - RL for complex reasoning

### 5. Development & Deployment

- **MCP Integration** - Model Context Protocol
- **Output Refinement** - Best-of-N and refinement techniques
- **Saving and Loading** - Model persistence
- **Caching** - Performance optimization
- **Deployment** - Production deployment strategies
- **Observability** - Debugging and monitoring
- **Optimizer Tracking** - Tracking optimization progress
- **Streaming** - Real-time processing
- **Async Processing** - Asynchronous operations

### 6. Real-World Examples

- **Financial Analysis** - Yahoo Finance integration
- **Email Extraction** - Email information processing
- **Code Generation** - Automated code creation
- **AI Text Game** - Creative text-based games
- **Memory-Enabled Agents** - Persistent memory systems

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

- **Python Version**: 3.12.11+ (< 3.14)
- **Package Manager**: UV (uv.lock ensures reproducible installs)
- **Key Dependencies**:
  - DSPy: 2.6.27+ (latest framework version)
  - OpenAI: 1.93.0+ (latest API client)
  - PyTorch: 2.7.1+ (latest stable)
  - Transformers: 4.53.0+ (latest Hugging Face)
  - NumPy: 2.2.6+ (latest 2.x series)
  - Pandas: 2.3.0+ (latest stable)

### Recent Updates

This project was recently upgraded from Python 3.11 to 3.12.11 with comprehensive dependency updates. See `PYTHON_UPGRADE_SUMMARY.md` for detailed information about:

- Version migrations and compatibility fixes
- Updated development tools (Black, MyPy, pytest)
- Modern package specifications and resolved conflicts

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DSPy Team](https://github.com/stanfordnlp/dspy) for the amazing framework
- [Astral](https://astral.sh/) for UV package manager
