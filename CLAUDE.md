# CLAUDE.md

## Project Overview

DSPy demo project with 30+ tutorials for learning DSPy 3.1.0 — a framework for building AI applications through "programming rather than prompting." Uses Python 3.10-3.14, UV package manager, and multiple LLM providers (OpenAI, Anthropic, Google, Groq).

## Common Commands

```bash
# Dependencies
uv sync                              # Install (creates .venv)
uv sync --dev                        # Install with dev deps
uv lock --upgrade && uv sync         # Update all deps

# Running
uv run jupyter lab                   # Interactive notebooks
uv run python scripts/01_fundamentals/getting_started.py  # Run a script

# Dev tools
uv run pytest                        # Tests
uv run black .                       # Format
uv run isort .                       # Sort imports
uv run mypy .                        # Type check
uv run pre-commit run --all-files    # All pre-commit hooks
```

## Architecture

### Dual-Format Design

Every tutorial exists as both a Jupyter notebook (`notebooks/`) and a Python script (`scripts/`). Both are organized into matching subdirectories:

| Directory | Content |
|---|---|
| `01_fundamentals/` | Core DSPy concepts |
| `02_core_modules/` | Built-in modules (Predict, ChainOfThought, Reasoning) |
| `03_building_programs/` | Building AI programs (RAG, agents) |
| `04_optimization/` | Optimization techniques |
| `05_advanced/` | Multimodal, adapters, usage tracking |
| `06_deployment/` | Production deployment patterns |
| `07_real_world/` | Real-world examples |

**Note:** Old directory names (`01_basics/`, `02_building/`, etc.) still exist for backward compatibility. New work should use the structure above.

### Key Directories

- **`utils/`** — Centralized LM setup (`setup_default_lm()`, `configure_dspy()`), color-coded output helpers (`print_step()`, `print_result()`, `print_error()`), dataset loaders, adapter/usage-tracking helpers. All scripts import from here. See source for full API.
- **`data/`** — Sample JSON datasets. Note: only `qa_dataset.json` is loaded from disk by `utils/datasets.py`; other dataset functions return hardcoded examples.
- **`models/`** — Minimal pre-trained model configs (only `basic_qa/` and `configs/` exist).
- **`docs/LEARNING_PATH.md`** — Structured curriculum for learning DSPy progressively.

### Environment

API keys and config go in `.env` (copy from `.env.example`). The `utils` package auto-loads it via `python-dotenv`. See `.env.example` for all supported variables.

### Standard Script Structure

All scripts follow this pattern: set `sys.path`, load `.env`, call `setup_default_lm()`, configure DSPy, then implement. See any script in `scripts/01_fundamentals/` for a concrete example.

## Development Guidelines

- Create parallel implementations in both `notebooks/` and `scripts/` under the appropriate category
- Use `utils` functions for LM setup and output formatting — don't configure providers directly
- Import sample data from `utils/datasets.py` when possible
- Follow existing naming: lowercase with underscores
- Code style: Black (line length 88), isort, type hints preferred
- Test both notebook and script versions when modifying shared utilities

## DSPy 3.x Features

DSPy 3.1.0 is backward compatible with 2.x — no migration needed. New features (with reference scripts):

- **Reasoning** (`dspy.Reasoning`) — reasoning-capable model support → `scripts/02_core_modules/05_reasoning.py`
- **Multimodal** (`dspy.Image`, `dspy.Audio`) — image/audio in signatures → `scripts/05_advanced/01_multimodal.py`
- **Adapters** (`ChatAdapter`, `JSONAdapter`, `XMLAdapter`) — prompt/output formatting → `scripts/05_advanced/02_adapters.py`
- **Advanced Optimizers** (GEPA, SIMBA, MIPROv2) → `scripts/04_optimization/03_gepa.py`
- **Usage Tracking** — token/cost monitoring via `configure_dspy(track_usage=True)` → `scripts/05_advanced/04_usage_tracking.py`
