"""
Common utilities for DSPy demo project.
"""

import os
import dspy
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()


def setup_openai_lm(model: str = "gpt-4o", max_tokens: int = 1000) -> dspy.LM:
    """Set up OpenAI language model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Optional organization ID - only use if explicitly set and not a placeholder
    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id and org_id.startswith("your_") or org_id == "your_openai_org_id_here":
        org_id = None  # Ignore placeholder values

    # Create LM configuration
    lm_config = {
        "model": f"openai/{model}",
        "api_key": api_key,
        "max_tokens": max_tokens
    }

    # Only add organization if it's explicitly set and valid
    if org_id:
        lm_config["organization"] = org_id

    return dspy.LM(**lm_config)


def setup_anthropic_lm(model: str = "claude-3-haiku-20240307", max_tokens: int = 1000) -> dspy.LM:
    """Set up Anthropic language model."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    return dspy.LM(
        model=f"anthropic/{model}",
        api_key=api_key,
        max_tokens=max_tokens
    )


def setup_google_lm(model: str = "gemini-pro", max_tokens: int = 1000) -> dspy.LM:
    """Set up Google language model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    return dspy.LM(
        model=f"google/{model}",
        api_key=api_key,
        max_tokens=max_tokens
    )


def setup_default_lm(provider: str = "openai", **kwargs) -> Optional[dspy.LM]:
    """Set up default language model based on provider."""
    providers = {
        "openai": setup_openai_lm,
        "anthropic": setup_anthropic_lm,
        "google": setup_google_lm
    }

    if provider not in providers:
        print_error(f"Unsupported provider: {provider}. Choose from {list(providers.keys())}")
        return None

    try:
        lm = providers[provider](**kwargs)
        dspy.configure(lm=lm)
        print_result(f"Successfully configured {provider} language model")
        return lm
    except Exception as e:
        print_error(f"Failed to setup {provider} language model: {e}")
        print_warning("Make sure you have set the appropriate API key in your .env file")
        return None


def configure_dspy(lm: Optional[dspy.LM] = None, **kwargs):
    """Configure DSPy with language model."""
    if lm is None:
        lm = setup_default_lm(**kwargs)

    dspy.configure(lm=lm)
    return lm


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_step(step: str, description: str = ""):
    """Print a colored step indicator."""
    print(f"{Colors.OKBLUE}{Colors.BOLD}=== {step} ==={Colors.ENDC}")
    if description:
        print(f"{Colors.OKCYAN}{description}{Colors.ENDC}")
    print()


def print_result(result: Any, label: str = "Result"):
    """Print a colored result."""
    print(f"{Colors.OKGREEN}{Colors.BOLD}{label}:{Colors.ENDC}")
    print(f"{result}")
    print()


def print_error(error: str):
    """Print a colored error message."""
    print(f"{Colors.FAIL}{Colors.BOLD}Error: {error}{Colors.ENDC}")
    print()


def print_warning(warning: str):
    """Print a colored warning message."""
    print(f"{Colors.WARNING}{Colors.BOLD}Warning: {warning}{Colors.ENDC}")
    print()
