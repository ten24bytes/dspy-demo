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


def setup_anthropic_lm(model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 1000) -> dspy.LM:
    """Set up Anthropic language model.

    Default model updated to Claude 3.5 Sonnet (latest as of DSPy 3.x).
    Other recommended models: claude-3-5-haiku-20241022, claude-3-7-sonnet-20250219
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    return dspy.LM(
        model=f"anthropic/{model}",
        api_key=api_key,
        max_tokens=max_tokens
    )


def setup_google_lm(model: str = "gemini-1.5-pro", max_tokens: int = 1000) -> dspy.LM:
    """Set up Google language model.

    Default model updated to Gemini 1.5 Pro (latest as of DSPy 3.x).
    Other recommended models: gemini-1.5-flash, gemini-2.0-flash-exp
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    return dspy.LM(
        model=f"google/{model}",
        api_key=api_key,
        max_tokens=max_tokens
    )


def setup_groq_lm(model: str = "llama-3.3-70b-versatile", max_tokens: int = 1000) -> dspy.LM:
    """Set up Groq language model.

    Groq provides fast inference for open-source models.
    Popular models: llama-3.3-70b-versatile, llama-3.1-70b-versatile, mixtral-8x7b-32768
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    return dspy.LM(
        model=f"groq/{model}",
        api_key=api_key,
        max_tokens=max_tokens
    )


def setup_default_lm(provider: str = "openai", **kwargs) -> Optional[dspy.LM]:
    """Set up default language model based on provider.

    Supported providers: openai, anthropic, google, groq
    """
    providers = {
        "openai": setup_openai_lm,
        "anthropic": setup_anthropic_lm,
        "google": setup_google_lm,
        "groq": setup_groq_lm
    }

    if provider not in providers:
        print_error(f"Unsupported provider: {provider}. Choose from {list(providers.keys())}")
        return None

    try:
        lm = providers[provider](**kwargs)
        print_result(f"Successfully configured {provider} language model")
        return lm
    except Exception as e:
        print_error(f"Failed to setup {provider} language model: {e}")
        print_warning("Make sure you have set the appropriate API key in your .env file")
        return None


def configure_dspy(
    lm: Optional[dspy.LM] = None,
    track_usage: bool = False,
    adapter: Optional[Any] = None,
    **kwargs
):
    """Configure DSPy with language model and optional features.

    Args:
        lm: Language model instance. If None, creates one from kwargs.
        track_usage: Enable usage tracking (DSPy 3.x feature) to monitor token usage.
        adapter: Optional adapter (e.g., dspy.ChatAdapter, dspy.JSONAdapter, dspy.XMLAdapter).
        **kwargs: Additional arguments passed to setup_default_lm if lm is None.

    Returns:
        Configured language model instance.
    """
    if lm is None:
        lm = setup_default_lm(**kwargs)

    # Configure DSPy with new 3.x features
    config_params = {"lm": lm}

    if track_usage:
        config_params["track_usage"] = True
        print_result("Usage tracking enabled. Use dspy.get_lm_usage() to view token usage.")

    if adapter:
        config_params["adapter"] = adapter
        print_result(f"Adapter configured: {adapter.__class__.__name__}")

    dspy.configure(**config_params)
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


def get_usage_stats() -> Dict[str, Any]:
    """Get LM usage statistics (DSPy 3.x feature).

    Returns a dictionary with token usage information if tracking is enabled.
    """
    try:
        usage = dspy.get_lm_usage()
        return usage
    except AttributeError:
        print_warning("Usage tracking not available. Enable with configure_dspy(track_usage=True)")
        return {}


def print_usage_stats():
    """Print formatted LM usage statistics."""
    usage = get_usage_stats()
    if usage:
        print_step("LM Usage Statistics")
        for key, value in usage.items():
            print(f"  {key}: {value}")
        print()


# DSPy 3.x adapter helpers
def create_chat_adapter():
    """Create a ChatAdapter for chat-style interactions.

    ChatAdapter formats messages in chat format for better conversational prompts.
    """
    try:
        return dspy.ChatAdapter()
    except AttributeError:
        print_warning("ChatAdapter not available in this DSPy version")
        return None


def create_json_adapter():
    """Create a JSONAdapter for structured JSON outputs.

    JSONAdapter helps ensure model outputs are valid JSON.
    """
    try:
        return dspy.JSONAdapter()
    except AttributeError:
        print_warning("JSONAdapter not available in this DSPy version")
        return None


def create_xml_adapter():
    """Create an XMLAdapter for XML-structured outputs.

    XMLAdapter helps ensure model outputs are valid XML.
    """
    try:
        return dspy.XMLAdapter()
    except AttributeError:
        print_warning("XMLAdapter not available in this DSPy version")
        return None
