import os
import dspy
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import chromadb
from pathlib import Path

# Load environment variables
load_dotenv()


def setup_dspy(
    model_name: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    cache_dir: str = ".cache",
    **kwargs
) -> Any:
    """
    Configure DSPy with specified parameters

    Args:
        model_name: Name of the LLM to use
        temperature: Temperature for generation (0.0 to 1.0)
        max_tokens: Maximum tokens for generation
        cache_dir: Directory for caching responses
        **kwargs: Additional model-specific parameters
    """
    # Initialize the language model
    if "openai" in model_name:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found in environment variables")

        lm = dspy.OpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif "anthropic" in model_name:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found in environment variables")

        lm = dspy.Anthropic(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    else:
        # Support for other providers can be added here
        raise NotImplementedError(f"Model {model_name} not supported yet")

    # Configure DSPy settings
    dspy.settings.configure(lm=lm, cache_dir=cache_dir)

    return lm


def setup_retriever(
    collection_name: str = "demo_collection",
    docs_dir: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 3
) -> dspy.Retrieve:
    """
    Set up a retriever for RAG applications with local document storage

    Args:
        collection_name: Name of the document collection
        docs_dir: Directory containing documents to index (optional)
        embedding_model: Model to use for embeddings
        k: Number of passages to retrieve
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=".chromadb")

    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_model
    )

    # Index documents if provided
    if docs_dir:
        docs_path = Path(docs_dir)
        if docs_path.exists():
            for file_path in docs_path.glob("**/*"):
                if file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        collection.add(
                            documents=[text],
                            ids=[str(file_path)]
                        )

    # Create DSPy retriever
    retriever = dspy.ColBERTv2(
        collection=collection,
        k=k
    )

    return retriever


class DSPyConfig:
    """Configuration manager for DSPy applications"""

    def __init__(self):
        self.lm = None
        self.retriever = None
        self._metrics = {}
        self._callbacks = []

    def initialize(
        self,
        model_name: str = "openai/gpt-3.5-turbo",
        retriever_config: Optional[Dict] = None,
        cache_dir: str = ".cache",
        **kwargs
    ):
        """Initialize DSPy with all required components"""
        # Setup language model
        self.lm = setup_dspy(
            model_name=model_name,
            cache_dir=cache_dir,
            **kwargs
        )

        # Setup retriever if config provided
        if retriever_config:
            self.retriever = setup_retriever(**retriever_config)

        # Register default metrics
        self.register_metric("exact_match", lambda x, y: x == y)
        self.register_metric("contains", lambda x, y: y in x)
        self.register_metric("length", lambda x, _: len(str(x)))

    def register_metric(self, name: str, fn: callable):
        """Register a custom evaluation metric"""
        self._metrics[name] = fn

    def get_metric(self, name: str) -> callable:
        """Get a registered metric by name"""
        return self._metrics.get(name)

    def register_callback(self, callback: callable):
        """Register a callback for monitoring"""
        self._callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            'model': self.lm.model if self.lm else None,
            'cache_hits': dspy.settings.get_cache_stats(),
            'total_calls': len(self._callbacks) if self._callbacks else 0
        }


# Create a global config instance
config = DSPyConfig()
