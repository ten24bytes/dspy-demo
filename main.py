import os
from dotenv import load_dotenv
from src.config import DSPyConfig
from src.basic_qa import BasicQA, RAGBasedQA
from src.advanced_rag import MultiHopRetriever, SelfCorrectingRAG, ProgressiveRAG
from src.optimization_patterns import SelfImprovingModule, OptimizedChainOfThought, AdaptivePrompting


def setup_environment():
    """Initialize the environment and configurations"""
    load_dotenv()
    config = DSPyConfig()
    config.initialize()
    return config


def demonstrate_basic_qa():
    """Demonstrate basic question answering"""
    qa = BasicQA()
    questions = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Explain quantum computing in simple terms"
    ]
    for question in questions:
        print(f"\nQ: {question}")
        print(f"A: {qa(question)}")


def demonstrate_rag():
    """Demonstrate different RAG approaches"""
    # Multi-hop RAG
    multihop = MultiHopRetriever()
    question = "What are the environmental and economic impacts of renewable energy?"
    result = multihop(question)
    print(f"\nMulti-hop RAG Result:")
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"Number of contexts used: {len(result['contexts'])}")

    # Self-correcting RAG
    correcting = SelfCorrectingRAG()
    result = correcting("Explain the theory of relativity")
    print(f"\nSelf-correcting RAG Result:")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Attempts: {result['attempts']}")


def demonstrate_optimization():
    """Demonstrate optimization and self-improvement patterns"""
    # Adaptive prompting
    adaptive = AdaptivePrompting()
    simple_input = "What is 2+2?"
    complex_input = "Analyze the implications of artificial intelligence on future job markets"

    simple_result = adaptive(simple_input)
    complex_result = adaptive(complex_input)

    print(f"\nAdaptive Prompting Results:")
    print(f"Simple input ({simple_result['method']}):")
    print(simple_result['output'])
    print(f"\nComplex input ({complex_result['method']}):")
    print(complex_result['output'])


def main():
    """Main demonstration script"""
    print("DSPy Demo Project - Examples and Patterns")
    print("========================================")

    # Setup
    config = setup_environment()

    # Run demonstrations
    print("\n1. Basic Question Answering")
    print("--------------------------")
    demonstrate_basic_qa()

    print("\n2. RAG Patterns")
    print("-------------")
    demonstrate_rag()

    print("\n3. Optimization Patterns")
    print("---------------------")
    demonstrate_optimization()


if __name__ == "__main__":
    main()
