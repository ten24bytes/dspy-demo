#!/usr/bin/env python3
"""
Advanced Multi-Hop RAG with DSPy

This script demonstrates advanced multi-hop Retrieval-Augmented Generation (RAG) techniques using DSPy.
We'll cover:
1. Multi-hop reasoning patterns
2. Dynamic query decomposition  
3. Iterative retrieval strategies
4. Answer synthesis from multiple sources
5. Performance optimization

Multi-hop RAG is essential for complex questions that require information from multiple sources
or reasoning across multiple steps.
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import dspy
import os
import sys
sys.path.append('../../')


# Load environment variables
load_dotenv('../../.env')

# Data Structures for Multi-Hop RAG


@dataclass
class Document:
    """Represents a document in our knowledge base."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Represents a retrieval result with relevance scoring."""
    document: Document
    score: float
    query: str
    hop_number: int


@dataclass
class ReasoningStep:
    """Represents a step in multi-hop reasoning."""
    query: str
    retrieved_docs: List[RetrievalResult]
    intermediate_answer: str
    confidence: float
    next_query: Optional[str] = None

# Query Decomposition Module


class QueryDecomposition(dspy.Signature):
    """Decompose a complex question into simpler sub-questions for multi-hop reasoning."""

    complex_question = dspy.InputField(desc="The original complex question")
    sub_questions = dspy.OutputField(desc="List of simpler sub-questions, one per line")
    reasoning_strategy = dspy.OutputField(desc="Brief explanation of the decomposition strategy")


class QueryDecomposer(dspy.Module):
    """Module for decomposing complex queries into sub-questions."""

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(QueryDecomposition)

    def forward(self, question: str) -> List[str]:
        result = self.decompose(complex_question=question)

        # Parse sub-questions from the output
        sub_questions = [
            q.strip() for q in result.sub_questions.split('\n')
            if q.strip() and not q.strip().startswith('-') and '?' in q
        ]

        return sub_questions

# Enhanced Retrieval Module


class EnhancedRetriever(dspy.Module):
    """Enhanced retriever with multi-hop capabilities."""

    def __init__(self, documents: List[Document], k: int = 3):
        super().__init__()
        self.documents = documents
        self.k = k

        # Simple TF-IDF based retrieval (in practice, use vector embeddings)
        self._build_index()

    def _build_index(self):
        """Build a simple keyword-based index."""
        self.doc_keywords = {}
        for doc in self.documents:
            # Simple keyword extraction
            text = f"{doc.title} {doc.content}".lower()
            keywords = set(word.strip('.,!?;:"()[]{}') for word in text.split())
            self.doc_keywords[doc.id] = keywords

    def retrieve(self, query: str, hop_number: int = 1,
                 previous_results: List[RetrievalResult] = None) -> List[RetrievalResult]:
        """Retrieve documents for a given query."""
        query_keywords = set(query.lower().split())

        # Calculate relevance scores
        scored_docs = []
        for doc in self.documents:
            doc_keywords = self.doc_keywords[doc.id]

            # Basic keyword overlap scoring
            overlap = len(query_keywords & doc_keywords)
            score = overlap / len(query_keywords) if query_keywords else 0

            # Boost score if document relates to previous results
            if previous_results and hop_number > 1:
                for prev_result in previous_results:
                    prev_keywords = self.doc_keywords[prev_result.document.id]
                    connection_score = len(doc_keywords & prev_keywords) / len(doc_keywords)
                    score += connection_score * 0.3

            if score > 0:
                scored_docs.append(RetrievalResult(
                    document=doc,
                    score=score,
                    query=query,
                    hop_number=hop_number
                ))

        # Sort by score and return top-k
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[:self.k]

# Multi-Hop Reasoning Module


class IntermediateReasoning(dspy.Signature):
    """Generate intermediate reasoning based on retrieved documents."""

    question = dspy.InputField(desc="The sub-question being answered")
    context = dspy.InputField(desc="Retrieved documents context")
    previous_reasoning = dspy.InputField(desc="Previous reasoning steps")

    intermediate_answer = dspy.OutputField(desc="Intermediate answer to the sub-question")
    confidence = dspy.OutputField(desc="Confidence score (0-1) in this answer")
    next_question = dspy.OutputField(desc="Next sub-question to explore, or 'COMPLETE' if done")


class MultiHopReasoner(dspy.Module):
    """Multi-hop reasoning module that iteratively retrieves and reasons."""

    def __init__(self, retriever: EnhancedRetriever, max_hops: int = 3):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.reasoning = dspy.ChainOfThought(IntermediateReasoning)

    def forward(self, question: str) -> List[ReasoningStep]:
        """Perform multi-hop reasoning for a question."""
        reasoning_steps = []
        current_question = question
        previous_results = []

        for hop in range(1, self.max_hops + 1):
            # Retrieve documents for current question
            retrieved_docs = self.retriever.retrieve(
                current_question, hop, previous_results
            )

            if not retrieved_docs:
                break

            # Format context from retrieved documents
            context = "\n\n".join([
                f"Document {i + 1}: {doc.document.title}\n{doc.document.content}"
                for i, doc in enumerate(retrieved_docs)
            ])

            # Format previous reasoning
            previous_reasoning = "\n".join([
                f"Step {i + 1}: {step.query} -> {step.intermediate_answer}"
                for i, step in enumerate(reasoning_steps)
            ])

            # Generate intermediate reasoning
            result = self.reasoning(
                question=current_question,
                context=context,
                previous_reasoning=previous_reasoning or "No previous steps"
            )

            # Parse confidence score
            try:
                confidence = float(result.confidence)
            except (ValueError, TypeError):
                confidence = 0.5

            # Create reasoning step
            step = ReasoningStep(
                query=current_question,
                retrieved_docs=retrieved_docs,
                intermediate_answer=result.intermediate_answer,
                confidence=confidence,
                next_query=result.next_question if result.next_question != "COMPLETE" else None
            )

            reasoning_steps.append(step)
            previous_results.extend(retrieved_docs)

            # Check if reasoning is complete
            if result.next_question == "COMPLETE" or not result.next_question:
                break

            current_question = result.next_question

        return reasoning_steps

# Answer Synthesis Module


class AnswerSynthesis(dspy.Signature):
    """Synthesize a final answer from multi-hop reasoning steps."""

    original_question = dspy.InputField(desc="The original complex question")
    reasoning_steps = dspy.InputField(desc="All reasoning steps with intermediate answers")
    confidence_scores = dspy.InputField(desc="Confidence scores for each step")

    final_answer = dspy.OutputField(desc="Comprehensive final answer")
    supporting_evidence = dspy.OutputField(desc="Key supporting evidence from the reasoning")
    overall_confidence = dspy.OutputField(desc="Overall confidence in the final answer (0-1)")


class AnswerSynthesizer(dspy.Module):
    """Module for synthesizing final answers from multi-hop reasoning."""

    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(AnswerSynthesis)

    def forward(self, original_question: str, reasoning_steps: List[ReasoningStep]) -> Dict[str, Any]:
        """Synthesize final answer from reasoning steps."""
        # Format reasoning steps
        steps_text = "\n\n".join([
            f"Step {i + 1}:\n"
            f"Question: {step.query}\n"
            f"Answer: {step.intermediate_answer}\n"
            f"Confidence: {step.confidence:.2f}"
            for i, step in enumerate(reasoning_steps)
        ])

        # Format confidence scores
        confidence_text = ", ".join([f"{step.confidence:.2f}" for step in reasoning_steps])

        # Synthesize answer
        result = self.synthesize(
            original_question=original_question,
            reasoning_steps=steps_text,
            confidence_scores=confidence_text
        )

        # Parse overall confidence
        try:
            overall_confidence = float(result.overall_confidence)
        except (ValueError, TypeError):
            # Calculate average confidence if parsing fails
            overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)

        return {
            "final_answer": result.final_answer,
            "supporting_evidence": result.supporting_evidence,
            "overall_confidence": overall_confidence,
            "reasoning_steps": reasoning_steps
        }

# Complete Multi-Hop RAG System


class MultiHopRAG(dspy.Module):
    """Complete multi-hop RAG system."""

    def __init__(self, documents: List[Document], max_hops: int = 3):
        super().__init__()
        self.retriever = EnhancedRetriever(documents)
        self.decomposer = QueryDecomposer()
        self.reasoner = MultiHopReasoner(self.retriever, max_hops)
        self.synthesizer = AnswerSynthesizer()

    def forward(self, question: str, use_decomposition: bool = True) -> Dict[str, Any]:
        """Process a question through multi-hop RAG."""
        results = {
            "original_question": question,
            "decomposed_questions": [],
            "reasoning_chains": [],
            "final_answer": "",
            "supporting_evidence": "",
            "overall_confidence": 0.0
        }

        if use_decomposition:
            # Decompose complex question
            sub_questions = self.decomposer(question)
            results["decomposed_questions"] = sub_questions

            # Process each sub-question
            all_reasoning_steps = []
            for sub_q in sub_questions:
                reasoning_steps = self.reasoner(sub_q)
                results["reasoning_chains"].append({
                    "sub_question": sub_q,
                    "steps": reasoning_steps
                })
                all_reasoning_steps.extend(reasoning_steps)
        else:
            # Process question directly
            all_reasoning_steps = self.reasoner(question)
            results["reasoning_chains"] = [{
                "sub_question": question,
                "steps": all_reasoning_steps
            }]

        # Synthesize final answer
        if all_reasoning_steps:
            synthesis_result = self.synthesizer(question, all_reasoning_steps)
            results.update(synthesis_result)

        return results

# Performance Analysis and Optimization


def analyze_performance(rag_system: MultiHopRAG, test_questions: List[str]) -> Dict[str, Any]:
    """Analyze performance of the multi-hop RAG system."""
    results = {
        "total_questions": len(test_questions),
        "processing_times": [],
        "average_hops": [],
        "confidence_scores": [],
        "decomposition_effectiveness": []
    }

    for question in test_questions:
        start_time = time.time()

        # Process with decomposition
        result_with_decomp = rag_system(question, use_decomposition=True)

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate metrics
        total_hops = sum(len(chain['steps']) for chain in result_with_decomp['reasoning_chains'])
        avg_hops = total_hops / len(result_with_decomp['reasoning_chains']) if result_with_decomp['reasoning_chains'] else 0

        # Store results
        results["processing_times"].append(processing_time)
        results["average_hops"].append(avg_hops)
        results["confidence_scores"].append(result_with_decomp['overall_confidence'])
        results["decomposition_effectiveness"].append(len(result_with_decomp['decomposed_questions']))

    # Calculate summary statistics
    results["avg_processing_time"] = sum(results["processing_times"]) / len(results["processing_times"])
    results["avg_confidence"] = sum(results["confidence_scores"]) / len(results["confidence_scores"])
    results["avg_decomposition_size"] = sum(results["decomposition_effectiveness"]) / len(results["decomposition_effectiveness"])

    return results

# Optimized Multi-Hop RAG with Caching


class OptimizedMultiHopRAG(MultiHopRAG):
    """Optimized version of multi-hop RAG with caching and parallel processing."""

    def __init__(self, documents: List[Document], max_hops: int = 3, enable_cache: bool = True):
        super().__init__(documents, max_hops)
        self.enable_cache = enable_cache
        self.retrieval_cache = {} if enable_cache else None
        self.reasoning_cache = {} if enable_cache else None

    def _get_cache_key(self, query: str, hop: int = 1) -> str:
        """Generate cache key for query and hop combination."""
        return f"{query.lower().strip()}_{hop}"

    def cached_retrieve(self, query: str, hop_number: int = 1,
                        previous_results: List[RetrievalResult] = None) -> List[RetrievalResult]:
        """Cached version of retrieval."""
        if not self.enable_cache:
            return self.retriever.retrieve(query, hop_number, previous_results)

        cache_key = self._get_cache_key(query, hop_number)

        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]

        results = self.retriever.retrieve(query, hop_number, previous_results)
        self.retrieval_cache[cache_key] = results
        return results

    def forward(self, question: str, use_decomposition: bool = True) -> Dict[str, Any]:
        """Optimized forward pass with caching."""
        # Check reasoning cache
        if self.enable_cache and question in self.reasoning_cache:
            return self.reasoning_cache[question]

        # Process normally
        result = super().forward(question, use_decomposition)

        # Cache result
        if self.enable_cache:
            self.reasoning_cache[question] = result

        return result

    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics."""
        if not self.enable_cache:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            "retrieval_cache_size": len(self.retrieval_cache),
            "reasoning_cache_size": len(self.reasoning_cache)
        }


def main():
    """Main function to demonstrate multi-hop RAG."""

    print_step("Multi-Hop RAG with DSPy", "Setting up language model and system")

    try:
        # Configure DSPy with OpenAI
        lm = setup_default_lm(provider="openai", model="gpt-4o")
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return

    # Create sample documents
    sample_documents = [
        Document(
            id="doc1",
            title="Machine Learning Fundamentals",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning. Common algorithms include linear regression, decision trees, and neural networks.",
            metadata={"category": "ML", "year": 2023}
        ),
        Document(
            id="doc2",
            title="Neural Networks and Deep Learning",
            content="Neural networks are computing systems inspired by biological neural networks. Deep learning uses multi-layer neural networks to model complex patterns in data. Popular architectures include CNNs for image processing and RNNs for sequence data.",
            metadata={"category": "DL", "year": 2023}
        ),
        Document(
            id="doc3",
            title="Natural Language Processing",
            content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through language. It includes tasks like text classification, named entity recognition, and machine translation. Modern NLP uses transformer models like BERT and GPT.",
            metadata={"category": "NLP", "year": 2023}
        ),
        Document(
            id="doc4",
            title="Transformer Architecture",
            content="The Transformer architecture revolutionized NLP with its attention mechanism. It consists of encoder and decoder layers with multi-head attention. This architecture enabled the development of large language models like GPT and BERT, which achieve state-of-the-art performance on many NLP tasks.",
            metadata={"category": "Architecture", "year": 2023}
        ),
        Document(
            id="doc5",
            title="Large Language Models",
            content="Large Language Models (LLMs) are neural networks trained on vast amounts of text data. They can generate human-like text, answer questions, and perform various language tasks. Examples include GPT-3, GPT-4, and Claude. These models use transformer architecture and are trained using self-supervised learning.",
            metadata={"category": "LLM", "year": 2023}
        )
    ]

    # Initialize the multi-hop RAG system
    print_step("System Initialization", f"Creating multi-hop RAG system with {len(sample_documents)} documents")
    multi_hop_rag = MultiHopRAG(sample_documents, max_hops=3)
    print_result("Multi-hop RAG system initialized successfully!")

    # Test with a complex question requiring multi-hop reasoning
    complex_question = "How do transformer architectures enable large language models to perform natural language processing tasks?"

    print_step("Processing Complex Question", f"Question: {complex_question}")

    # Process the question
    result = multi_hop_rag(complex_question, use_decomposition=True)

    # Display results
    print("\n" + "=" * 60)
    print("MULTI-HOP RAG RESULTS")
    print("=" * 60)

    print(f"\nOriginal Question: {result['original_question']}")

    if result['decomposed_questions']:
        print(f"\nDecomposed Sub-questions:")
        for i, sub_q in enumerate(result['decomposed_questions'], 1):
            print(f"  {i}. {sub_q}")

    print(f"\nReasoning Chains:")
    for i, chain in enumerate(result['reasoning_chains'], 1):
        print(f"\n  Chain {i} - {chain['sub_question']}:")
        for j, step in enumerate(chain['steps'], 1):
            print(f"    Step {j}: {step.query}")
            print(f"    Answer: {step.intermediate_answer}")
            print(f"    Confidence: {step.confidence:.2f}")
            if step.next_query:
                print(f"    Next Query: {step.next_query}")
            print()

    print(f"Final Answer: {result['final_answer']}")
    print(f"\nSupporting Evidence: {result['supporting_evidence']}")
    print(f"\nOverall Confidence: {result['overall_confidence']:.2f}")

    # Performance analysis
    print_step("Performance Analysis", "Running performance tests")

    test_questions = [
        "What is machine learning?",
        "How do neural networks work in deep learning?",
        "What makes transformer architectures effective for NLP?",
        "How are large language models trained and what can they do?"
    ]

    perf_results = analyze_performance(multi_hop_rag, test_questions)

    print("\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Total Questions Processed: {perf_results['total_questions']}")
    print(f"Average Processing Time: {perf_results['avg_processing_time']:.2f} seconds")
    print(f"Average Confidence Score: {perf_results['avg_confidence']:.2f}")
    print(f"Average Decomposition Size: {perf_results['avg_decomposition_size']:.1f} sub-questions")
    print(f"Average Reasoning Hops: {sum(perf_results['average_hops']) / len(perf_results['average_hops']):.1f}")

    # Test optimized version
    print_step("Optimized System", "Testing optimized multi-hop RAG with caching")

    optimized_rag = OptimizedMultiHopRAG(sample_documents, max_hops=3, enable_cache=True)
    print_result("Optimized system created")
    print("Cache stats:", optimized_rag.get_cache_stats())

    print_result("Multi-hop RAG demonstration completed successfully!")


if __name__ == "__main__":
    main()
