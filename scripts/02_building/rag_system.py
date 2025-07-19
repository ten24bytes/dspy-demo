#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) with DSPy - Python Script Version

This script demonstrates building RAG systems using DSPy:
- Document retrieval with TF-IDF
- Basic and advanced RAG implementations
- Multi-query RAG for improved coverage
- RAG system evaluation
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from utils.datasets import get_sample_rag_documents
from utils import setup_default_lm, print_step, print_result, print_error
from typing import List
import numpy as np
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class SimpleRetriever:
    """Simple TF-IDF based document retriever."""

    def __init__(self, documents: List[str]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.document_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top-k most relevant documents for the query."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        # Get top-k document indices
        top_indices = np.argsort(similarities)[::-1][:k]

        return [self.documents[i] for i in top_indices]


class BasicRAG(dspy.Module):
    """Basic RAG module that retrieves documents and generates answers."""

    def __init__(self, retriever, k=3):
        super().__init__()
        self.retriever = retriever
        self.k = k

        class GenerateAnswer(dspy.Signature):
            """Answer a question using the provided context documents."""
            context = dspy.InputField(desc="Relevant documents or passages")
            question = dspy.InputField(desc="The question to answer")
            answer = dspy.OutputField(desc="A comprehensive answer based on the context")

        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # Retrieve relevant documents
        context_docs = self.retriever.retrieve(question, k=self.k)

        # Combine contexts
        context = "\\n\\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(context_docs)])

        # Generate answer
        result = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            context=context,
            reasoning=result.reasoning,
            answer=result.answer
        )


class AdvancedRAG(dspy.Module):
    """Advanced RAG module with citation support."""

    def __init__(self, retriever, k=3):
        super().__init__()
        self.retriever = retriever
        self.k = k

        class GenerateAnswerWithCitation(dspy.Signature):
            """Answer a question using provided context and include citations."""
            context = dspy.InputField(desc="Relevant documents or passages")
            question = dspy.InputField(desc="The question to answer")
            answer = dspy.OutputField(desc="A comprehensive answer based on the context")
            citations = dspy.OutputField(desc="Citations or references to specific parts of the context")

        self.generate_answer = dspy.ChainOfThought(GenerateAnswerWithCitation)

    def forward(self, question):
        # Retrieve relevant documents
        context_docs = self.retriever.retrieve(question, k=self.k)

        # Create numbered context with clear document boundaries
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[Document {i}]: {doc}")

        context = "\\n\\n".join(context_parts)

        # Generate answer with citations
        result = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            context=context,
            reasoning=result.reasoning,
            answer=result.answer,
            citations=result.citations,
            retrieved_docs=context_docs
        )


class MultiQueryRAG(dspy.Module):
    """RAG module that generates multiple queries for better retrieval coverage."""

    def __init__(self, retriever, k=2):
        super().__init__()
        self.retriever = retriever
        self.k = k

        class QueryExpansion(dspy.Signature):
            """Generate multiple related queries to improve document retrieval."""
            original_query = dspy.InputField(desc="The original question")
            expanded_queries = dspy.OutputField(desc="3-5 related queries that could help find relevant information")

        class GenerateAnswer(dspy.Signature):
            """Answer a question using the provided context documents."""
            context = dspy.InputField(desc="Relevant documents or passages")
            question = dspy.InputField(desc="The question to answer")
            answer = dspy.OutputField(desc="A comprehensive answer based on the context")

        self.expand_query = dspy.Predict(QueryExpansion)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # Expand the original query
        expansion_result = self.expand_query(original_query=question)

        # Parse expanded queries (simple split by newline)
        expanded_queries = [q.strip() for q in expansion_result.expanded_queries.split('\\n') if q.strip()]
        all_queries = [question] + expanded_queries[:3]  # Limit to avoid too many queries

        # Retrieve documents for each query
        all_docs = []
        for query in all_queries:
            docs = self.retriever.retrieve(query, k=self.k)
            all_docs.extend(docs)

        # Remove duplicates while preserving order
        unique_docs = []
        seen = set()
        for doc in all_docs:
            if doc not in seen:
                unique_docs.append(doc)
                seen.add(doc)

        # Limit to top documents
        final_docs = unique_docs[:4]  # Limit to 4 documents

        # Create context
        context = "\\n\\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(final_docs)])

        # Generate answer
        result = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            original_question=question,
            expanded_queries=expanded_queries,
            context=context,
            reasoning=result.reasoning,
            answer=result.answer,
            retrieved_docs=final_docs
        )


def evaluate_rag_systems(basic_rag, advanced_rag, multi_query_rag, test_questions):
    """Evaluate different RAG systems."""
    print_step("RAG System Evaluation", "Comparing different RAG approaches")

    class AnswerQuality(dspy.Signature):
        """Evaluate the quality of an answer given a question and context."""
        question = dspy.InputField(desc="The original question")
        context = dspy.InputField(desc="The context used to generate the answer")
        answer = dspy.InputField(desc="The generated answer")
        quality_score = dspy.OutputField(desc="Quality score from 1-10 with explanation")

    evaluator = dspy.Predict(AnswerQuality)

    for question in test_questions:
        print(f"\\nQuestion: {question}")
        print("-" * 50)

        # Test basic RAG
        basic_result = basic_rag(question=question)
        basic_eval = evaluator(
            question=question,
            context=basic_result.context[:500] + "...",
            answer=basic_result.answer
        )

        # Test advanced RAG
        advanced_result = advanced_rag(question=question)
        advanced_eval = evaluator(
            question=question,
            context=advanced_result.context[:500] + "...",
            answer=advanced_result.answer
        )

        print(f"Basic RAG Score: {basic_eval.quality_score}")
        print(f"Advanced RAG Score: {advanced_eval.quality_score}")


def demo_rag_systems(basic_rag, advanced_rag, multi_query_rag, question: str):
    """Demonstrate all RAG systems with a given question."""
    print(f"\\nQuestion: {question}")
    print("=" * 60)

    # Basic RAG
    print("\\n🔍 Basic RAG:")
    basic_result = basic_rag(question=question)
    print(f"Answer: {basic_result.answer}")

    # Advanced RAG with Citations
    print("\\n🎯 Advanced RAG with Citations:")
    advanced_result = advanced_rag(question=question)
    print(f"Answer: {advanced_result.answer}")
    if hasattr(advanced_result, 'citations') and advanced_result.citations:
        print(f"Citations: {advanced_result.citations}")

    # Multi-Query RAG
    print("\\n🚀 Multi-Query RAG:")
    multi_result = multi_query_rag(question=question)
    print(f"Answer: {multi_result.answer}")
    print(f"Expanded Queries Used: {str(multi_result.expanded_queries)[:100]}...")

    print("\\n" + "=" * 60)


def main():
    """Main function demonstrating RAG systems with DSPy."""
    print("🔍 RAG (Retrieval-Augmented Generation) with DSPy")
    print("=" * 60)

    # Load environment variables
    load_dotenv('.env')

    # Configure Language Model
    print_step("Configuring Language Model", "Setting up DSPy with OpenAI")

    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=1000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return

    # Setup Retriever
    print_step("Setting up Document Retriever", "Creating TF-IDF based retriever")

    documents = get_sample_rag_documents()
    retriever = SimpleRetriever(documents)

    print_result(f"Retriever initialized with {len(documents)} documents")

    # Test retriever
    test_query = "What is machine learning?"
    retrieved_docs = retriever.retrieve(test_query, k=2)
    print(f"\\nTest Query: {test_query}")
    print("Retrieved documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc[:100]}...")

    # Create RAG Systems
    print_step("Creating RAG Systems", "Building different RAG approaches")

    basic_rag = BasicRAG(retriever, k=2)
    advanced_rag = AdvancedRAG(retriever, k=2)
    multi_query_rag = MultiQueryRAG(retriever, k=2)

    print_result("All RAG systems created successfully!")

    # Test Questions
    test_questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What programming language is mentioned in the documents?",
        "What is the relationship between machine learning and deep learning?",
        "How can AI help with data analysis?"
    ]

    # Demo each system
    print_step("RAG System Demonstrations", "Testing different approaches")

    for question in test_questions[:2]:  # Test first 2 questions
        demo_rag_systems(basic_rag, advanced_rag, multi_query_rag, question)

    # Evaluate systems
    evaluate_rag_systems(basic_rag, advanced_rag, multi_query_rag, test_questions[:2])

    print("\\n🎉 RAG demonstration completed!")
    print("\\nKey Takeaways:")
    print("- Basic RAG combines retrieval and generation")
    print("- Advanced RAG adds citations for transparency")
    print("- Multi-Query RAG uses query expansion for better coverage")
    print("- Each approach has different strengths for different use cases")


if __name__ == "__main__":
    main()
