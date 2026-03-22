#!/usr/bin/env python3
"""
DSPy Signatures Deep Dive

This script explores DSPy signatures in depth - the core abstraction that defines
what your language model should do.

What You'll Learn:
- String signatures vs class-based signatures
- Using InputField and OutputField with descriptions
- How docstrings guide model behavior
- Multi-field signatures
- Signature composition patterns
- Best practices for writing effective signatures
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy signatures."""
    print("=== DSPy Signatures Deep Dive ===")
    print("Understanding the core abstraction for LM interactions")
    print("=" * 80)

    # Load environment variables
    load_dotenv('.env')

    # Setup Language Model
    print_step("Setting up Language Model", "Configuring DSPy with OpenAI gpt-4o")

    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=500)
        configure_dspy(lm=lm)
        print_result("Language model configured successfully!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
        return

    # Example 1: String Signatures (Inline)
    print_step(
        "Example 1: String Signatures",
        "The simplest way to define input/output specs"
    )

    # String signatures use "input -> output" format
    simple_qa = dspy.Predict("question -> answer")
    result = simple_qa(question="What is the capital of Japan?")
    print_result(f"Question: What is the capital of Japan?\nAnswer: {result.answer}", "Simple String Signature")

    # Multiple inputs and outputs
    multi_sig = dspy.Predict("context, question -> answer, confidence")
    result = multi_sig(
        context="Python was created by Guido van Rossum in 1991.",
        question="Who created Python?"
    )
    print_result(
        f"Answer: {result.answer}\nConfidence: {result.confidence}",
        "Multi-field String Signature"
    )

    # Example 2: Class-Based Signatures
    print_step(
        "Example 2: Class-Based Signatures",
        "More expressive signatures with descriptions and docstrings"
    )

    class Summarizer(dspy.Signature):
        """Summarize the given text in 2-3 sentences, preserving key information."""
        text = dspy.InputField(desc="The text to summarize")
        summary = dspy.OutputField(desc="A concise 2-3 sentence summary")

    summarizer = dspy.Predict(Summarizer)

    long_text = """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing algorithms that can access data and use it to learn
    for themselves. The process begins with observations or data, such as examples,
    direct experience, or instruction, to look for patterns in data and make
    better decisions in the future based on the examples provided.
    """

    result = summarizer(text=long_text)
    print_result(f"Summary: {result.summary}", "Class-Based Signature")

    # Example 3: Docstrings as Task Instructions
    print_step(
        "Example 3: Docstrings Guide Behavior",
        "How the signature docstring shapes model output"
    )

    # The docstring acts as the task instruction
    class FormalResponder(dspy.Signature):
        """Respond to the message in a formal, professional tone suitable for business communication."""
        message = dspy.InputField(desc="An informal message")
        response = dspy.OutputField(desc="A formal, professional version of the message")

    class CasualResponder(dspy.Signature):
        """Respond to the message in a casual, friendly tone as if texting a close friend."""
        message = dspy.InputField(desc="A formal message")
        response = dspy.OutputField(desc="A casual, friendly version of the message")

    formal = dspy.Predict(FormalResponder)
    casual = dspy.Predict(CasualResponder)

    test_message = "Hey, we need to push the meeting to next week because stuff came up."

    formal_result = formal(message=test_message)
    casual_result = casual(message="We hereby inform you that the quarterly review has been rescheduled.")

    print_result(f"Original: {test_message}\nFormal: {formal_result.response}", "Formal Tone")
    print_result(f"Casual: {casual_result.response}", "Casual Tone")

    # Example 4: Field Descriptions Shape Output
    print_step(
        "Example 4: Field Descriptions",
        "How desc= parameters guide the model's output format"
    )

    class DetailedAnalysis(dspy.Signature):
        """Analyze the given topic."""
        topic = dspy.InputField(desc="A topic to analyze")
        pros = dspy.OutputField(desc="3 advantages or benefits, as a numbered list")
        cons = dspy.OutputField(desc="3 disadvantages or drawbacks, as a numbered list")
        verdict = dspy.OutputField(desc="A one-sentence overall assessment")

    analyzer = dspy.Predict(DetailedAnalysis)
    result = analyzer(topic="Remote work for software developers")

    print_result(
        f"Topic: Remote work for software developers\n\n"
        f"Pros:\n{result.pros}\n\n"
        f"Cons:\n{result.cons}\n\n"
        f"Verdict: {result.verdict}",
        "Multi-Output Analysis"
    )

    # Example 5: Composing Signatures
    print_step(
        "Example 5: Signature Composition",
        "Using multiple signatures together in a pipeline"
    )

    class ExtractKeywords(dspy.Signature):
        """Extract the most important keywords from the text."""
        text = dspy.InputField(desc="Text to extract keywords from")
        keywords = dspy.OutputField(desc="Comma-separated list of 3-5 key terms")

    class GenerateTitle(dspy.Signature):
        """Generate a compelling article title based on the keywords."""
        keywords = dspy.InputField(desc="Key terms to base the title on")
        title = dspy.OutputField(desc="A compelling, concise article title")

    extract = dspy.Predict(ExtractKeywords)
    generate = dspy.Predict(GenerateTitle)

    article_text = """
    Recent advances in quantum computing have enabled researchers to solve
    optimization problems that were previously intractable. Google's latest
    quantum processor achieved results in minutes that would take classical
    computers thousands of years.
    """

    keywords_result = extract(text=article_text)
    title_result = generate(keywords=keywords_result.keywords)

    print_result(
        f"Keywords: {keywords_result.keywords}\n"
        f"Generated Title: {title_result.title}",
        "Signature Pipeline"
    )

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. String signatures ('input -> output') are quick for simple tasks")
    print("2. Class signatures give you docstrings and field descriptions")
    print("3. The docstring acts as the task instruction for the model")
    print("4. Field descriptions (desc=) guide the format and content of outputs")
    print("5. Compose multiple signatures to build multi-step pipelines")


if __name__ == "__main__":
    main()
