#!/usr/bin/env python3
"""
DSPy Predict Module Deep Dive

This script explores dspy.Predict in depth - the most fundamental building block
in DSPy for making language model calls.

What You'll Learn:
- How dspy.Predict works under the hood
- Inspecting completions and model responses
- Using Predict with different signature styles
- Temperature and sampling parameters
- Structured outputs with Predict
- When to use Predict vs other modules
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating dspy.Predict."""
    print("=== DSPy Predict Module Deep Dive ===")
    print("The fundamental building block for LM interactions")
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

    # Example 1: Basic Predict Usage
    print_step(
        "Example 1: Basic Predict",
        "The simplest way to call a language model"
    )

    class SimpleQA(dspy.Signature):
        """Answer the question concisely."""
        question = dspy.InputField()
        answer = dspy.OutputField()

    predictor = dspy.Predict(SimpleQA)
    result = predictor(question="What is the chemical formula for water?")

    print_result(f"Answer: {result.answer}", "Basic Predict")
    print(f"  Return type: {type(result).__name__}")

    # Example 2: Predict with String Signatures
    print_step(
        "Example 2: String Signatures",
        "Using inline string signatures with Predict"
    )

    # Quick one-liner for simple tasks
    translator = dspy.Predict("english_text -> french_text")
    result = translator(english_text="Hello, how are you?")
    print_result(f"French: {result.french_text}", "String Signature")

    # Multiple outputs
    analyzer = dspy.Predict("text -> sentiment, key_topics, word_count")
    result = analyzer(text="Artificial intelligence is revolutionizing healthcare with amazing new diagnostic tools.")
    print_result(
        f"Sentiment: {result.sentiment}\n"
        f"Key Topics: {result.key_topics}\n"
        f"Word Count: {result.word_count}",
        "Multi-Output"
    )

    # Example 3: Inspecting Completions
    print_step(
        "Example 3: Inspecting Completions",
        "Looking at what the model actually returned"
    )

    class DetailedQA(dspy.Signature):
        """Provide a detailed answer to the question."""
        question = dspy.InputField(desc="A question to answer")
        answer = dspy.OutputField(desc="A detailed, informative answer")

    detailed_qa = dspy.Predict(DetailedQA)
    result = detailed_qa(question="Why is the sky blue?")

    print_result(f"Answer: {result.answer}", "Detailed QA")

    # Inspect the last LM call
    history = lm.history
    if history:
        last_call = history[-1]
        print(f"\n  LM call details:")
        print(f"    Messages sent: {len(last_call.get('messages', []))}")
        print(f"    Model: {last_call.get('model', 'N/A')}")

    # Example 4: Structured Output Extraction
    print_step(
        "Example 4: Structured Outputs",
        "Using Predict to extract structured data"
    )

    class PersonExtractor(dspy.Signature):
        """Extract person information from the text."""
        text = dspy.InputField(desc="Text mentioning a person")
        name = dspy.OutputField(desc="The person's full name")
        age = dspy.OutputField(desc="The person's age (number only)")
        occupation = dspy.OutputField(desc="The person's job or role")

    extractor = dspy.Predict(PersonExtractor)

    texts = [
        "Dr. Sarah Chen, 42, is a leading neurosurgeon at Johns Hopkins.",
        "At just 19, Marcus Johnson is already a professional basketball player."
    ]

    for text in texts:
        result = extractor(text=text)
        print_result(
            f"Text: {text}\n  Name: {result.name}\n  Age: {result.age}\n  Occupation: {result.occupation}",
            "Extraction"
        )

    # Example 5: When to Use Predict vs Other Modules
    print_step(
        "Example 5: Predict vs Other Modules",
        "Understanding when Predict is the right choice"
    )

    class MathProblem(dspy.Signature):
        """Solve this math problem."""
        problem = dspy.InputField()
        answer = dspy.OutputField()

    # Predict: direct answer, minimal tokens
    predict_solver = dspy.Predict(MathProblem)
    predict_result = predict_solver(problem="What is 15% of 200?")

    # ChainOfThought: adds reasoning, more tokens but often better quality
    cot_solver = dspy.ChainOfThought(MathProblem)
    cot_result = cot_solver(problem="What is 15% of 200?")

    print(f"  Predict answer: {predict_result.answer}")
    print(f"  ChainOfThought answer: {cot_result.answer}")
    print(f"  ChainOfThought reasoning: {cot_result.reasoning[:100]}...")

    print("\n  When to use Predict:")
    print("  - Simple factual lookups")
    print("  - Text classification")
    print("  - Data extraction")
    print("  - When tokens/cost matter")
    print("  - When the task doesn't need reasoning")
    print("\n  When to use ChainOfThought instead:")
    print("  - Complex reasoning or math")
    print("  - Multi-step problems")
    print("  - When you need to see the model's thinking")

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. dspy.Predict is the simplest way to call an LM with a signature")
    print("2. Works with both string and class-based signatures")
    print("3. Returns a Prediction object with named output fields")
    print("4. Use for simple tasks where reasoning isn't needed")
    print("5. Prefer ChainOfThought for complex problems requiring step-by-step thinking")


if __name__ == "__main__":
    main()
