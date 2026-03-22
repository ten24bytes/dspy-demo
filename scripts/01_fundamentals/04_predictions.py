#!/usr/bin/env python3
"""
DSPy Predictions and Examples

This script explores how to work with DSPy's data and prediction objects:
- dspy.Prediction for accessing model outputs
- dspy.Example for training/evaluation data
- Metrics and evaluation with dspy.Evaluate

What You'll Learn:
- How to create and inspect Prediction objects
- How to create Examples with with_inputs()
- How to write metric functions
- How to evaluate modules with dspy.Evaluate
- How to work with datasets from utils.datasets
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
from utils.datasets import get_sample_qa_data, get_sample_classification_data
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy predictions and examples."""
    print("=== DSPy Predictions and Examples ===")
    print("Working with data, predictions, and evaluation")
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

    # Example 1: Working with Predictions
    print_step(
        "Example 1: Prediction Objects",
        "Understanding dspy.Prediction"
    )

    class QA(dspy.Signature):
        """Answer the question concisely."""
        question = dspy.InputField()
        answer = dspy.OutputField()

    qa = dspy.Predict(QA)
    prediction = qa(question="What is the largest ocean on Earth?")

    # Accessing fields
    print(f"  prediction.answer = {prediction.answer}")
    print(f"  type(prediction) = {type(prediction).__name__}")

    # Creating predictions manually
    manual_pred = dspy.Prediction(answer="The Pacific Ocean", confidence="high")
    print(f"\n  Manual prediction: {manual_pred.answer}, confidence: {manual_pred.confidence}")

    # Example 2: Working with Examples
    print_step(
        "Example 2: Example Objects",
        "Creating training and evaluation data"
    )

    # Create examples manually
    examples = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
        dspy.Example(question="Largest planet?", answer="Jupiter").with_inputs("question"),
    ]

    print("  Created examples with with_inputs():")
    for i, ex in enumerate(examples, 1):
        print(f"    {i}. Q: {ex.question} -> A: {ex.answer}")

    # with_inputs() marks which fields are inputs vs labels
    ex = examples[0]
    print(f"\n  ex.inputs() = {ex.inputs()}")
    print(f"  ex.labels() = {ex.labels()}")

    # Example 3: Loading Datasets
    print_step(
        "Example 3: Loading Datasets",
        "Using the utils.datasets module"
    )

    qa_data = get_sample_qa_data()
    print(f"  QA dataset: {len(qa_data)} examples")
    for ex in qa_data[:3]:
        print(f"    Q: {ex.question[:50]}... -> A: {ex.answer[:30]}...")

    class_data = get_sample_classification_data()
    print(f"\n  Classification dataset: {len(class_data)} examples")
    for ex in class_data[:3]:
        print(f"    Text: {ex.text[:40]}... -> Label: {ex.label}")

    # Example 4: Writing Metric Functions
    print_step(
        "Example 4: Metric Functions",
        "How to evaluate predictions against ground truth"
    )

    def exact_match(example, prediction, trace=None):
        """Strict exact match metric."""
        return prediction.answer.strip().lower() == example.answer.strip().lower()

    def fuzzy_match(example, prediction, trace=None):
        """Fuzzy match: check if key words from the answer appear in prediction."""
        pred_words = set(prediction.answer.lower().split())
        true_words = set(example.answer.lower().split())
        overlap = len(pred_words & true_words)
        if not true_words:
            return 0.0
        return overlap / len(true_words)

    # Test metrics
    test_example = dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question")
    test_pred = qa(question=test_example.question)

    exact = exact_match(test_example, test_pred)
    fuzzy = fuzzy_match(test_example, test_pred)

    print(f"  Question: {test_example.question}")
    print(f"  Expected: {test_example.answer}")
    print(f"  Predicted: {test_pred.answer}")
    print(f"  Exact match: {exact}")
    print(f"  Fuzzy match: {fuzzy:.2f}")

    # Example 5: Evaluating Modules
    print_step(
        "Example 5: Module Evaluation",
        "Using dspy.Evaluate to assess performance"
    )

    # Create a small evaluation set
    eval_set = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(question="Capital of Japan?", answer="Tokyo").with_inputs("question"),
        dspy.Example(question="Largest planet?", answer="Jupiter").with_inputs("question"),
    ]

    # Manual evaluation loop
    print("\n  Manual evaluation:")
    scores = []
    for ex in eval_set:
        pred = qa(question=ex.question)
        score = fuzzy_match(ex, pred)
        scores.append(score)
        print(f"    Q: {ex.question} | Expected: {ex.answer} | Got: {pred.answer} | Score: {score:.2f}")

    avg_score = sum(scores) / len(scores)
    print(f"\n  Average score: {avg_score:.2f}")

    # Using dspy.Evaluate
    print("\n  Using dspy.Evaluate:")
    try:
        evaluator = dspy.Evaluate(
            devset=eval_set,
            metric=fuzzy_match,
            num_threads=1,
            display_progress=True
        )
        overall_score = evaluator(qa)
        print(f"  Overall evaluation score: {overall_score}")
    except Exception as e:
        print(f"  dspy.Evaluate encountered an error: {e}")
        print("  (This is expected if the API format differs; manual evaluation above works fine)")

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. Predictions hold model outputs - access fields with dot notation")
    print("2. Examples represent labeled data - use with_inputs() to mark input fields")
    print("3. ex.inputs() returns input fields, ex.labels() returns label fields")
    print("4. Metrics take (example, prediction, trace) and return a score")
    print("5. dspy.Evaluate runs systematic evaluation across a dataset")


if __name__ == "__main__":
    main()
