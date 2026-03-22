#!/usr/bin/env python3
"""
DSPy ChainOfThought Module Deep Dive

This script explores dspy.ChainOfThought - the module that adds step-by-step
reasoning to language model calls.

What You'll Learn:
- How ChainOfThought adds reasoning to predictions
- Inspecting the rationale field
- When ChainOfThought helps vs hurts performance
- ChainOfThought with different signature types
- Combining ChainOfThought with few-shot examples
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating dspy.ChainOfThought."""
    print("=== DSPy ChainOfThought Deep Dive ===")
    print("Adding step-by-step reasoning to LM calls")
    print("=" * 80)

    # Load environment variables
    load_dotenv('.env')

    # Setup Language Model
    print_step("Setting up Language Model", "Configuring DSPy with OpenAI gpt-4o")

    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=1000)
        configure_dspy(lm=lm)
        print_result("Language model configured successfully!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
        return

    # Example 1: Basic ChainOfThought
    print_step(
        "Example 1: Basic ChainOfThought",
        "How ChainOfThought adds a reasoning field"
    )

    class MathSolver(dspy.Signature):
        """Solve the math problem."""
        problem = dspy.InputField()
        answer = dspy.OutputField()

    cot = dspy.ChainOfThought(MathSolver)
    result = cot(problem="A store has 45 apples. If 3/5 are sold, how many remain?")

    print_result(
        f"Problem: A store has 45 apples. If 3/5 are sold, how many remain?\n\n"
        f"Reasoning: {result.reasoning}\n\n"
        f"Answer: {result.answer}",
        "Math with CoT"
    )

    # Example 2: Inspecting the Rationale
    print_step(
        "Example 2: The Reasoning Field",
        "ChainOfThought automatically adds a 'reasoning' output field"
    )

    class FactChecker(dspy.Signature):
        """Determine if the statement is true or false."""
        statement = dspy.InputField(desc="A factual claim to verify")
        verdict = dspy.OutputField(desc="Either 'true' or 'false'")

    checker = dspy.ChainOfThought(FactChecker)

    statements = [
        "The Great Wall of China is visible from space with the naked eye.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Lightning never strikes the same place twice."
    ]

    for stmt in statements:
        result = checker(statement=stmt)
        print(f"\n  Statement: {stmt}")
        print(f"  Reasoning: {result.reasoning[:150]}...")
        print(f"  Verdict: {result.verdict}")

    # Example 3: When CoT Helps
    print_step(
        "Example 3: When CoT Helps",
        "Comparing Predict vs ChainOfThought on a tricky problem"
    )

    class LogicProblem(dspy.Signature):
        """Solve this logic problem carefully."""
        problem = dspy.InputField()
        answer = dspy.OutputField()

    tricky_problem = "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"

    # Without reasoning
    simple = dspy.Predict(LogicProblem)
    simple_result = simple(problem=tricky_problem)
    print(f"  Predict answer: {simple_result.answer}")

    # With reasoning
    reasoned = dspy.ChainOfThought(LogicProblem)
    reasoned_result = reasoned(problem=tricky_problem)
    print(f"  CoT reasoning: {reasoned_result.reasoning[:200]}...")
    print(f"  CoT answer: {reasoned_result.answer}")

    # Example 4: When CoT May Not Help
    print_step(
        "Example 4: When CoT May Not Help",
        "Simple tasks where reasoning adds unnecessary overhead"
    )

    class SimpleClassifier(dspy.Signature):
        """Classify the text as positive or negative."""
        text = dspy.InputField()
        sentiment = dspy.OutputField(desc="positive or negative")

    simple_cls = dspy.Predict(SimpleClassifier)
    cot_cls = dspy.ChainOfThought(SimpleClassifier)

    test_text = "I love this product! It's amazing!"

    simple_r = simple_cls(text=test_text)
    cot_r = cot_cls(text=test_text)

    print(f"  Text: {test_text}")
    print(f"  Predict: {simple_r.sentiment} (direct, fewer tokens)")
    print(f"  CoT: {cot_r.sentiment} (with reasoning: '{cot_r.reasoning[:80]}...')")
    print("\n  For simple classification, both give the same answer,")
    print("  but Predict uses fewer tokens. Save CoT for complex tasks.")

    # Example 5: CoT in a Custom Module
    print_step(
        "Example 5: CoT in Custom Modules",
        "Using ChainOfThought as part of a larger pipeline"
    )

    class DebateModule(dspy.Module):
        """Generate arguments for and against a position."""
        def __init__(self):
            super().__init__()

            class ArgumentFor(dspy.Signature):
                """Make the strongest argument in favor of the position."""
                position = dspy.InputField()
                argument = dspy.OutputField(desc="A compelling argument supporting the position")

            class ArgumentAgainst(dspy.Signature):
                """Make the strongest argument against the position."""
                position = dspy.InputField()
                argument = dspy.OutputField(desc="A compelling counter-argument")

            class Judge(dspy.Signature):
                """Evaluate both arguments and determine which is stronger."""
                position = dspy.InputField()
                for_argument = dspy.InputField()
                against_argument = dspy.InputField()
                verdict = dspy.OutputField(desc="Which side has the stronger argument and why")

            # CoT for nuanced evaluation
            self.argue_for = dspy.ChainOfThought(ArgumentFor)
            self.argue_against = dspy.ChainOfThought(ArgumentAgainst)
            self.judge = dspy.ChainOfThought(Judge)

        def forward(self, position):
            pro = self.argue_for(position=position)
            con = self.argue_against(position=position)
            verdict = self.judge(
                position=position,
                for_argument=pro.argument,
                against_argument=con.argument
            )
            return dspy.Prediction(
                pro_argument=pro.argument,
                con_argument=con.argument,
                verdict=verdict.verdict
            )

    debate = DebateModule()
    result = debate(position="Schools should replace textbooks with tablets")

    print_result(
        f"Position: Schools should replace textbooks with tablets\n\n"
        f"FOR: {result.pro_argument[:200]}...\n\n"
        f"AGAINST: {result.con_argument[:200]}...\n\n"
        f"VERDICT: {result.verdict[:200]}...",
        "Debate"
    )

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. ChainOfThought adds a 'reasoning' field to any signature")
    print("2. The model reasons step-by-step before giving the final answer")
    print("3. CoT significantly helps with complex, multi-step problems")
    print("4. For simple tasks (classification, extraction), Predict may be sufficient")
    print("5. Combine CoT with custom modules for sophisticated reasoning pipelines")


if __name__ == "__main__":
    main()
