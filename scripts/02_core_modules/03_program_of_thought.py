#!/usr/bin/env python3
"""
DSPy ProgramOfThought Module

This script demonstrates dspy.ProgramOfThought - a module that generates and
executes Python code to solve problems, combining language models with computation.

What You'll Learn:
- How ProgramOfThought works (generate code, execute, return result)
- When to use ProgramOfThought vs ChainOfThought
- Solving math and data problems with code generation
- Safety considerations for code execution
- Building custom code-generation workflows
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating dspy.ProgramOfThought."""
    print("=== DSPy ProgramOfThought Module ===")
    print("Solving problems by generating and executing code")
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

    # Example 1: Basic ProgramOfThought
    print_step(
        "Example 1: Basic ProgramOfThought",
        "Solving a math problem by generating Python code"
    )

    class MathSolver(dspy.Signature):
        """Solve the math problem by writing Python code."""
        problem = dspy.InputField(desc="A mathematical problem to solve")
        answer = dspy.OutputField(desc="The numerical answer")

    pot_solver = dspy.ProgramOfThought(MathSolver)

    math_problem = "What is the sum of the first 20 prime numbers?"

    try:
        result = pot_solver(problem=math_problem)
        print_result(
            f"Problem: {math_problem}\nAnswer: {result.answer}",
            "ProgramOfThought"
        )
    except Exception as e:
        print_error(f"ProgramOfThought error: {e}")
        print("  This is expected if code execution is restricted in your environment.")

    # Example 2: Comparing CoT vs PoT
    print_step(
        "Example 2: ChainOfThought vs ProgramOfThought",
        "When code execution gives more precise results"
    )

    class PrecisionProblem(dspy.Signature):
        """Solve this problem accurately."""
        problem = dspy.InputField()
        answer = dspy.OutputField(desc="The precise numerical answer")

    complex_problem = "If you invest $1000 at 5% annual compound interest, how much money will you have after 10 years? Round to 2 decimal places."

    # CoT approach - model reasons in text
    cot_solver = dspy.ChainOfThought(PrecisionProblem)
    try:
        cot_result = cot_solver(problem=complex_problem)
        print(f"  ChainOfThought answer: {cot_result.answer}")
        print(f"  Reasoning: {cot_result.reasoning[:150]}...")
    except Exception as e:
        print(f"  CoT error: {e}")

    # PoT approach - model writes code and executes it
    pot_solver = dspy.ProgramOfThought(PrecisionProblem)
    try:
        pot_result = pot_solver(problem=complex_problem)
        print(f"\n  ProgramOfThought answer: {pot_result.answer}")
    except Exception as e:
        print(f"\n  PoT error: {e}")
        print("  (Code execution may be restricted)")

    # Correct answer for reference
    correct = round(1000 * (1.05 ** 10), 2)
    print(f"\n  Correct answer: ${correct}")

    # Example 3: Data Processing Problems
    print_step(
        "Example 3: Data Processing",
        "ProgramOfThought for data manipulation tasks"
    )

    class DataProcessor(dspy.Signature):
        """Solve the data processing problem by writing Python code."""
        problem = dspy.InputField(desc="A data processing task")
        answer = dspy.OutputField(desc="The result")

    data_pot = dspy.ProgramOfThought(DataProcessor)

    data_problem = """Given the list of scores [85, 92, 78, 95, 88, 76, 90, 82, 94, 87],
    calculate the mean, median, and standard deviation. Format as: mean=X, median=Y, std=Z (rounded to 2 decimals)."""

    try:
        result = data_pot(problem=data_problem)
        print_result(f"Problem: {data_problem.strip()}\n\nAnswer: {result.answer}", "Data Processing")
    except Exception as e:
        print_error(f"Error: {e}")

    # Example 4: When to Use ProgramOfThought
    print_step(
        "Example 4: When to Use ProgramOfThought",
        "Choosing the right module for the task"
    )

    print("  Use ProgramOfThought when:")
    print("  - The problem requires precise numerical computation")
    print("  - You need data manipulation (sorting, filtering, aggregation)")
    print("  - The task involves mathematical formulas")
    print("  - Accuracy is more important than speed")
    print("  - The problem can be solved with a short Python script")
    print()
    print("  Use ChainOfThought instead when:")
    print("  - The problem is primarily about language/reasoning")
    print("  - No computation is needed")
    print("  - You need fast responses (no code execution overhead)")
    print("  - The environment restricts code execution")
    print()
    print("  Use Predict when:")
    print("  - Simple factual answers or classification")
    print("  - No reasoning or computation needed")

    # Example 5: Custom Code Generation Module
    print_step(
        "Example 5: Custom Code Generation",
        "Building a module that uses code generation without ProgramOfThought"
    )

    class CodeGenerator(dspy.Module):
        """Generate Python code to solve a problem (without executing it)."""
        def __init__(self):
            super().__init__()

            class GenerateCode(dspy.Signature):
                """Write Python code to solve the given problem. Return only the code."""
                problem = dspy.InputField(desc="Problem description")
                python_code = dspy.OutputField(desc="Python code that solves the problem")

            class ExplainCode(dspy.Signature):
                """Explain what the code does in plain English."""
                code = dspy.InputField(desc="Python code")
                explanation = dspy.OutputField(desc="Plain English explanation of the code")

            self.generate = dspy.Predict(GenerateCode)
            self.explain = dspy.Predict(ExplainCode)

        def forward(self, problem):
            code = self.generate(problem=problem)
            explanation = self.explain(code=code.python_code)
            return dspy.Prediction(
                code=code.python_code,
                explanation=explanation.explanation
            )

    code_gen = CodeGenerator()
    result = code_gen(problem="Find all palindromes in a list of words")

    print_result(
        f"Generated Code:\n{result.code}\n\nExplanation: {result.explanation}",
        "Code Generation"
    )

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. ProgramOfThought generates and executes Python code to solve problems")
    print("2. Best for numerical computation, data processing, and formula-based tasks")
    print("3. More precise than ChainOfThought for math problems")
    print("4. Has overhead from code generation + execution")
    print("5. You can build custom code-generation modules for more control")


if __name__ == "__main__":
    main()
