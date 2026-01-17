#!/usr/bin/env python3
"""
DSPy Reasoning Module - Working with Reasoning Models

This script demonstrates how to use the dspy.Reasoning module with reasoning-capable models
like OpenAI's o1 models or Claude models with extended thinking capabilities.

The Reasoning module captures and exposes the model's internal reasoning process,
making it visible for debugging, validation, and learning purposes.

What You'll Learn:
- How to use dspy.Reasoning with reasoning models
- How to capture and inspect reasoning traces
- How to use reasoning for complex problem solving
- How to combine reasoning with other DSPy modules
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy Reasoning module."""
    print("=== DSPy Reasoning Module Tutorial ===")
    print("Working with reasoning-capable language models")
    print("=" * 80)

    # Load environment variables
    load_dotenv('.env')

    # Setup Language Model
    # Note: For best results with reasoning, use models like:
    # - OpenAI: gpt-4o (has reasoning capabilities)
    # - Anthropic: claude-3-7-sonnet-20250219 (extended thinking)
    print_step("Setting up Language Model", "Configuring DSPy with a reasoning-capable model")

    try:
        # Using GPT-4o which has good reasoning capabilities
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=2000)
        configure_dspy(lm=lm)
        print_result("Language model configured successfully!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
        return

    # Example 1: Basic Reasoning
    print_step(
        "Example 1: Basic Reasoning",
        "Using dspy.Reasoning to solve a complex problem with visible reasoning steps"
    )

    class ComplexProblemSolver(dspy.Signature):
        """Solve a complex problem with detailed reasoning."""
        problem = dspy.InputField(desc="A complex problem requiring multi-step reasoning")
        # The reasoning field captures the model's thinking process
        reasoning = dspy.OutputField(desc="Step-by-step reasoning and thought process")
        solution = dspy.OutputField(desc="The final solution to the problem")

    # Create a Reasoning module
    # This is similar to ChainOfThought but specifically designed for reasoning models
    reasoning_solver = dspy.Reasoning(ComplexProblemSolver)

    # Test with a complex logic problem
    problem1 = """
    Three friends - Alice, Bob, and Carol - are standing in a line.
    - Alice is not at the front.
    - Bob is not at the back.
    - Carol is not in the middle.
    What is the order of the three friends from front to back?
    """

    result1 = reasoning_solver(problem=problem1)
    print_result(
        f"Problem: {problem1.strip()}\n\n"
        f"Reasoning Process:\n{result1.reasoning}\n\n"
        f"Solution: {result1.solution}",
        "Logic Problem"
    )

    # Example 2: Mathematical Reasoning
    print_step(
        "Example 2: Mathematical Reasoning",
        "Solving complex math problems with detailed steps"
    )

    class MathReasoning(dspy.Signature):
        """Solve a mathematical problem with detailed reasoning."""
        problem = dspy.InputField(desc="A mathematical problem to solve")
        reasoning = dspy.OutputField(desc="Mathematical reasoning and steps")
        answer = dspy.OutputField(desc="The numerical answer")

    math_reasoner = dspy.Reasoning(MathReasoning)

    math_problem = """
    A train travels from City A to City B at 60 mph. On the return trip,
    it travels at 40 mph. What is the average speed for the entire round trip?
    (Note: This is a trick question - the answer is NOT 50 mph!)
    """

    result2 = math_reasoner(problem=math_problem)
    print_result(
        f"Problem: {math_problem.strip()}\n\n"
        f"Reasoning:\n{result2.reasoning}\n\n"
        f"Answer: {result2.answer}",
        "Math Problem"
    )

    # Example 3: Combining Reasoning with Custom Modules
    print_step(
        "Example 3: Advanced Reasoning Module",
        "Building a custom module that uses reasoning for problem decomposition"
    )

    class ProblemDecomposer(dspy.Module):
        """
        A custom module that breaks down complex problems into steps,
        solves each step with reasoning, and combines the results.
        """
        def __init__(self):
            super().__init__()

            # Signature for decomposing problems into sub-problems
            class DecomposeProblem(dspy.Signature):
                """Break down a complex problem into simpler sub-problems."""
                problem = dspy.InputField(desc="A complex problem")
                sub_problems = dspy.OutputField(desc="List of 2-4 simpler sub-problems, one per line")

            # Signature for solving individual sub-problems
            class SolveSubProblem(dspy.Signature):
                """Solve a sub-problem with reasoning."""
                sub_problem = dspy.InputField(desc="A sub-problem to solve")
                reasoning = dspy.OutputField(desc="Reasoning process")
                solution = dspy.OutputField(desc="Solution to the sub-problem")

            # Signature for combining solutions
            class CombineSolutions(dspy.Signature):
                """Combine sub-problem solutions into a final answer."""
                original_problem = dspy.InputField(desc="The original problem")
                solutions = dspy.InputField(desc="Solutions to all sub-problems")
                final_answer = dspy.OutputField(desc="The comprehensive final answer")

            # Initialize predictors
            self.decompose = dspy.Predict(DecomposeProblem)
            self.solve_step = dspy.Reasoning(SolveSubProblem)
            self.combine = dspy.Predict(CombineSolutions)

        def forward(self, problem):
            """
            Forward pass: decompose → solve each step → combine solutions.
            """
            # Step 1: Decompose the problem
            decomposition = self.decompose(problem=problem)
            sub_problems = [sp.strip() for sp in decomposition.sub_problems.split('\n') if sp.strip()]

            print(f"\n  Decomposed into {len(sub_problems)} sub-problems:")
            for i, sp in enumerate(sub_problems, 1):
                print(f"    {i}. {sp}")

            # Step 2: Solve each sub-problem with reasoning
            solutions = []
            for i, sub_problem in enumerate(sub_problems, 1):
                print(f"\n  Solving sub-problem {i}...")
                solution = self.solve_step(sub_problem=sub_problem)
                solutions.append(f"Sub-problem {i}: {sub_problem}\nReasoning: {solution.reasoning}\nSolution: {solution.solution}")

            # Step 3: Combine solutions
            combined_solutions = "\n\n".join(solutions)
            final = self.combine(
                original_problem=problem,
                solutions=combined_solutions
            )

            return dspy.Prediction(
                sub_problems=sub_problems,
                solutions=solutions,
                final_answer=final.final_answer
            )

    # Create and test the decomposer
    decomposer = ProblemDecomposer()

    complex_problem = """
    A company wants to organize a team-building event for 150 employees.
    They have a budget of $7,500. They need to:
    1. Choose a venue (indoor or outdoor)
    2. Arrange catering (breakfast, lunch, and snacks)
    3. Plan 3 team activities
    4. Arrange transportation if needed

    Create a comprehensive plan that stays within budget and maximizes employee engagement.
    """

    print("\n  Starting problem decomposition and solving...")
    result3 = decomposer(problem=complex_problem.strip())

    print_result(
        f"Original Problem: {complex_problem.strip()}\n\n"
        f"Final Comprehensive Solution:\n{result3.final_answer}",
        "Complex Planning Problem"
    )

    # Example 4: Comparing Reasoning vs ChainOfThought
    print_step(
        "Example 4: Reasoning vs ChainOfThought",
        "Understanding the differences between dspy.Reasoning and dspy.ChainOfThought"
    )

    class ProblemSolver(dspy.Signature):
        """Solve a problem."""
        problem = dspy.InputField()
        reasoning = dspy.OutputField()
        answer = dspy.OutputField()

    # Create both versions
    reasoning_module = dspy.Reasoning(ProblemSolver)
    cot_module = dspy.ChainOfThought(ProblemSolver)

    test_problem = "If you're in a race and you pass the person in 2nd place, what place are you in?"

    print("\n  Using dspy.Reasoning:")
    reasoning_result = reasoning_module(problem=test_problem)
    print(f"  Reasoning: {reasoning_result.reasoning[:200]}...")
    print(f"  Answer: {reasoning_result.answer}")

    print("\n  Using dspy.ChainOfThought:")
    cot_result = cot_module(problem=test_problem)
    print(f"  Reasoning: {cot_result.reasoning[:200]}...")
    print(f"  Answer: {cot_result.answer}")

    print("\n  Key Differences:")
    print("  - dspy.Reasoning: Designed for reasoning-capable models (o1, Claude thinking)")
    print("    → Captures native reasoning traces from the model")
    print("    → Better for complex, multi-step problems")
    print("  - dspy.ChainOfThought: Works with all models")
    print("    → Prompts model to show reasoning via instruction")
    print("    → Good for general step-by-step thinking")

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. dspy.Reasoning captures native reasoning from reasoning-capable models")
    print("2. Use it for complex problems that benefit from detailed thinking")
    print("3. The reasoning trace is visible and can be used for debugging/validation")
    print("4. Combine with custom modules for advanced problem-solving workflows")
    print("5. For best results, use with models that support native reasoning (o1, Claude)")


if __name__ == "__main__":
    main()
