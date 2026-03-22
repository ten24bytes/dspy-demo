#!/usr/bin/env python3
"""
DSPy ReAct Agents

This script demonstrates dspy.ReAct - the module for building agents that
reason and act by using tools in an iterative loop.

What You'll Learn:
- How ReAct (Reasoning + Acting) works
- Defining tools for agents
- Building a simple agent with dspy.ReAct
- Observation/action loops
- When to use ReAct vs simpler modules
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# Define tools that the agent can use
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 3 * 4')
    """
    try:
        # Safe evaluation of math expressions
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def word_count(text: str) -> str:
    """Count the number of words in the given text.

    Args:
        text: The text to count words in
    """
    count = len(text.split())
    return f"{count} words"


def reverse_string(text: str) -> str:
    """Reverse the given string.

    Args:
        text: The text to reverse
    """
    return text[::-1]


def lookup_capital(country: str) -> str:
    """Look up the capital city of a country.

    Args:
        country: The name of the country
    """
    capitals = {
        "france": "Paris",
        "japan": "Tokyo",
        "brazil": "Brasilia",
        "australia": "Canberra",
        "egypt": "Cairo",
        "canada": "Ottawa",
        "germany": "Berlin",
        "india": "New Delhi",
    }
    result = capitals.get(country.lower().strip())
    if result:
        return f"The capital of {country} is {result}"
    return f"Capital not found for {country}"


def main():
    """Main function demonstrating dspy.ReAct."""
    print("=== DSPy ReAct Agents ===")
    print("Building agents that reason and act with tools")
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

    # Example 1: Simple Calculator Agent
    print_step(
        "Example 1: Calculator Agent",
        "An agent that uses a calculator tool to solve math problems"
    )

    class MathQuestion(dspy.Signature):
        """Answer the math question using available tools."""
        question = dspy.InputField(desc="A math question")
        answer = dspy.OutputField(desc="The final numerical answer")

    math_agent = dspy.ReAct(MathQuestion, tools=[calculator])

    questions = [
        "What is 15 * 23 + 47?",
        "What is 1024 / 16?",
    ]

    for q in questions:
        try:
            result = math_agent(question=q)
            print_result(f"Q: {q}\nA: {result.answer}", "Calculator Agent")
        except Exception as e:
            print_error(f"Error on '{q}': {e}")

    # Example 2: Multi-Tool Agent
    print_step(
        "Example 2: Multi-Tool Agent",
        "An agent with access to multiple tools"
    )

    class GeneralQuestion(dspy.Signature):
        """Answer the question using available tools. Use tools when they can help."""
        question = dspy.InputField(desc="A general question")
        answer = dspy.OutputField(desc="The answer to the question")

    multi_agent = dspy.ReAct(
        GeneralQuestion,
        tools=[calculator, word_count, lookup_capital],
        max_iters=5
    )

    multi_questions = [
        "What is the capital of Japan?",
        "What is 256 * 3 + 42?",
    ]

    for q in multi_questions:
        try:
            result = multi_agent(question=q)
            print_result(f"Q: {q}\nA: {result.answer}", "Multi-Tool Agent")
        except Exception as e:
            print_error(f"Error on '{q}': {e}")

    # Example 3: How ReAct Works
    print_step(
        "Example 3: Understanding the ReAct Loop",
        "How the agent reasons and acts step by step"
    )

    print("  The ReAct pattern works in a loop:")
    print("  1. THINK: The agent reasons about what to do next")
    print("  2. ACT: It chooses a tool and provides arguments")
    print("  3. OBSERVE: It receives the tool's output")
    print("  4. REPEAT: Back to thinking with new information")
    print("  5. FINISH: When it has enough info, it produces the final answer")
    print()
    print("  Example trace for 'What is the capital of France?':")
    print("    Thought: I need to look up the capital of France")
    print("    Action: lookup_capital('France')")
    print("    Observation: The capital of France is Paris")
    print("    Thought: I now know the answer")
    print("    Answer: Paris")

    # Example 4: Custom Tool Functions
    print_step(
        "Example 4: Writing Good Tool Functions",
        "Best practices for defining agent tools"
    )

    print("  Tool function requirements:")
    print("  1. Clear function name (the agent uses it to decide which tool to call)")
    print("  2. Descriptive docstring (the agent reads this to understand the tool)")
    print("  3. Type-annotated parameters with descriptions")
    print("  4. Return a string result")
    print()
    print("  Example of a well-defined tool:")
    print('    def search_database(query: str) -> str:')
    print('        """Search the product database for items matching the query.')
    print('        ')
    print('        Args:')
    print('            query: Search terms to find products')
    print('        """')
    print('        # ... implementation')
    print('        return f"Found {len(results)} products"')

    # Example 5: When to Use ReAct
    print_step(
        "Example 5: When to Use ReAct",
        "Choosing ReAct vs simpler approaches"
    )

    print("  Use ReAct when:")
    print("  - The task requires external tools or data lookup")
    print("  - The problem needs multiple steps with tool interactions")
    print("  - You need the agent to decide which tools to use")
    print("  - The task involves exploration or search")
    print()
    print("  Use ChainOfThought instead when:")
    print("  - No tools are needed (pure reasoning)")
    print("  - The answer comes from the model's knowledge")
    print("  - You want faster, cheaper responses")
    print()
    print("  Use Predict when:")
    print("  - Simple, direct answers needed")
    print("  - No reasoning or tools required")
    print()
    print("  Key parameters for ReAct:")
    print("  - tools: List of callable tool functions")
    print("  - max_iters: Maximum reasoning-action cycles (default: 20)")

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. ReAct combines reasoning (Think) with acting (Tool Use)")
    print("2. Define tools as Python functions with clear docstrings")
    print("3. The agent decides which tool to call and with what arguments")
    print("4. Use max_iters to limit the number of reasoning-action cycles")
    print("5. Best for tasks that require external tools or multi-step interactions")


if __name__ == "__main__":
    main()
