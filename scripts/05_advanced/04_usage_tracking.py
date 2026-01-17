#!/usr/bin/env python3
"""
DSPy Usage Tracking - Monitor Token Usage and Costs

This script demonstrates how to use DSPy 3.x's built-in usage tracking to:
- Monitor token usage across your application
- Track costs per module or operation
- Optimize for efficiency
- Debug expensive operations

What You'll Learn:
- How to enable usage tracking in DSPy
- How to retrieve and analyze usage statistics
- How to track usage per module
- How to optimize based on usage data
- How to implement cost monitoring
"""

from dotenv import load_dotenv
from utils import (
    setup_default_lm,
    configure_dspy,
    get_usage_stats,
    print_usage_stats,
    print_step,
    print_result,
    print_error
)
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy usage tracking."""
    print("=== DSPy Usage Tracking Tutorial ===")
    print("Monitor and optimize token usage and costs")
    print("=" * 80)

    # Load environment variables
    load_dotenv('.env')

    # ========== PART 1: Enabling Usage Tracking ==========
    print_step(
        "Part 1: Enabling Usage Tracking",
        "How to configure DSPy to track token usage"
    )

    try:
        # Setup LM and enable usage tracking
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=500)

        # Enable usage tracking with configure_dspy
        # This tells DSPy to track all LM calls
        configure_dspy(lm=lm, track_usage=True)

        print_result("Language model configured with usage tracking enabled!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return

    print("\n  Usage tracking is now active!")
    print("  DSPy will automatically track:")
    print("  - Total tokens used (input + output)")
    print("  - Input tokens")
    print("  - Output tokens")
    print("  - Number of API calls")
    print("  - Cost estimates (when available)")

    # ========== PART 2: Basic Usage Tracking ==========
    print_step(
        "Part 2: Basic Usage Tracking",
        "Tracking usage for simple operations"
    )

    class QuestionAnswering(dspy.Signature):
        """Answer questions concisely."""
        question = dspy.InputField()
        answer = dspy.OutputField()

    qa = dspy.Predict(QuestionAnswering)

    # Make a few predictions
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?"
    ]

    print("\n  Making predictions...")
    for i, q in enumerate(questions, 1):
        result = qa(question=q)
        print(f"  {i}. Q: {q}")
        print(f"     A: {result.answer}")

    # Get and display usage statistics
    print("\n  Usage statistics after 3 predictions:")
    print_usage_stats()

    usage = get_usage_stats()
    if usage:
        # Calculate approximate cost (example rates for GPT-4o)
        input_cost_per_1k = 0.005  # $0.005 per 1K input tokens
        output_cost_per_1k = 0.015  # $0.015 per 1K output tokens

        if 'input_tokens' in usage and 'output_tokens' in usage:
            total_cost = (
                (usage['input_tokens'] / 1000) * input_cost_per_1k +
                (usage['output_tokens'] / 1000) * output_cost_per_1k
            )
            print(f"  Estimated cost: ${total_cost:.4f}")

    # ========== PART 3: Module-Level Usage Tracking ==========
    print_step(
        "Part 3: Module-Level Usage Tracking",
        "Track usage for custom modules"
    )

    class SmartQA(dspy.Module):
        """A multi-step QA system with usage tracking."""
        def __init__(self):
            super().__init__()

            class ClassifyQuestion(dspy.Signature):
                """Classify the question type."""
                question = dspy.InputField()
                question_type = dspy.OutputField(desc="factual, mathematical, or creative")

            class AnswerQuestion(dspy.Signature):
                """Answer based on question type."""
                question = dspy.InputField()
                question_type = dspy.InputField()
                answer = dspy.OutputField()

            self.classifier = dspy.Predict(ClassifyQuestion)
            self.answerer = dspy.ChainOfThought(AnswerQuestion)

        def forward(self, question):
            """Two-step QA with classification."""
            # Get initial usage
            initial_usage = get_usage_stats()

            # Step 1: Classify
            classification = self.classifier(question=question)

            # Step 2: Answer
            answer = self.answerer(
                question=question,
                question_type=classification.question_type
            )

            # Calculate usage for this module call
            final_usage = get_usage_stats()

            # Track usage delta if available
            usage_delta = {}
            if initial_usage and final_usage:
                for key in final_usage:
                    if key in initial_usage:
                        usage_delta[key] = final_usage[key] - initial_usage[key]

            return dspy.Prediction(
                question_type=classification.question_type,
                answer=answer.answer,
                reasoning=answer.reasoning,
                usage=usage_delta
            )

    smart_qa = SmartQA()

    print("\n  Testing multi-step module...")
    test_question = "If I have 10 apples and give away 3, how many do I have left?"

    result = smart_qa(question=test_question)
    print(f"\n  Question: {test_question}")
    print(f"  Type: {result.question_type}")
    print(f"  Reasoning: {result.reasoning}")
    print(f"  Answer: {result.answer}")

    if result.usage:
        print(f"\n  Usage for this module call:")
        for key, value in result.usage.items():
            print(f"    {key}: {value}")

    # ========== PART 4: Comparing Approaches ==========
    print_step(
        "Part 4: Comparing Approaches",
        "Use tracking to compare efficiency of different implementations"
    )

    # Reset usage tracking (if possible)
    print("\n  Comparing Predict vs ChainOfThought...")

    class SimpleSignature(dspy.Signature):
        """Solve a problem."""
        problem = dspy.InputField()
        solution = dspy.OutputField()

    # Approach 1: Simple Predict
    print("\n  Approach 1: Using dspy.Predict")
    simple_predictor = dspy.Predict(SimpleSignature)

    usage_before = get_usage_stats()
    result1 = simple_predictor(problem="What is 7 * 8?")
    usage_after = get_usage_stats()

    if usage_before and usage_after:
        tokens_used = usage_after.get('total_tokens', 0) - usage_before.get('total_tokens', 0)
        print(f"  Tokens used: ~{tokens_used}")
    print(f"  Solution: {result1.solution}")

    # Approach 2: ChainOfThought
    print("\n  Approach 2: Using dspy.ChainOfThought")
    cot_predictor = dspy.ChainOfThought(SimpleSignature)

    usage_before = get_usage_stats()
    result2 = cot_predictor(problem="What is 7 * 8?")
    usage_after = get_usage_stats()

    if usage_before and usage_after:
        tokens_used = usage_after.get('total_tokens', 0) - usage_before.get('total_tokens', 0)
        print(f"  Tokens used: ~{tokens_used}")
    print(f"  Solution: {result2.solution}")

    print("\n  Key insight:")
    print("  ChainOfThought typically uses more tokens (includes reasoning)")
    print("  but may provide better quality answers for complex problems.")

    # ========== PART 5: Cost Monitoring ==========
    print_step(
        "Part 5: Cost Monitoring",
        "Track and alert on costs"
    )

    class CostMonitor:
        """Monitor and track LM costs."""

        def __init__(self, model="gpt-4o"):
            """Initialize with model-specific pricing."""
            # Pricing per 1K tokens (example rates)
            self.pricing = {
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
                "claude-3-7-sonnet": {"input": 0.003, "output": 0.015},
            }
            self.model = model
            self.initial_usage = get_usage_stats()
            self.cost_limit = None

        def get_current_cost(self):
            """Calculate current session cost."""
            current_usage = get_usage_stats()
            if not current_usage or not self.initial_usage:
                return 0.0

            input_tokens = current_usage.get('input_tokens', 0) - self.initial_usage.get('input_tokens', 0)
            output_tokens = current_usage.get('output_tokens', 0) - self.initial_usage.get('output_tokens', 0)

            pricing = self.pricing.get(self.model, {"input": 0.005, "output": 0.015})

            cost = (
                (input_tokens / 1000) * pricing['input'] +
                (output_tokens / 1000) * pricing['output']
            )
            return cost

        def set_limit(self, limit_usd):
            """Set cost limit for alerts."""
            self.cost_limit = limit_usd

        def check_limit(self):
            """Check if cost limit exceeded."""
            if self.cost_limit is None:
                return False

            current_cost = self.get_current_cost()
            if current_cost > self.cost_limit:
                print_error(f"Cost limit exceeded! Current: ${current_cost:.4f}, Limit: ${self.cost_limit:.4f}")
                return True
            return False

        def report(self):
            """Print cost report."""
            cost = self.get_current_cost()
            usage = get_usage_stats()
            initial = self.initial_usage

            print("\n  Cost Monitoring Report:")
            print(f"  Model: {self.model}")

            if usage and initial:
                input_delta = usage.get('input_tokens', 0) - initial.get('input_tokens', 0)
                output_delta = usage.get('output_tokens', 0) - initial.get('output_tokens', 0)
                print(f"  Input tokens: {input_delta:,}")
                print(f"  Output tokens: {output_delta:,}")

            print(f"  Estimated cost: ${cost:.4f}")

            if self.cost_limit:
                remaining = self.cost_limit - cost
                print(f"  Budget remaining: ${remaining:.4f}")

    # Use cost monitor
    monitor = CostMonitor(model="gpt-4o")
    monitor.set_limit(0.10)  # $0.10 limit

    print("\n  Cost monitor initialized with $0.10 limit")

    # Make some predictions
    for _ in range(2):
        qa(question="What is the meaning of life?")

    monitor.report()
    monitor.check_limit()

    # ========== PART 6: Best Practices ==========
    print_step(
        "Part 6: Best Practices",
        "Tips for effective usage tracking"
    )

    print("  1. Always enable tracking in development:")
    print("     configure_dspy(lm=lm, track_usage=True)")
    print()
    print("  2. Monitor usage during development:")
    print("     - Identify expensive operations")
    print("     - Compare different implementations")
    print("     - Set cost budgets for testing")
    print()
    print("  3. Optimize based on data:")
    print("     - Use simpler modules when possible (Predict vs ChainOfThought)")
    print("     - Reduce max_tokens for simple tasks")
    print("     - Cache responses when appropriate")
    print("     - Use cheaper models for simple operations")
    print()
    print("  4. Production monitoring:")
    print("     - Set up cost alerts")
    print("     - Log usage per user/session")
    print("     - Track trends over time")
    print("     - Budget for different features")
    print()
    print("  5. Cost optimization strategies:")
    print("     - Use gpt-4o-mini for simple tasks")
    print("     - Implement caching for repeated queries")
    print("     - Batch similar requests")
    print("     - Use streaming for long outputs")
    print("     - Set appropriate max_tokens limits")

    # ========== PART 7: Usage Tracking Template ==========
    print_step(
        "Part 7: Production Usage Tracking Template",
        "Complete example for production use"
    )

    print("""
  Production-ready usage tracking pattern:

  ```python
  import dspy
  from datetime import datetime

  class UsageTracker:
      def __init__(self, log_file='usage.log'):
          self.log_file = log_file
          self.sessions = []

      def start_session(self, user_id, feature):
          session = {
              'user_id': user_id,
              'feature': feature,
              'start_time': datetime.now(),
              'start_usage': dspy.get_lm_usage()
          }
          self.sessions.append(session)
          return len(self.sessions) - 1

      def end_session(self, session_id):
          session = self.sessions[session_id]
          session['end_time'] = datetime.now()
          session['end_usage'] = dspy.get_lm_usage()

          # Calculate deltas
          start = session['start_usage']
          end = session['end_usage']
          session['tokens_used'] = end['total_tokens'] - start['total_tokens']
          session['duration'] = (session['end_time'] - session['start_time']).total_seconds()

          # Log to file
          with open(self.log_file, 'a') as f:
              f.write(f"{session}\\n")

          return session

  # Usage
  tracker = UsageTracker()

  # Start tracking
  session_id = tracker.start_session(user_id='user123', feature='qa_system')

  # Do work
  result = qa(question="Your question here")

  # End tracking
  session_data = tracker.end_session(session_id)
  print(f"Session cost: ${session_data['cost']:.4f}")
  ```
  """)

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. Enable usage tracking with configure_dspy(track_usage=True)")
    print("2. Use get_usage_stats() to retrieve current usage")
    print("3. Track usage deltas for module-level monitoring")
    print("4. Compare approaches to optimize for efficiency")
    print("5. Implement cost monitoring and alerts")
    print("6. Log usage data for production monitoring")
    print("7. Optimize based on real usage data")


if __name__ == "__main__":
    main()
