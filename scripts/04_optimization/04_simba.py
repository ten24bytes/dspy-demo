#!/usr/bin/env python3
"""
SIMBA Optimizer - Self-Reflective Improvement for DSPy Programs

This script demonstrates how to use the SIMBA optimizer in DSPy 3.x.
SIMBA uses self-reflective improvement rules to iteratively optimize
prompts and few-shot examples.

What You'll Learn:
- How SIMBA works (self-reflective improvement)
- How to configure SIMBA parameters
- How SIMBA differs from GEPA, BootstrapFewShot, and MIPROv2
- When to use SIMBA vs other optimizers
- A production-ready SIMBA optimization template
"""

from dotenv import load_dotenv
from utils import setup_default_lm, configure_dspy, print_step, print_result, print_error
from utils.datasets import get_sample_qa_data
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating SIMBA optimizer."""
    print("=== SIMBA Optimizer Tutorial ===")
    print("Self-reflective improvement for DSPy programs")
    print("=" * 80)

    # Load environment variables
    load_dotenv('.env')

    # Setup Language Model
    print_step("Setting up Language Model", "Configuring DSPy for optimization")

    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=500)
        configure_dspy(lm=lm)
        print_result("Language model configured successfully!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return

    # ========== PART 1: Understanding SIMBA ==========
    print_step(
        "Part 1: Understanding SIMBA",
        "What makes SIMBA different from other optimizers"
    )

    print("  SIMBA (Self-Improving Model-Based Adaptation) works by:")
    print("  1. Running the current program on training examples")
    print("  2. Identifying failures and analyzing error patterns")
    print("  3. Generating candidate improvements (instructions, demos)")
    print("  4. Evaluating candidates and keeping the best ones")
    print("  5. Repeating for multiple improvement steps")
    print()
    print("  Key characteristics:")
    print("  - Iterative self-improvement with reflection")
    print("  - Optimizes both instructions and few-shot demos")
    print("  - Uses the LM itself to propose improvements")
    print("  - Batch-based evaluation for efficiency")

    # ========== PART 2: Basic SIMBA Example ==========
    print_step(
        "Part 2: Basic SIMBA Setup",
        "Setting up a QA system for SIMBA optimization"
    )

    class QuestionAnswering(dspy.Signature):
        """Answer questions accurately and concisely."""
        question = dspy.InputField(desc="A question to answer")
        answer = dspy.OutputField(desc="A concise and accurate answer")

    class QASystem(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.ChainOfThought(QuestionAnswering)

        def forward(self, question):
            result = self.qa(question=question)
            return dspy.Prediction(
                answer=result.answer,
                reasoning=result.reasoning
            )

    # Load training data
    print("\n  Loading sample QA dataset...")
    all_data = get_sample_qa_data()
    train_data = all_data[:8]
    val_data = all_data[8:]
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")

    # Define metric
    def qa_metric(example, prediction, trace=None):
        """Evaluate QA predictions."""
        pred_answer = prediction.answer.lower().strip()
        true_answer = example.answer.lower().strip()

        if pred_answer == true_answer:
            return 1.0

        true_words = set(true_answer.split())
        pred_words = set(pred_answer.split())
        overlap = len(true_words & pred_words)

        if overlap > 0:
            return overlap / max(len(true_words), len(pred_words))
        return 0.0

    # Test baseline
    print("\n  Testing baseline performance...")
    baseline = QASystem()
    baseline_scores = []
    for example in val_data[:3]:
        try:
            prediction = baseline(question=example.question)
            score = qa_metric(example, prediction)
            baseline_scores.append(score)
            print(f"    Q: {example.question[:60]}...")
            print(f"    Expected: {example.answer}")
            print(f"    Got: {prediction.answer}")
            print(f"    Score: {score:.2f}\n")
        except Exception as e:
            print(f"    Error: {e}")
            baseline_scores.append(0.0)

    avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    print(f"  Baseline average score: {avg_baseline:.2f}")

    # ========== PART 3: SIMBA Configuration ==========
    print_step(
        "Part 3: SIMBA Configuration",
        "Understanding and tuning SIMBA parameters"
    )

    print("  Key SIMBA Parameters:")
    print()
    print("  1. metric (required)")
    print("     - Evaluation function: (example, prediction) -> float")
    print("     - Used to evaluate candidates and track improvement")
    print()
    print("  2. bsize (default: 32)")
    print("     - Batch size for evaluation")
    print("     - Larger = more robust evaluation, slower per step")
    print("     - Recommended: 16-64")
    print()
    print("  3. num_candidates (default: 6)")
    print("     - Number of improvement candidates per step")
    print("     - More candidates = better exploration, more API calls")
    print("     - Recommended: 4-10")
    print()
    print("  4. max_steps (default: 8)")
    print("     - Maximum improvement iterations")
    print("     - More steps = potentially better results, longer time")
    print("     - Recommended: 5-15")
    print()
    print("  5. max_demos (default: 4)")
    print("     - Maximum few-shot demonstrations to include")
    print("     - More demos = more context, higher token cost")
    print("     - Recommended: 2-8")
    print()
    print("  6. num_threads (default: None)")
    print("     - Parallel evaluation threads")
    print("     - Set to 4-8 for faster optimization")
    print()
    print("  7. temperature_for_sampling (default: 0.2)")
    print("     - Temperature for generating candidate improvements")
    print("     - Higher = more diverse candidates")

    # ========== PART 4: Running SIMBA ==========
    print_step(
        "Part 4: Running SIMBA Optimization",
        "Optimizing the QA system with SIMBA"
    )

    print("  Note: SIMBA optimization can take several minutes.")
    print("  For demonstration, we show the setup and explain the process.")
    print()

    try:
        print("  Initializing SIMBA optimizer...")

        simba_optimizer = dspy.SIMBA(
            metric=qa_metric,
            bsize=8,               # Small for demo
            num_candidates=3,      # Small for demo; use 6+ in practice
            max_steps=2,           # Small for demo; use 5-10 in practice
            max_demos=3,
            num_threads=1
        )

        print("\n  SIMBA optimization process:")
        print("  Step 1: Evaluate current program on a batch of training examples")
        print("  Step 2: Identify examples where the program fails or scores low")
        print("  Step 3: Generate candidate improvements (new instructions, better demos)")
        print("  Step 4: Evaluate each candidate on the training batch")
        print("  Step 5: Keep the best candidate and repeat")
        print()

        # Uncomment to actually run optimization:
        # optimized = simba_optimizer.compile(student=QASystem(), trainset=train_data)
        # print("  Optimization complete!")

        print("  [Optimization code is commented out for demo purposes]")
        print("  In practice, SIMBA typically improves scores through iterative refinement.")

    except Exception as e:
        print_error(f"SIMBA initialization error: {e}")

    # ========== PART 5: SIMBA vs Other Optimizers ==========
    print_step(
        "Part 5: Choosing the Right Optimizer",
        "When to use SIMBA vs GEPA vs BootstrapFewShot vs MIPROv2"
    )

    print("  SIMBA - Best for: Iterative refinement with self-reflection")
    print("  - Learns from failures to improve prompts")
    print("  - Good with moderate data (20-100 examples)")
    print("  - Moderate optimization time")
    print("  - Works well for instruction + demo optimization")
    print()
    print("  GEPA - Best for: Multi-objective optimization")
    print("  - Genetic algorithms + Pareto optimization")
    print("  - Best for complex tasks with multiple goals")
    print("  - Longest optimization time")
    print("  - Needs more training data (50+)")
    print()
    print("  BootstrapFewShot - Best for: Quick improvements")
    print("  - Simplest and fastest optimizer")
    print("  - Generates and selects few-shot examples")
    print("  - Works with very little data (<20 examples)")
    print("  - Good starting point before trying advanced optimizers")
    print()
    print("  MIPROv2 - Best for: Balanced optimization")
    print("  - Bayesian optimization approach")
    print("  - Efficient hyperparameter tuning")
    print("  - Good balance of speed and quality")
    print("  - Works well with 20-50 examples")

    # ========== PART 6: Production Template ==========
    print_step(
        "Part 6: Complete SIMBA Optimization Template",
        "Production-ready code structure"
    )

    print("""
    # 1. Define your module
    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought(MySignature)

        def forward(self, **inputs):
            result = self.predictor(**inputs)
            return dspy.Prediction(...)

    # 2. Prepare data
    train_data = [...]  # 20-100 examples
    val_data = [...]    # 10-30 examples

    # 3. Define metric
    def my_metric(example, prediction, trace=None):
        score = ...  # 0-1 score
        return score

    # 4. Configure SIMBA
    optimizer = dspy.SIMBA(
        metric=my_metric,
        bsize=32,
        num_candidates=6,
        max_steps=8,
        max_demos=4,
        num_threads=4
    )

    # 5. Optimize
    optimized = optimizer.compile(
        student=MyModule(),
        trainset=train_data
    )

    # 6. Evaluate
    scores = [my_metric(ex, optimized(**ex.inputs())) for ex in val_data]
    print(f"Average score: {sum(scores) / len(scores):.3f}")

    # 7. Save
    optimized.save("models/my_simba_optimized_model")
    """)

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. SIMBA uses self-reflective improvement to optimize programs")
    print("2. It learns from failures and iteratively proposes better prompts")
    print("3. Key parameters: bsize, num_candidates, max_steps, max_demos")
    print("4. Works well with moderate training data (20-100 examples)")
    print("5. Best when the task benefits from iterative prompt refinement")
    print("6. Complements GEPA - SIMBA for refinement, GEPA for exploration")
    print("7. Start with BootstrapFewShot, then try SIMBA or GEPA for more improvement")


if __name__ == "__main__":
    main()
