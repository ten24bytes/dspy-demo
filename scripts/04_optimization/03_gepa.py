#!/usr/bin/env python3
"""
GEPA Optimizer - Genetic-Pareto Optimization with Reflection

This script demonstrates how to use the GEPA (Genetic-Pareto) optimizer in DSPy 3.x.
GEPA is an advanced optimization technique that:
- Uses genetic algorithms to explore the prompt space
- Applies Pareto optimization for multi-objective improvement
- Incorporates reflective improvement for better prompts

What You'll Learn:
- How to use the GEPA optimizer
- How GEPA differs from other optimizers
- How to configure GEPA parameters
- When to use GEPA vs other optimizers
- How to evaluate and compare optimized models
"""

from dotenv import load_dotenv
from utils import setup_default_lm, configure_dspy, print_step, print_result, print_error
from utils.datasets import get_sample_qa_data
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating GEPA optimizer."""
    print("=== GEPA Optimizer Tutorial ===")
    print("Advanced optimization with genetic algorithms and Pareto optimization")
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

    # ========== PART 1: Understanding GEPA ==========
    print_step(
        "Part 1: Understanding GEPA",
        "What makes GEPA different from other optimizers"
    )

    print("  GEPA (Genetic-Pareto) Optimizer combines:")
    print("  1. Genetic Algorithms: Evolve prompts through mutations and crossover")
    print("  2. Pareto Optimization: Balance multiple objectives (accuracy, efficiency)")
    print("  3. Reflective Improvement: Self-improve prompts based on failures")
    print()
    print("  Comparison with other optimizers:")
    print()
    print("  BootstrapFewShot:")
    print("  - Simple and fast")
    print("  - Good for basic improvements")
    print("  - Limited exploration of prompt space")
    print()
    print("  MIPROv2:")
    print("  - Bayesian optimization")
    print("  - Efficient hyperparameter tuning")
    print("  - Good balance of speed and quality")
    print()
    print("  GEPA:")
    print("  - Most sophisticated approach")
    print("  - Best for complex tasks")
    print("  - Multi-objective optimization")
    print("  - Longer optimization time but better results")
    print("  - Self-reflective improvement")

    # ========== PART 2: Basic GEPA Example ==========
    print_step(
        "Part 2: Basic GEPA Optimization",
        "Using GEPA to optimize a question-answering system"
    )

    # Define a simple QA signature
    class QuestionAnswering(dspy.Signature):
        """Answer questions accurately and concisely."""
        question = dspy.InputField(desc="A question to answer")
        answer = dspy.OutputField(desc="A concise and accurate answer")

    # Create a basic QA module
    class QASystem(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.ChainOfThought(QuestionAnswering)

        def forward(self, question):
            """Answer a question."""
            result = self.qa(question=question)
            return dspy.Prediction(
                answer=result.answer,
                reasoning=result.reasoning
            )

    # Create training and validation data
    print("\n  Loading sample QA dataset...")
    all_data = get_sample_qa_data()

    # Split into train and validation
    train_data = all_data[:8]  # Use 8 for training
    val_data = all_data[8:]    # Use rest for validation

    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")

    # Define evaluation metric
    def qa_metric(example, prediction, trace=None):
        """
        Evaluate QA predictions.
        Returns a score between 0 and 1.
        """
        # Simple metric: check if answer is present and reasonable
        pred_answer = prediction.answer.lower().strip()
        true_answer = example.answer.lower().strip()

        # Exact match gets full score
        if pred_answer == true_answer:
            return 1.0

        # Partial match (answer contains key words)
        true_words = set(true_answer.split())
        pred_words = set(pred_answer.split())
        overlap = len(true_words & pred_words)

        if overlap > 0:
            return overlap / max(len(true_words), len(pred_words))

        return 0.0

    # Test baseline performance
    print("\n  Testing baseline performance...")
    baseline_system = QASystem()

    baseline_scores = []
    for example in val_data[:3]:  # Test on first 3 for speed
        try:
            prediction = baseline_system(question=example.question)
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

    # ========== PART 3: GEPA Optimization ==========
    print_step(
        "Part 3: Running GEPA Optimization",
        "Optimizing the QA system with GEPA"
    )

    print("  Note: GEPA optimization can take several minutes.")
    print("  For demonstration, we'll use small parameters.")
    print()

    try:
        # Create GEPA optimizer
        # Key parameters:
        # - population_size: Number of prompt variants in each generation
        # - num_generations: How many evolutionary cycles to run
        # - metric: Evaluation function
        # - num_threads: Parallel evaluation (speeds up optimization)
        print("  Initializing GEPA optimizer...")

        gepa_optimizer = dspy.GEPA(
            metric=qa_metric,
            population_size=4,      # Small for demo; use 10-20 in practice
            num_generations=2,      # Small for demo; use 5-10 in practice
            verbose=True
        )

        print("\n  Starting optimization (this may take a few minutes)...")
        print("  GEPA will:")
        print("  1. Create initial population of prompt variants")
        print("  2. Evaluate each variant on training data")
        print("  3. Apply genetic operations (mutation, crossover)")
        print("  4. Reflect on failures and improve")
        print("  5. Repeat for specified generations")
        print()

        # Compile (optimize) the module
        # Note: This is commented out as it requires API calls
        # Uncomment to actually run optimization
        """
        optimized_system = gepa_optimizer.compile(
            student=QASystem(),
            trainset=train_data
        )

        print("\n  Optimization complete!")
        print("  Testing optimized system...")

        optimized_scores = []
        for example in val_data[:3]:
            try:
                prediction = optimized_system(question=example.question)
                score = qa_metric(example, prediction)
                optimized_scores.append(score)
                print(f"    Q: {example.question[:60]}...")
                print(f"    Expected: {example.answer}")
                print(f"    Got: {prediction.answer}")
                print(f"    Score: {score:.2f}\n")
            except Exception as e:
                print(f"    Error: {e}")
                optimized_scores.append(0.0)

        avg_optimized = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
        print(f"  Optimized average score: {avg_optimized:.2f}")
        print(f"  Improvement: {(avg_optimized - avg_baseline):.2f}")
        """

        print("  [Optimization code is commented out for demo purposes]")
        print("  In practice, GEPA typically improves scores by 10-30%")

    except AttributeError as e:
        print_error(f"GEPA optimizer may not be available: {e}")
        print("  This is expected if using an older version of DSPy")

    # ========== PART 4: GEPA Configuration ==========
    print_step(
        "Part 4: GEPA Configuration",
        "Understanding and tuning GEPA parameters"
    )

    print("  Key GEPA Parameters:")
    print()
    print("  1. population_size (default: 10)")
    print("     - Number of prompt variants per generation")
    print("     - Larger = more exploration, slower optimization")
    print("     - Recommended: 10-20 for most tasks")
    print()
    print("  2. num_generations (default: 5)")
    print("     - Number of evolutionary cycles")
    print("     - More generations = better results, longer time")
    print("     - Recommended: 5-10 for most tasks")
    print()
    print("  3. mutation_rate (default: 0.1)")
    print("     - Probability of mutating each prompt component")
    print("     - Higher = more exploration, less stability")
    print("     - Recommended: 0.1-0.3")
    print()
    print("  4. crossover_rate (default: 0.5)")
    print("     - Probability of combining two prompts")
    print("     - Higher = more combination, potential for innovation")
    print("     - Recommended: 0.5-0.7")
    print()
    print("  5. reflection_enabled (default: True)")
    print("     - Whether to use reflective improvement")
    print("     - Analyzes failures and adjusts prompts")
    print("     - Recommended: Always True")
    print()
    print("  6. num_threads (default: 1)")
    print("     - Parallel evaluation threads")
    print("     - Higher = faster optimization (if API allows)")
    print("     - Recommended: 4-8 for production")

    # ========== PART 5: When to Use GEPA ==========
    print_step(
        "Part 5: When to Use GEPA",
        "Choosing the right optimizer for your task"
    )

    print("  Use GEPA when:")
    print("  ✓ You have a complex task with multiple objectives")
    print("  ✓ You want the best possible performance")
    print("  ✓ You have time for longer optimization")
    print("  ✓ You have sufficient training data (50+ examples)")
    print("  ✓ Your metric can evaluate multiple dimensions")
    print()
    print("  Use BootstrapFewShot when:")
    print("  ✓ You want quick results")
    print("  ✓ Your task is relatively simple")
    print("  ✓ You have limited training data (<20 examples)")
    print("  ✓ You're prototyping and iterating quickly")
    print()
    print("  Use MIPROv2 when:")
    print("  ✓ You want good balance of speed and quality")
    print("  ✓ You need hyperparameter tuning")
    print("  ✓ You have moderate training data (20-50 examples)")
    print("  ✓ You want Bayesian optimization")
    print()
    print("  Use SIMBA when:")
    print("  ✓ You want self-reflective improvement")
    print("  ✓ You have examples of failures to learn from")
    print("  ✓ Your task benefits from iterative refinement")

    # ========== PART 6: Practical Example Template ==========
    print_step(
        "Part 6: Complete GEPA Optimization Template",
        "Production-ready code structure"
    )

    print("  Complete template for GEPA optimization:")
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
    train_data = [...]  # 50-100+ examples
    val_data = [...]    # 20-30 examples

    # 3. Define metric
    def my_metric(example, prediction, trace=None):
        # Multi-objective scoring
        accuracy = ...  # 0-1 score
        efficiency = ...  # 0-1 score
        return (accuracy + efficiency) / 2

    # 4. Configure GEPA
    optimizer = dspy.GEPA(
        metric=my_metric,
        population_size=15,
        num_generations=7,
        mutation_rate=0.2,
        crossover_rate=0.6,
        reflection_enabled=True,
        num_threads=4,
        verbose=True
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
    optimized.save("models/my_optimized_model")
    """)

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. GEPA combines genetic algorithms with Pareto optimization")
    print("2. Best for complex tasks requiring sophisticated optimization")
    print("3. Key parameters: population_size, num_generations, mutation_rate")
    print("4. Reflection enables self-improvement from failures")
    print("5. Longer optimization time but better final results")
    print("6. Use multi-objective metrics for best results")
    print("7. Requires more training data than simpler optimizers")


if __name__ == "__main__":
    main()
