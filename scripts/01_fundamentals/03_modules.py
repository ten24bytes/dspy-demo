#!/usr/bin/env python3
"""
DSPy Modules Deep Dive

This script explores DSPy modules - the building blocks for composing
language model programs.

What You'll Learn:
- How dspy.Module works (init and forward)
- Wrapping predictors in modules
- Composing modules (module calling module)
- Inspecting module parameters with named_predictors()
- Modules with conditional logic
- Saving and loading modules
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy modules."""
    print("=== DSPy Modules Deep Dive ===")
    print("Building blocks for composable LM programs")
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

    # Example 1: Simplest Module
    print_step(
        "Example 1: Simplest Module",
        "Wrapping a single Predict in a Module"
    )

    class SimpleQA(dspy.Module):
        """The simplest possible DSPy module."""
        def __init__(self):
            super().__init__()

            class QASignature(dspy.Signature):
                """Answer the question concisely."""
                question = dspy.InputField()
                answer = dspy.OutputField()

            self.predict = dspy.Predict(QASignature)

        def forward(self, question):
            return self.predict(question=question)

    qa = SimpleQA()
    result = qa(question="What is the speed of light?")
    print_result(f"Answer: {result.answer}", "Simple Module")

    # Example 2: Module with Multiple Predictors
    print_step(
        "Example 2: Multiple Predictors",
        "A module that chains two LM calls"
    )

    class TranslateAndSummarize(dspy.Module):
        """Translate text to English, then summarize it."""
        def __init__(self):
            super().__init__()

            class Translate(dspy.Signature):
                """Translate the given text to English."""
                text = dspy.InputField(desc="Text in any language")
                english_text = dspy.OutputField(desc="English translation")

            class Summarize(dspy.Signature):
                """Summarize the text in one sentence."""
                text = dspy.InputField()
                summary = dspy.OutputField(desc="One-sentence summary")

            self.translator = dspy.Predict(Translate)
            self.summarizer = dspy.Predict(Summarize)

        def forward(self, text):
            translated = self.translator(text=text)
            summarized = self.summarizer(text=translated.english_text)
            return dspy.Prediction(
                english_text=translated.english_text,
                summary=summarized.summary
            )

    pipeline = TranslateAndSummarize()
    result = pipeline(text="La inteligencia artificial está transformando la industria tecnológica a nivel mundial.")

    print_result(
        f"Translation: {result.english_text}\nSummary: {result.summary}",
        "Translate + Summarize"
    )

    # Example 3: Module Composition
    print_step(
        "Example 3: Module Composition",
        "Modules that use other modules"
    )

    class Classifier(dspy.Module):
        """Classify text into a category."""
        def __init__(self):
            super().__init__()

            class ClassifyText(dspy.Signature):
                """Classify the text into one of: technology, science, sports, politics, entertainment."""
                text = dspy.InputField()
                category = dspy.OutputField(desc="One of: technology, science, sports, politics, entertainment")

            self.classify = dspy.Predict(ClassifyText)

        def forward(self, text):
            return self.classify(text=text)

    class CategoryExpert(dspy.Module):
        """Generate an expert analysis based on text and its category."""
        def __init__(self):
            super().__init__()
            self.classifier = Classifier()  # Reuse another module

            class ExpertAnalysis(dspy.Signature):
                """Provide expert analysis of the text from the perspective of the given category."""
                text = dspy.InputField()
                category = dspy.InputField()
                analysis = dspy.OutputField(desc="Expert analysis from the category perspective")

            self.analyzer = dspy.ChainOfThought(ExpertAnalysis)

        def forward(self, text):
            classification = self.classifier(text=text)
            analysis = self.analyzer(text=text, category=classification.category)
            return dspy.Prediction(
                category=classification.category,
                analysis=analysis.analysis,
                reasoning=analysis.reasoning
            )

    expert = CategoryExpert()
    result = expert(text="SpaceX successfully launched its Starship rocket into orbit for the first time.")

    print_result(
        f"Category: {result.category}\nAnalysis: {result.analysis}",
        "Composed Modules"
    )

    # Example 4: Inspecting Module Parameters
    print_step(
        "Example 4: Inspecting Modules",
        "Using named_predictors() to see module structure"
    )

    print("  CategoryExpert module predictors:")
    for name, predictor in expert.named_predictors():
        print(f"    - {name}: {type(predictor).__name__}")

    print(f"\n  TranslateAndSummarize module predictors:")
    for name, predictor in pipeline.named_predictors():
        print(f"    - {name}: {type(predictor).__name__}")

    # Example 5: Conditional Logic in Modules
    print_step(
        "Example 5: Conditional Logic",
        "Modules with dynamic behavior in forward()"
    )

    class AdaptiveQA(dspy.Module):
        """Answer questions using simple or detailed mode based on complexity."""
        def __init__(self):
            super().__init__()

            class AssessComplexity(dspy.Signature):
                """Assess whether the question is simple or complex."""
                question = dspy.InputField()
                complexity = dspy.OutputField(desc="Either 'simple' or 'complex'")

            class SimpleAnswer(dspy.Signature):
                """Give a brief, direct answer."""
                question = dspy.InputField()
                answer = dspy.OutputField(desc="A brief one-line answer")

            class DetailedAnswer(dspy.Signature):
                """Give a thorough answer with explanation."""
                question = dspy.InputField()
                answer = dspy.OutputField(desc="A detailed answer with explanation")

            self.assess = dspy.Predict(AssessComplexity)
            self.simple_qa = dspy.Predict(SimpleAnswer)
            self.detailed_qa = dspy.ChainOfThought(DetailedAnswer)

        def forward(self, question):
            assessment = self.assess(question=question)

            if "complex" in assessment.complexity.lower():
                result = self.detailed_qa(question=question)
                return dspy.Prediction(
                    answer=result.answer,
                    mode="detailed",
                    reasoning=result.reasoning
                )
            else:
                result = self.simple_qa(question=question)
                return dspy.Prediction(
                    answer=result.answer,
                    mode="simple",
                    reasoning="N/A (simple mode)"
                )

    adaptive = AdaptiveQA()

    simple_q = "What color is the sky?"
    complex_q = "How does photosynthesis convert light energy into chemical energy at the molecular level?"

    for question in [simple_q, complex_q]:
        result = adaptive(question=question)
        print_result(
            f"Q: {question}\nMode: {result.mode}\nAnswer: {result.answer}",
            "Adaptive QA"
        )

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. Modules inherit from dspy.Module and implement forward()")
    print("2. Predictors (Predict, ChainOfThought) are set up in __init__()")
    print("3. Modules can compose other modules for complex pipelines")
    print("4. Use named_predictors() to inspect module structure")
    print("5. forward() can contain conditional logic for adaptive behavior")


if __name__ == "__main__":
    main()
