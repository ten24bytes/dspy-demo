#!/usr/bin/env python3
"""
Getting Started with DSPy - Python Script Version

This script demonstrates the fundamental concepts of DSPy:
- Setting up language models
- Creating signatures
- Using basic modules
- Making predictions
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy basics."""
    print("🚀 Getting Started with DSPy")
    print("=" * 50)

    # Load environment variables
    load_dotenv('.env')

    # Setup Language Model
    print_step("Setting up Language Model", "Configuring DSPy with OpenAI gpt-4o")

    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=500)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
        return

    # Create Signatures
    print_step("Creating DSPy Signatures", "Defining input/output specifications")

    class QuestionAnswering(dspy.Signature):
        """Answer the given question with a concise and accurate response."""
        question = dspy.InputField(desc="The question to be answered")
        answer = dspy.OutputField(desc="A concise answer to the question")

    class SentimentClassification(dspy.Signature):
        """Classify the sentiment of the given text as positive, negative, or neutral."""
        text = dspy.InputField(desc="The text to classify")
        sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")

    print_result("Signatures created successfully!")

    # Basic Prediction
    print_step("Using Predict Module", "Making basic predictions with our signatures")

    qa_predictor = dspy.Predict(QuestionAnswering)
    sentiment_predictor = dspy.Predict(SentimentClassification)

    # Test question answering
    question = "What is the capital of France?"
    qa_result = qa_predictor(question=question)
    print_result(f"Question: {question}\\nAnswer: {qa_result.answer}", "Question Answering")

    # Test sentiment classification
    text = "I absolutely love this new product! It's fantastic!"
    sentiment_result = sentiment_predictor(text=text)
    print_result(f"Text: {text}\\nSentiment: {sentiment_result.sentiment}", "Sentiment Classification")

    # Chain of Thought Reasoning
    print_step("Using ChainOfThought Module", "Adding reasoning steps to predictions")

    class MathReasoning(dspy.Signature):
        """Solve the mathematical problem step by step."""
        problem = dspy.InputField(desc="The mathematical problem to solve")
        reasoning = dspy.OutputField(desc="Step-by-step reasoning")
        answer = dspy.OutputField(desc="The final numerical answer")

    math_cot = dspy.ChainOfThought(MathReasoning)

    problem = "If a rectangle has a length of 8 meters and a width of 5 meters, what is its area?"
    math_result = math_cot(problem=problem)

    print_result(
        f"Problem: {problem}\\n"
        f"Reasoning: {math_result.reasoning}\\n"
        f"Answer: {math_result.answer}",
        "Math Reasoning"
    )

    # Custom Module
    print_step("Creating Custom Module", "Building a comprehensive question answering system")

    class SmartQA(dspy.Module):
        def __init__(self):
            super().__init__()

            class QuestionType(dspy.Signature):
                """Classify the type of question being asked."""
                question = dspy.InputField(desc="The question to classify")
                question_type = dspy.OutputField(desc="Type: factual, mathematical, creative, or analytical")

            class AnswerQuestion(dspy.Signature):
                """Answer the question based on its type."""
                question = dspy.InputField(desc="The question to answer")
                question_type = dspy.InputField(desc="The type of question")
                answer = dspy.OutputField(desc="A comprehensive answer")

            self.classify_question = dspy.Predict(QuestionType)
            self.answer_question = dspy.ChainOfThought(AnswerQuestion)

        def forward(self, question):
            classification = self.classify_question(question=question)
            answer = self.answer_question(
                question=question,
                question_type=classification.question_type
            )

            return dspy.Prediction(
                question_type=classification.question_type,
                reasoning=answer.reasoning,
                answer=answer.answer
            )

    smart_qa = SmartQA()

    test_questions = [
        "What is the speed of light?",
        "If I have 10 apples and eat 3, how many do I have left?",
        "Write a creative story about a robot learning to paint.",
    ]

    for question in test_questions:
        result = smart_qa(question=question)
        print_result(
            f"Question: {question}\\n"
            f"Type: {result.question_type}\\n"
            f"Reasoning: {result.reasoning}\\n"
            f"Answer: {result.answer}",
            "Smart QA Result"
        )
        print("-" * 80)

    # Working with Examples
    print_step("Working with Examples", "Creating and using DSPy Example objects")

    examples = [
        dspy.Example(question="What is 2+2?", answer="4"),
        dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare"),
        dspy.Example(question="What is the largest planet?", answer="Jupiter"),
    ]

    print_result(f"Created {len(examples)} examples")

    print("Testing predictor on examples:")
    for i, example in enumerate(examples, 1):
        prediction = qa_predictor(question=example.question)
        print(f"\\nExample {i}:")
        print(f"Question: {example.question}")
        print(f"Expected: {example.answer}")
        print(f"Predicted: {prediction.answer}")
        print(f"Match: {prediction.answer.lower().strip() == example.answer.lower().strip()}")

    print("\\n🎉 DSPy basics demonstration completed!")


if __name__ == "__main__":
    main()
