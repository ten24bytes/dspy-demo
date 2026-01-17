#!/usr/bin/env python3
"""
Text Classification with DSPy

This script demonstrates how to build text classification systems using DSPy.
It covers different classification approaches, optimization, and evaluation.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error

@dataclass
class Example:
    """Training/evaluation example for classification."""
    text: str
    label: str

class ClassifyText(dspy.Signature):
    """Classify the given text into one of the predefined categories."""
    
    text = dspy.InputField(desc="The text to classify")
    reasoning = dspy.OutputField(desc="Reasoning for the classification")
    category = dspy.OutputField(desc="The predicted category")

class SentimentClassifier(dspy.Signature):
    """Analyze the sentiment of the given text."""
    
    text = dspy.InputField(desc="The text to analyze")
    sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")
    confidence = dspy.OutputField(desc="Confidence score from 0 to 1")

class TopicClassifier(dspy.Signature):
    """Classify text into predefined topic categories."""
    
    text = dspy.InputField(desc="The text to classify")
    topic = dspy.OutputField(desc="The topic category: technology, sports, politics, entertainment, science")

class BasicClassifier(dspy.Module):
    """Basic text classifier using Chain of Thought."""
    
    def __init__(self, signature_class=ClassifyText):
        super().__init__()
        self.classify = dspy.ChainOfThought(signature_class)
    
    def forward(self, text: str) -> dspy.Prediction:
        return self.classify(text=text)

class MultiStepClassifier(dspy.Module):
    """Multi-step classifier with preprocessing and reasoning."""
    
    def __init__(self):
        super().__init__()
        self.preprocess = dspy.ChainOfThought("text -> cleaned_text")
        self.classify = dspy.ChainOfThought(ClassifyText)
        self.validate = dspy.ChainOfThought("text, category -> is_valid, explanation")
    
    def forward(self, text: str) -> dspy.Prediction:
        # Step 1: Preprocess text
        preprocessed = self.preprocess(text=text)
        
        # Step 2: Classify
        classification = self.classify(text=preprocessed.cleaned_text)
        
        # Step 3: Validate classification
        validation = self.validate(
            text=preprocessed.cleaned_text,
            category=classification.category
        )
        
        return dspy.Prediction(
            text=text,
            cleaned_text=preprocessed.cleaned_text,
            category=classification.category,
            reasoning=classification.reasoning,
            is_valid=validation.is_valid,
            validation_explanation=validation.explanation
        )

def create_sample_data() -> List[Example]:
    """Create sample classification data."""
    
    examples = [
        # Technology
        Example("The new AI model shows remarkable performance improvements", "technology"),
        Example("Quantum computing breakthrough announced by researchers", "technology"),
        Example("Latest smartphone features advanced camera technology", "technology"),
        
        # Sports
        Example("The basketball team won the championship game", "sports"),
        Example("Olympic swimmer breaks world record in freestyle", "sports"),
        Example("Football season starts with exciting matchups", "sports"),
        
        # Politics
        Example("New policy announced by government officials", "politics"),
        Example("Election results show close race between candidates", "politics"),
        Example("Congressional hearing addresses important issues", "politics"),
        
        # Entertainment
        Example("New movie breaks box office records this weekend", "entertainment"),
        Example("Popular TV series announces final season", "entertainment"),
        Example("Music festival lineup includes top artists", "entertainment"),
        
        # Science
        Example("Research reveals new insights about climate change", "science"),
        Example("Medical breakthrough offers hope for patients", "science"),
        Example("Space exploration mission discovers new phenomena", "science"),
    ]
    
    return examples

def evaluate_classifier(classifier, test_data: List[Example]) -> Dict[str, float]:
    """Evaluate classifier performance."""
    
    correct = 0
    total = len(test_data)
    predictions = []
    
    for example in test_data:
        try:
            result = classifier(text=example.text)
            predicted = result.category.lower().strip()
            actual = example.label.lower().strip()
            
            if predicted == actual:
                correct += 1
            
            predictions.append({
                'text': example.text,
                'predicted': predicted,
                'actual': actual,
                'correct': predicted == actual
            })
            
        except Exception as e:
            print(f"Error classifying text: {e}")
            predictions.append({
                'text': example.text,
                'predicted': 'error',
                'actual': example.label,
                'correct': False
            })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions
    }

def demonstrate_sentiment_analysis():
    """Demonstrate sentiment analysis."""
    
    print_step("Sentiment Analysis")
    
    sentiment_classifier = BasicClassifier(SentimentClassifier)
    
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. Worst experience ever.",
        "The weather is okay today, nothing special.",
        "Fantastic job on the presentation! Well done.",
        "I'm disappointed with the service quality."
    ]
    
    for text in test_texts:
        try:
            result = sentiment_classifier(text=text)
            print_result(f"Text: {text}")
            print_result(f"Sentiment: {result.sentiment}")
            print_result(f"Confidence: {result.confidence}")
            print("-" * 50)
        except Exception as e:
            print_error(f"Error analyzing sentiment: {e}")

def demonstrate_topic_classification():
    """Demonstrate topic classification."""
    
    print_step("Topic Classification")
    
    topic_classifier = BasicClassifier(TopicClassifier)
    
    test_texts = [
        "Scientists discover new exoplanet in nearby star system",
        "Local football team advances to playoffs after victory",
        "Government announces new environmental regulations",
        "Blockbuster movie dominates weekend box office",
        "Researchers develop breakthrough cancer treatment"
    ]
    
    for text in test_texts:
        try:
            result = topic_classifier(text=text)
            print_result(f"Text: {text}")
            print_result(f"Topic: {result.topic}")
            print("-" * 50)
        except Exception as e:
            print_error(f"Error classifying topic: {e}")

def demonstrate_optimization():
    """Demonstrate classifier optimization with BootstrapFewShot."""
    
    print_step("Classifier Optimization")
    
    # Create training data
    training_data = create_sample_data()
    random.shuffle(training_data)
    
    # Split data
    train_size = int(0.7 * len(training_data))
    train_examples = training_data[:train_size]
    test_examples = training_data[train_size:]
    
    print_result(f"Training examples: {len(train_examples)}")
    print_result(f"Test examples: {len(test_examples)}")
    
    # Convert to DSPy examples
    dspy_train = [
        dspy.Example(text=ex.text, category=ex.label).with_inputs('text')
        for ex in train_examples
    ]
    
    # Create and optimize classifier
    classifier = BasicClassifier(TopicClassifier)
    
    # Baseline evaluation
    print_step("Baseline Performance")
    baseline_results = evaluate_classifier(classifier, test_examples)
    print_result(f"Baseline Accuracy: {baseline_results['accuracy']:.2%}")
    
    # Optimize with BootstrapFewShot
    try:
        print_step("Optimizing with BootstrapFewShot")
        
        # Define metric
        def accuracy_metric(example, pred, trace=None):
            return pred.topic.lower().strip() == example.category.lower().strip()
        
        # Create optimizer
        optimizer = dspy.BootstrapFewShot(
            metric=accuracy_metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=4
        )
        
        # Optimize
        optimized_classifier = optimizer.compile(
            classifier,
            trainset=dspy_train[:8]  # Use subset for demo
        )
        
        # Evaluate optimized classifier
        print_step("Optimized Performance")
        optimized_results = evaluate_classifier(optimized_classifier, test_examples)
        print_result(f"Optimized Accuracy: {optimized_results['accuracy']:.2%}")
        
        # Show improvement
        improvement = optimized_results['accuracy'] - baseline_results['accuracy']
        print_result(f"Improvement: {improvement:+.2%}")
        
    except Exception as e:
        print_error(f"Error during optimization: {e}")

def main():
    """Main function demonstrating text classification."""
    
    print("=" * 60)
    print("DSPy Text Classification Demo")
    print("=" * 60)
    
    # Setup language model
    lm = setup_default_lm()
    if not lm:
        return
    
    try:
        # Basic classification
        print_step("Basic Text Classification")
        
        classifier = BasicClassifier()
        
        test_text = "The new smartphone features an advanced AI chip for better performance"
        result = classifier(text=test_text)
        
        print_result(f"Text: {test_text}")
        print_result(f"Category: {result.category}")
        print_result(f"Reasoning: {result.reasoning}")
        
        # Multi-step classification
        print_step("Multi-Step Classification")
        
        multi_step = MultiStepClassifier()
        result = multi_step(text=test_text)
        
        print_result(f"Original Text: {result.text}")
        print_result(f"Cleaned Text: {result.cleaned_text}")
        print_result(f"Category: {result.category}")
        print_result(f"Valid: {result.is_valid}")
        print_result(f"Validation: {result.validation_explanation}")
        
        # Sentiment analysis
        demonstrate_sentiment_analysis()
        
        # Topic classification
        demonstrate_topic_classification()
        
        # Optimization
        demonstrate_optimization()
        
        print_step("Classification Complete!")
        
    except Exception as e:
        print_error(f"Error in classification demo: {e}")

if __name__ == "__main__":
    main()
