#!/usr/bin/env python3
"""
Classification Finetuning with DSPy

This script demonstrates how to finetune classification models using DSPy optimizers
for improved accuracy and performance on specific tasks.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import setup_default_lm, print_step, print_result, print_error
from utils.datasets import get_sample_classification_data
from dotenv import load_dotenv
import random

@dataclass
class ClassificationResult:
    predicted_class: str
    confidence: float
    reasoning: str

def main():
    """Main function demonstrating classification finetuning with DSPy."""
    print("=" * 70)
    print("CLASSIFICATION FINETUNING WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for classification finetuning")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=1000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Classification Signatures
    class SentimentClassification(dspy.Signature):
        """Classify the sentiment of text as positive, negative, or neutral."""
        text = dspy.InputField(desc="The text to classify")
        sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")
        confidence = dspy.OutputField(desc="Confidence score (0-100)")
        reasoning = dspy.OutputField(desc="Brief explanation for the classification")
    
    class TopicClassification(dspy.Signature):
        """Classify text into predefined topic categories."""
        text = dspy.InputField(desc="The text to classify")
        categories = dspy.InputField(desc="Available categories for classification")
        topic = dspy.OutputField(desc="The most appropriate topic category")
        confidence = dspy.OutputField(desc="Confidence score (0-100)")
        secondary_topics = dspy.OutputField(desc="Other relevant topics if any")
    
    class IntentClassification(dspy.Signature):
        """Classify user intent from text input."""
        user_input = dspy.InputField(desc="User's input text")
        possible_intents = dspy.InputField(desc="List of possible user intents")
        intent = dspy.OutputField(desc="The classified user intent")
        confidence = dspy.OutputField(desc="Confidence score (0-100)")
        extracted_entities = dspy.OutputField(desc="Any entities extracted from the input")
    
    # Data Generation Functions
    def generate_sentiment_data() -> List[dspy.Example]:
        """Generate comprehensive sentiment classification dataset."""
        sentiment_examples = [
            # Positive examples
            ("I absolutely love this product! It exceeded all my expectations.", "positive"),
            ("Amazing service and friendly staff. Highly recommend!", "positive"),
            ("This is the best purchase I've made in years.", "positive"),
            ("Fantastic quality and fast delivery. Very satisfied!", "positive"),
            ("Brilliant design and excellent functionality.", "positive"),
            
            # Negative examples
            ("This is terrible. Complete waste of money.", "negative"),
            ("Worst customer service I've ever experienced.", "negative"),
            ("The product broke after just one day. Very disappointed.", "negative"),
            ("Overpriced and poor quality. Would not recommend.", "negative"),
            ("Frustrated with the complicated setup process.", "negative"),
            
            # Neutral examples
            ("The product works as described. Nothing special.", "neutral"),
            ("It's okay, does what it's supposed to do.", "neutral"),
            ("Average quality for the price point.", "neutral"),
            ("Standard service, no complaints but nothing outstanding.", "neutral"),
            ("Decent product, meets basic requirements.", "neutral"),
            
            # Complex/nuanced examples
            ("The product has some great features but the price is too high.", "neutral"),
            ("Love the design but wish it was more durable.", "neutral"),
            ("Good customer service helped resolve my initial disappointment.", "positive"),
            ("Initially frustrated but the replacement worked perfectly.", "positive"),
            ("Mixed feelings - some aspects are great, others not so much.", "neutral"),
        ]
        
        return [dspy.Example(text=text, sentiment=sentiment) for text, sentiment in sentiment_examples]
    
    def generate_topic_data() -> List[dspy.Example]:
        """Generate topic classification dataset."""
        topic_examples = [
            # Technology
            ("The new iPhone features advanced AI capabilities and improved battery life.", "technology"),
            ("Machine learning algorithms are revolutionizing data analysis.", "technology"),
            ("Cloud computing offers scalable solutions for businesses.", "technology"),
            
            # Health
            ("Regular exercise and a balanced diet are essential for good health.", "health"),
            ("The new vaccine shows promising results in clinical trials.", "health"),
            ("Mental health awareness has increased significantly in recent years.", "health"),
            
            # Business
            ("Quarterly earnings exceeded analyst expectations by 15%.", "business"),
            ("The merger will create synergies and reduce operational costs.", "business"),
            ("Market volatility affects investor confidence and trading volumes.", "business"),
            
            # Sports
            ("The championship game was decided in overtime with a spectacular goal.", "sports"),
            ("The athlete's training regimen includes both strength and endurance work.", "sports"),
            ("Team management announced several trades before the deadline.", "sports"),
            
            # Entertainment
            ("The movie received critical acclaim for its cinematography and acting.", "entertainment"),
            ("The concert tour sold out within minutes of tickets going on sale.", "entertainment"),
            ("Streaming platforms are investing heavily in original content.", "entertainment"),
        ]
        
        return [dspy.Example(text=text, topic=topic) for text, topic in topic_examples]
    
    def generate_intent_data() -> List[dspy.Example]:
        """Generate intent classification dataset."""
        intent_examples = [
            # Booking intents
            ("I want to book a flight to New York for next Tuesday", "book_flight"),
            ("Can you help me reserve a table for four at 7 PM?", "book_restaurant"),
            ("I need to schedule an appointment with Dr. Smith", "book_appointment"),
            
            # Information requests
            ("What's the weather like in London today?", "get_weather"),
            ("How much does the premium subscription cost?", "get_pricing"),
            ("What are your business hours?", "get_info"),
            
            # Support requests
            ("I'm having trouble logging into my account", "technical_support"),
            ("My order hasn't arrived yet. Can you check the status?", "order_status"),
            ("I need to cancel my subscription", "cancel_service"),
            
            # General inquiries
            ("Tell me about your return policy", "policy_inquiry"),
            ("Do you offer student discounts?", "discount_inquiry"),
            ("I want to speak with a manager", "escalate_request"),
        ]
        
        return [dspy.Example(user_input=text, intent=intent) for text, intent in intent_examples]
    
    # Classification Modules
    class SentimentClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.classify = dspy.ChainOfThought(SentimentClassification)
        
        def forward(self, text):
            result = self.classify(text=text)
            return dspy.Prediction(
                sentiment=result.sentiment,
                confidence=result.confidence,
                reasoning=result.reasoning
            )
    
    class TopicClassifier(dspy.Module):
        def __init__(self, categories: List[str]):
            super().__init__()
            self.categories = categories
            self.classify = dspy.ChainOfThought(TopicClassification)
        
        def forward(self, text):
            categories_str = ", ".join(self.categories)
            result = self.classify(text=text, categories=categories_str)
            return dspy.Prediction(
                topic=result.topic,
                confidence=result.confidence,
                secondary_topics=result.secondary_topics
            )
    
    class IntentClassifier(dspy.Module):
        def __init__(self, intents: List[str]):
            super().__init__()
            self.intents = intents
            self.classify = dspy.ChainOfThought(IntentClassification)
        
        def forward(self, user_input):
            intents_str = ", ".join(self.intents)
            result = self.classify(user_input=user_input, possible_intents=intents_str)
            return dspy.Prediction(
                intent=result.intent,
                confidence=result.confidence,
                extracted_entities=result.extracted_entities
            )
    
    # Evaluation Functions
    def evaluate_classifier(classifier, test_data: List[dspy.Example], task_type: str) -> Dict[str, Any]:
        """Evaluate classifier performance."""
        predictions = []
        ground_truth = []
        
        for example in test_data:
            try:
                if task_type == "sentiment":
                    pred = classifier(text=example.text)
                    predictions.append(pred.sentiment.lower().strip())
                    ground_truth.append(example.sentiment.lower().strip())
                elif task_type == "topic":
                    pred = classifier(text=example.text)
                    predictions.append(pred.topic.lower().strip())
                    ground_truth.append(example.topic.lower().strip())
                elif task_type == "intent":
                    pred = classifier(user_input=example.user_input)
                    predictions.append(pred.intent.lower().strip())
                    ground_truth.append(example.intent.lower().strip())
            except Exception as e:
                print_error(f"Error evaluating example: {e}")
                continue
        
        if not predictions:
            return {"accuracy": 0, "error": "No valid predictions"}
        
        accuracy = accuracy_score(ground_truth, predictions)
        
        return {
            "accuracy": accuracy,
            "num_examples": len(predictions),
            "predictions": predictions[:5],  # Sample predictions
            "ground_truth": ground_truth[:5]  # Sample ground truth
        }
    
    # Demo 1: Sentiment Classification Training
    print_step("Sentiment Classification", "Training and evaluating sentiment classifier")
    
    # Generate training data
    sentiment_data = generate_sentiment_data()
    train_sentiment, test_sentiment = train_test_split(sentiment_data, test_size=0.3, random_state=42)
    
    print_result(f"Training examples: {len(train_sentiment)}, Test examples: {len(test_sentiment)}")
    
    # Initialize classifier
    sentiment_classifier = SentimentClassifier()
    
    # Evaluate before optimization
    initial_performance = evaluate_classifier(sentiment_classifier, test_sentiment, "sentiment")
    print_result(f"Initial Accuracy: {initial_performance['accuracy']:.2%}")
    
    # Demonstrate few-shot optimization
    try:
        # Use BootstrapFewShot for optimization
        from dspy.teleprompt import BootstrapFewShot
        
        # Define evaluation metric
        def sentiment_accuracy(example, pred, trace=None):
            return example.sentiment.lower().strip() == pred.sentiment.lower().strip()
        
        # Optimize with few-shot examples
        optimizer = BootstrapFewShot(metric=sentiment_accuracy, max_bootstrapped_demos=5)
        optimized_classifier = optimizer.compile(sentiment_classifier, trainset=train_sentiment)
        
        # Evaluate optimized classifier
        optimized_performance = evaluate_classifier(optimized_classifier, test_sentiment, "sentiment")
        print_result(f"Optimized Accuracy: {optimized_performance['accuracy']:.2%}")
        print_result(f"Improvement: {(optimized_performance['accuracy'] - initial_performance['accuracy']):.2%}")
        
    except Exception as e:
        print_error(f"Optimization failed: {e}")
        optimized_classifier = sentiment_classifier
    
    # Test with new examples
    test_sentences = [
        "This product is absolutely fantastic! I couldn't be happier.",
        "The service was disappointing and the staff was rude.",
        "It's an average product, nothing particularly special about it.",
        "Great value for money, though it could use some improvements."
    ]
    
    print("\n--- Sentiment Classification Results ---")
    for i, sentence in enumerate(test_sentences, 1):
        try:
            result = optimized_classifier(text=sentence)
            print(f"\nTest {i}: {sentence}")
            print_result(f"Sentiment: {result.sentiment}", "Prediction")
            print_result(f"Confidence: {result.confidence}", "Confidence")
            print_result(f"Reasoning: {result.reasoning}", "Reasoning")
        except Exception as e:
            print_error(f"Error classifying sentence {i}: {e}")
    
    # Demo 2: Topic Classification
    print_step("Topic Classification", "Training topic classifier with multiple categories")
    
    topic_categories = ["technology", "health", "business", "sports", "entertainment"]
    topic_data = generate_topic_data()
    train_topic, test_topic = train_test_split(topic_data, test_size=0.3, random_state=42)
    
    topic_classifier = TopicClassifier(topic_categories)
    
    # Evaluate topic classifier
    topic_performance = evaluate_classifier(topic_classifier, test_topic, "topic")
    print_result(f"Topic Classification Accuracy: {topic_performance['accuracy']:.2%}")
    
    # Test with new examples
    test_topics = [
        "The startup raised $50 million in Series B funding to expand operations.",
        "Scientists discovered a new treatment for Alzheimer's disease.",
        "The basketball team signed a new star player for the upcoming season.",
        "The latest smartphone features revolutionary camera technology.",
        "The streaming service announced several new original series."
    ]
    
    print("\n--- Topic Classification Results ---")
    for i, text in enumerate(test_topics, 1):
        try:
            result = topic_classifier(text=text)
            print(f"\nTest {i}: {text}")
            print_result(f"Topic: {result.topic}", "Primary Topic")
            print_result(f"Confidence: {result.confidence}", "Confidence")
            print_result(f"Secondary Topics: {result.secondary_topics}", "Secondary Topics")
        except Exception as e:
            print_error(f"Error classifying topic {i}: {e}")
    
    # Demo 3: Intent Classification
    print_step("Intent Classification", "Training intent classifier for customer service")
    
    intent_categories = [
        "book_flight", "book_restaurant", "book_appointment",
        "get_weather", "get_pricing", "get_info",
        "technical_support", "order_status", "cancel_service",
        "policy_inquiry", "discount_inquiry", "escalate_request"
    ]
    
    intent_data = generate_intent_data()
    train_intent, test_intent = train_test_split(intent_data, test_size=0.3, random_state=42)
    
    intent_classifier = IntentClassifier(intent_categories)
    
    # Evaluate intent classifier
    intent_performance = evaluate_classifier(intent_classifier, test_intent, "intent")
    print_result(f"Intent Classification Accuracy: {intent_performance['accuracy']:.2%}")
    
    # Test with new examples
    test_intents = [
        "I'd like to make a reservation for tomorrow evening",
        "Can you tell me what the current temperature is?",
        "My package tracking shows no updates for a week",
        "Is there a student discount available for this course?",
        "I can't access my account after the password reset"
    ]
    
    print("\n--- Intent Classification Results ---")
    for i, text in enumerate(test_intents, 1):
        try:
            result = intent_classifier(user_input=text)
            print(f"\nTest {i}: {text}")
            print_result(f"Intent: {result.intent}", "Detected Intent")
            print_result(f"Confidence: {result.confidence}", "Confidence")
            print_result(f"Entities: {result.extracted_entities}", "Extracted Entities")
        except Exception as e:
            print_error(f"Error classifying intent {i}: {e}")
    
    # Demo 4: Multi-label Classification
    print_step("Multi-label Classification", "Handling texts with multiple categories")
    
    class MultiLabelClassification(dspy.Signature):
        """Classify text into multiple relevant categories."""
        text = dspy.InputField(desc="The text to classify")
        available_labels = dspy.InputField(desc="Available classification labels")
        
        primary_label = dspy.OutputField(desc="The most relevant label")
        secondary_labels = dspy.OutputField(desc="Other relevant labels (comma-separated)")
        confidence_scores = dspy.OutputField(desc="Confidence for each assigned label")
        explanation = dspy.OutputField(desc="Explanation for the classification choices")
    
    class MultiLabelClassifier(dspy.Module):
        def __init__(self, labels: List[str]):
            super().__init__()
            self.labels = labels
            self.classify = dspy.ChainOfThought(MultiLabelClassification)
        
        def forward(self, text):
            labels_str = ", ".join(self.labels)
            result = self.classify(text=text, available_labels=labels_str)
            return result
    
    # Multi-label examples
    multi_labels = ["technology", "business", "innovation", "healthcare", "education", "environment"]
    multi_classifier = MultiLabelClassifier(multi_labels)
    
    multi_test_cases = [
        "The EdTech startup uses AI to personalize learning experiences for students with disabilities.",
        "Green technology investments are driving sustainable business growth in renewable energy sectors.",
        "Telemedicine platforms are revolutionizing healthcare delivery while reducing operational costs.",
        "Universities are adopting blockchain technology for secure credential verification systems."
    ]
    
    print("\n--- Multi-label Classification Results ---")
    for i, text in enumerate(multi_test_cases, 1):
        try:
            result = multi_classifier(text=text)
            print(f"\nTest {i}: {text}")
            print_result(f"Primary: {result.primary_label}", "Primary Label")
            print_result(f"Secondary: {result.secondary_labels}", "Secondary Labels")
            print_result(f"Confidence: {result.confidence_scores}", "Confidence Scores")
            print_result(f"Explanation: {result.explanation}", "Explanation")
        except Exception as e:
            print_error(f"Error in multi-label classification {i}: {e}")
    
    # Demo 5: Performance Analysis
    print_step("Performance Analysis", "Analyzing classification performance across tasks")
    
    performance_summary = {
        "Sentiment Classification": {
            "accuracy": initial_performance.get("accuracy", 0),
            "optimized_accuracy": optimized_performance.get("accuracy", 0),
            "improvement": optimized_performance.get("accuracy", 0) - initial_performance.get("accuracy", 0)
        },
        "Topic Classification": {
            "accuracy": topic_performance.get("accuracy", 0),
            "categories": len(topic_categories)
        },
        "Intent Classification": {
            "accuracy": intent_performance.get("accuracy", 0),
            "intents": len(intent_categories)
        }
    }
    
    print("\n--- Performance Summary ---")
    for task, metrics in performance_summary.items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value}")
    
    # Demo 6: Error Analysis
    print_step("Error Analysis", "Analyzing classification errors and patterns")
    
    def analyze_errors(predictions: List[str], ground_truth: List[str], examples: List[str]) -> Dict[str, Any]:
        """Analyze classification errors."""
        errors = []
        for i, (pred, true, example) in enumerate(zip(predictions, ground_truth, examples)):
            if pred != true:
                errors.append({
                    "index": i,
                    "predicted": pred,
                    "actual": true,
                    "text": example[:100] + "..." if len(example) > 100 else example
                })
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(predictions) if predictions else 0,
            "sample_errors": errors[:3]  # Show first 3 errors
        }
    
    # Analyze sentiment classification errors
    if optimized_performance.get("predictions") and optimized_performance.get("ground_truth"):
        sentiment_texts = [ex.text for ex in test_sentiment[:len(optimized_performance["predictions"])]]
        error_analysis = analyze_errors(
            optimized_performance["predictions"],
            optimized_performance["ground_truth"],
            sentiment_texts
        )
        
        print_result(f"Error Rate: {error_analysis['error_rate']:.2%}", "Sentiment Errors")
        print_result(f"Total Errors: {error_analysis['total_errors']}", "Error Count")
        
        if error_analysis["sample_errors"]:
            print("\nSample Errors:")
            for error in error_analysis["sample_errors"]:
                print(f"  Text: {error['text']}")
                print(f"  Predicted: {error['predicted']}, Actual: {error['actual']}")
    
    print("\n" + "="*70)
    print("CLASSIFICATION FINETUNING COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy classification finetuning techniques!")

if __name__ == "__main__":
    main()
