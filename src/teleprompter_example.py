import dspy
from dspy.datasets import Dataset
from dspy.teleprompt import BootstrapFewShot
from typing import List, Dict, Any


class Classifier(dspy.Module):
    """A text classifier with self-improving capabilities"""

    def __init__(self, labels: List[str] = None):
        super().__init__()
        self.labels = labels or ["positive", "negative", "neutral"]
        self.classify = dspy.ChainOfThought(
            "text -> reasoning, label, confidence")

    def forward(self, text: str) -> Dict[str, Any]:
        pred = self.classify(text=text)
        return {
            'label': pred.label,
            'confidence': float(pred.confidence),
            'reasoning': pred.reasoning
        }


class BootstrappedClassifier(dspy.Module):
    """A classifier that uses bootstrapped examples for better performance"""

    def __init__(self, num_bootstrap_examples: int = 3):
        super().__init__()
        self.num_bootstrap_examples = num_bootstrap_examples
        self.analyze = dspy.ChainOfThought("text -> key_features, complexity")
        self.classify = dspy.ChainOfThought(
            "text, key_features -> label, confidence")
        self._bootstrap = BootstrapFewShot(metric="exact_match")
        self._compiled = False

    def compile_with_examples(self, examples: List[Dict[str, str]]):
        dataset = Dataset.from_list(examples)
        self._optimized_module = self._bootstrap.compile(
            self,
            trainset=dataset,
            max_bootstrapped_demos=self.num_bootstrap_examples,
            max_labeled_demos=len(examples)
        )
        self._compiled = True

    def forward(self, text: str) -> Dict[str, Any]:
        if not self._compiled:
            raise RuntimeError("Model must be compiled with examples first")

        # Analyze text features
        analysis = self.analyze(text=text)

        # Use optimized classification
        result = self.classify(
            text=text,
            key_features=analysis.key_features
        )

        return {
            'label': result.label,
            'confidence': float(result.confidence),
            'complexity': float(analysis.complexity)
        }


def create_training_data() -> Dataset:
    """Create example dataset for sentiment analysis"""
    examples = [
        {
            "text": "This movie was fantastic!",
            "label": "positive"
        },
        {
            "text": "Worst experience ever.",
            "label": "negative"
        },
        {
            "text": "The food was okay, nothing special.",
            "label": "neutral"
        },
        {
            "text": "I absolutely love this product!",
            "label": "positive"
        },
        {
            "text": "Service was terrible and staff was rude.",
            "label": "negative"
        }
    ]
    return Dataset.from_list(examples)


class AdaptiveTelepromptModule(dspy.Module):
    """A module that adapts its prompting strategy based on past performance"""

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(
            "input -> complexity, required_approach")
        self.generate_examples = dspy.ChainOfThought(
            "input, approach -> relevant_examples")
        self.process = dspy.ChainOfThought(
            "input, examples -> output, confidence")
        self._history: List[Dict] = []

    def forward(self, input_text: str, actual: str = None) -> Dict[str, Any]:
        # Assess input and determine approach
        assessment = self.assess(input=input_text)

        # Generate relevant examples based on history
        examples = self.generate_examples(
            input=input_text,
            approach=assessment.required_approach
        )

        # Process with examples
        result = self.process(
            input=input_text,
            examples=examples.relevant_examples
        )

        # Update history if actual result is provided
        if actual:
            self._history.append({
                'input': input_text,
                'prediction': result.output,
                'actual': actual,
                'approach': assessment.required_approach
            })

        return {
            'output': result.output,
            'confidence': float(result.confidence),
            'approach': assessment.required_approach
        }


def bootstrap_classifier(
    examples: List[Dict[str, str]] = None,
    num_examples: int = 3
) -> BootstrappedClassifier:
    """Create and compile a bootstrapped classifier"""
    classifier = BootstrappedClassifier(num_bootstrap_examples=num_examples)
    examples = examples or create_training_data()
    classifier.compile_with_examples(examples)
    return classifier
