import dspy
from typing import List, Dict, Any, Optional
from dspy.teleprompt import BootstrapFewShot


class SelfImprovingModule(dspy.Module):
    """A module that learns from its own mistakes and improves over time"""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("input -> prediction, confidence")
        self.evaluate = dspy.ChainOfThought(
            "prediction, actual -> score, feedback")
        self.improve = dspy.ChainOfThought(
            "feedback, previous_examples -> improved_strategy")
        self._history: List[Dict] = []

    def forward(self, input_text: str, actual: Optional[str] = None) -> Dict[str, Any]:
        # Make prediction
        prediction = self.predict(input=input_text)
        result = {
            'prediction': prediction.prediction,
            'confidence': prediction.confidence
        }

        # If actual result provided, learn from it
        if actual:
            evaluation = self.evaluate(
                prediction=prediction.prediction,
                actual=actual
            )

            # Store example for future learning
            self._history.append({
                'input': input_text,
                'prediction': prediction.prediction,
                'actual': actual,
                'feedback': evaluation.feedback
            })

            # If we have enough examples, try to improve
            if len(self._history) >= 3:
                improvement = self.improve(
                    feedback=evaluation.feedback,
                    previous_examples=self._history[-3:]
                )
                result['improvement'] = improvement.improved_strategy

        return result


class OptimizedChainOfThought(dspy.Module):
    """A module that uses bootstrapped optimization for better reasoning"""

    def __init__(self, num_bootstrap_examples: int = 3):
        super().__init__()
        self.num_bootstrap_examples = num_bootstrap_examples
        self.reason = dspy.ChainOfThought(
            "input -> reasoning_steps, conclusion")
        self._bootstrap = BootstrapFewShot(metric="exact_match")

    def compile_with_examples(self, examples: List[Dict[str, str]]):
        """Compile the module with example data for optimization"""
        dataset = dspy.Example.from_list(examples)
        self._optimized_module = self._bootstrap.compile(
            self,
            trainset=dataset,
            max_bootstrapped_demos=self.num_bootstrap_examples
        )

    def forward(self, input_text: str) -> Dict[str, Any]:
        if hasattr(self, '_optimized_module'):
            module = self._optimized_module
        else:
            module = self

        result = module.reason(input=input_text)
        return {
            'reasoning': result.reasoning_steps,
            'conclusion': result.conclusion
        }


class AdaptivePrompting(dspy.Module):
    """A module that adapts its prompting strategy based on input complexity"""

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(
            "input -> complexity, required_steps")
        self.simple_process = dspy.Predict("input -> output")
        self.complex_process = dspy.ChainOfThought(
            "input, steps -> output, reasoning")

    def forward(self, input_text: str) -> Dict[str, Any]:
        # Assess input complexity
        assessment = self.assess(input=input_text)
        complexity = float(assessment.complexity)

        if complexity < 0.5:
            # Use simple processing for straightforward inputs
            result = self.simple_process(input=input_text)
            return {
                'output': result.output,
                'complexity': complexity,
                'method': 'simple'
            }
        else:
            # Use detailed reasoning for complex inputs
            result = self.complex_process(
                input=input_text,
                steps=assessment.required_steps
            )
            return {
                'output': result.output,
                'reasoning': result.reasoning,
                'complexity': complexity,
                'method': 'complex'
            }
