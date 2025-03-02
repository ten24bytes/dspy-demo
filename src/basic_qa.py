import dspy
from typing import Dict, Any, List


class BasicQA(dspy.Module):
    """A simple question answering module"""

    def __init__(self):
        super().__init__()
        self.gen_answer = dspy.ChainOfThought("question -> reasoning, answer")

    def forward(self, question: str) -> Dict[str, str]:
        response = self.gen_answer(question=question)
        return {
            'answer': response.answer,
            'reasoning': response.reasoning
        }


class RAGBasedQA(dspy.Module):
    """Question answering with Retrieval-Augmented Generation"""

    def __init__(self, retriever):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought(
            "context, question -> reasoning, answer, relevant_passages")
        self.retriever = retriever

    def forward(self, question: str) -> Dict[str, Any]:
        # Retrieve relevant passages
        passages = self.retrieve(question).passages

        # Generate answer using the retrieved context
        response = self.generate_answer(
            context=passages,
            question=question
        )

        return {
            'answer': response.answer,
            'reasoning': response.reasoning,
            'relevant_passages': response.relevant_passages
        }


class ConversationalQA(dspy.Module):
    """A module for handling conversational question answering"""

    def __init__(self, history_size: int = 3):
        super().__init__()
        self.history_size = history_size
        self.process_history = dspy.ChainOfThought(
            "history, question -> context_needed")
        self.retrieve = dspy.Retrieve(k=2)
        self.generate = dspy.ChainOfThought(
            "history, context, question -> answer, reasoning")

    def forward(self, question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        if history is None:
            history = []

        # Keep only recent history
        recent_history = history[-self.history_size:] if history else []

        # Check if we need context based on history
        context_check = self.process_history(
            history=recent_history,
            question=question
        )

        # Retrieve context if needed
        context = []
        if context_check.context_needed:
            context = self.retrieve(question).passages

        # Generate answer
        response = self.generate(
            history=recent_history,
            context=context,
            question=question
        )

        return {
            'answer': response.answer,
            'reasoning': response.reasoning
        }


class StructuredQA(dspy.Module):
    """A module that provides structured answers in specific formats"""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(
            "question -> required_fields, approach")
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(
            "context, question, required_fields -> structured_answer")

    def forward(self, question: str, output_format: Dict[str, str] = None) -> Dict[str, Any]:
        # Analyze question and determine required fields
        analysis = self.analyze(question=question)

        # Retrieve relevant information
        context = self.retrieve(question).passages

        # Generate structured answer
        response = self.generate(
            context=context,
            question=question,
            required_fields=output_format or analysis.required_fields
        )

        return {
            'structured_answer': response.structured_answer,
            'approach': analysis.approach
        }
