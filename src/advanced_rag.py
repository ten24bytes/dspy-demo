import dspy
from typing import List, Dict, Any


class MultiHopRetriever(dspy.Module):
    def __init__(self, initial_k: int = 3, follow_up_k: int = 2):
        super().__init__()
        self.initial_k = initial_k
        self.follow_up_k = follow_up_k
        self.retrieve = dspy.Retrieve(k=initial_k)
        self.generate_questions = dspy.ChainOfThought(
            "context, question -> follow_up_questions")
        self.synthesize = dspy.ChainOfThought(
            "all_contexts, question -> final_answer")

    def forward(self, question: str) -> Dict[str, Any]:
        # Initial retrieval
        initial_contexts = self.retrieve(question).passages

        # Generate follow-up questions based on initial context
        questions = self.generate_questions(
            context=initial_contexts,
            question=question
        )

        # Retrieve context for follow-up questions
        all_contexts = initial_contexts
        for q in questions.follow_up_questions.split(';'):
            if q.strip():
                follow_up_contexts = self.retrieve(
                    q.strip(), k=self.follow_up_k).passages
                all_contexts.extend(follow_up_contexts)

        # Synthesize final answer
        final_response = self.synthesize(
            all_contexts=all_contexts,
            question=question
        )

        return {
            'answer': final_response.final_answer,
            'contexts': all_contexts
        }


class SelfCorrectingRAG(dspy.Module):
    def __init__(self, max_attempts: int = 2):
        super().__init__()
        self.max_attempts = max_attempts
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(
            "context, question -> answer, reasoning")
        self.verify = dspy.ChainOfThought(
            "context, answer, reasoning -> is_correct, corrections, confidence")

    def forward(self, question: str) -> Dict[str, Any]:
        contexts = self.retrieve(question).passages

        for attempt in range(self.max_attempts):
            # Generate answer
            response = self.generate(
                context=contexts,
                question=question
            )

            # Verify answer
            verification = self.verify(
                context=contexts,
                answer=response.answer,
                reasoning=response.reasoning
            )

            # If confident and correct, return answer
            if verification.is_correct and float(verification.confidence) > 0.8:
                return {
                    'answer': response.answer,
                    'reasoning': response.reasoning,
                    'attempts': attempt + 1,
                    'confidence': verification.confidence
                }

            # Update answer based on corrections
            response = self.generate(
                context=contexts,
                question=question + "\nCorrections: " + verification.corrections
            )

        # Return best attempt if max attempts reached
        return {
            'answer': response.answer,
            'reasoning': response.reasoning,
            'attempts': self.max_attempts,
            'confidence': verification.confidence
        }


class ProgressiveRAG(dspy.Module):
    """RAG implementation that progressively refines the query and retrieval"""

    def __init__(self, max_iterations: int = 3):
        super().__init__()
        self.max_iterations = max_iterations
        self.retrieve = dspy.Retrieve(k=2)
        self.refine_query = dspy.ChainOfThought(
            "original_question, current_context, previous_queries -> refined_query")
        self.assess_relevance = dspy.ChainOfThought(
            "context, question -> relevance_score")
        self.generate_answer = dspy.ChainOfThought(
            "all_contexts, question -> final_answer")

    def forward(self, question: str) -> Dict[str, Any]:
        all_contexts = []
        queries = [question]

        for i in range(self.max_iterations):
            # Retrieve based on current query
            current_contexts = self.retrieve(queries[-1]).passages

            # Assess relevance
            relevance = self.assess_relevance(
                context=current_contexts,
                question=question
            )

            # Store relevant contexts
            if float(relevance.relevance_score) > 0.7:
                all_contexts.extend(current_contexts)

            # Refine query if needed
            if i < self.max_iterations - 1:
                refined = self.refine_query(
                    original_question=question,
                    current_context=current_contexts,
                    previous_queries=queries
                )
                queries.append(refined.refined_query)

        # Generate final answer
        final_response = self.generate_answer(
            all_contexts=all_contexts,
            question=question
        )

        return {
            'answer': final_response.final_answer,
            'queries_used': queries,
            'context_count': len(all_contexts)
        }
