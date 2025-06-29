#!/usr/bin/env python3
"""
Output Refinement with DSPy

This script demonstrates output refinement techniques including best-of-n sampling,
response refinement, and quality improvement strategies for DSPy applications.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import json
import random
import statistics
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

@dataclass
class RefinementResult:
    original_output: str
    refined_output: str
    quality_score: float
    refinement_steps: List[str]
    metadata: Dict[str, Any]

def main():
    """Main function demonstrating output refinement with DSPy."""
    print("=" * 70)
    print("OUTPUT REFINEMENT WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for output refinement")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # DSPy Signatures for Output Refinement
    class QualityAssessment(dspy.Signature):
        """Assess the quality of generated output."""
        content = dspy.InputField(desc="Content to be assessed for quality")
        criteria = dspy.InputField(desc="Quality criteria to evaluate against")
        
        quality_score = dspy.OutputField(desc="Quality score from 1-10")
        strengths = dspy.OutputField(desc="Identified strengths in the content")
        weaknesses = dspy.OutputField(desc="Areas that need improvement")
        specific_issues = dspy.OutputField(desc="Specific issues that should be addressed")
    
    class ContentRefinement(dspy.Signature):
        """Refine content based on quality assessment feedback."""
        original_content = dspy.InputField(desc="Original content to be refined")
        feedback = dspy.InputField(desc="Feedback and improvement suggestions")
        refinement_goals = dspy.InputField(desc="Specific goals for refinement")
        
        refined_content = dspy.OutputField(desc="Improved version of the content")
        changes_made = dspy.OutputField(desc="Description of changes made")
        quality_improvements = dspy.OutputField(desc="Expected quality improvements")
    
    class ResponseComparison(dspy.Signature):
        """Compare multiple responses and select the best one."""
        responses = dspy.InputField(desc="List of response candidates to compare")
        evaluation_criteria = dspy.InputField(desc="Criteria for comparing responses")
        
        best_response = dspy.OutputField(desc="The highest quality response")
        ranking = dspy.OutputField(desc="Ranking of all responses with explanations")
        selection_reasoning = dspy.OutputField(desc="Detailed reasoning for the selection")
    
    class StyleRefinement(dspy.Signature):
        """Refine content style while preserving meaning."""
        content = dspy.InputField(desc="Content to be refined for style")
        target_style = dspy.InputField(desc="Target style specifications")
        constraints = dspy.InputField(desc="Constraints to maintain during refinement")
        
        styled_content = dspy.OutputField(desc="Content refined for target style")
        style_changes = dspy.OutputField(desc="Style changes applied")
        preserved_elements = dspy.OutputField(desc="Elements preserved from original")
    
    # Quality Metrics
    class QualityMetrics:
        """Calculate various quality metrics for content."""
        
        @staticmethod
        def calculate_readability_score(text: str) -> float:
            """Calculate a simple readability score."""
            if not text:
                return 0.0
            
            sentences = text.split('.')
            words = text.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            
            # Simple readability score (lower is better, normalized to 1-10)
            score = max(1, min(10, 10 - (avg_sentence_length - 15) / 5))
            return score
        
        @staticmethod
        def calculate_coherence_score(text: str) -> float:
            """Calculate coherence based on logical flow indicators."""
            if not text:
                return 0.0
            
            coherence_indicators = [
                'therefore', 'however', 'moreover', 'furthermore', 'consequently',
                'in addition', 'for example', 'specifically', 'in contrast',
                'similarly', 'meanwhile', 'subsequently', 'as a result'
            ]
            
            text_lower = text.lower()
            indicator_count = sum(1 for indicator in coherence_indicators if indicator in text_lower)
            
            # Normalize to 1-10 scale
            words = len(text.split())
            if words == 0:
                return 0.0
            
            coherence_ratio = indicator_count / (words / 100)  # per 100 words
            score = min(10, max(1, coherence_ratio * 3 + 5))
            return score
        
        @staticmethod
        def calculate_completeness_score(text: str, topic: str) -> float:
            """Calculate completeness based on topic coverage."""
            if not text or not topic:
                return 0.0
            
            # Simple heuristic based on length and topic keyword presence
            words = text.split()
            topic_words = topic.lower().split()
            
            topic_coverage = sum(1 for word in topic_words if word in text.lower())
            coverage_ratio = topic_coverage / len(topic_words) if topic_words else 0
            
            # Length factor (assuming adequate coverage needs reasonable length)
            length_factor = min(1.0, len(words) / 100)  # Optimal around 100+ words
            
            score = (coverage_ratio * 0.7 + length_factor * 0.3) * 10
            return min(10, max(1, score))
        
        @staticmethod
        def calculate_overall_quality(text: str, topic: str = "") -> Dict[str, float]:
            """Calculate overall quality metrics."""
            readability = QualityMetrics.calculate_readability_score(text)
            coherence = QualityMetrics.calculate_coherence_score(text)
            completeness = QualityMetrics.calculate_completeness_score(text, topic)
            
            overall = (readability + coherence + completeness) / 3
            
            return {
                "readability": readability,
                "coherence": coherence,
                "completeness": completeness,
                "overall": overall
            }
    
    # Output Refinement System
    class OutputRefiner(dspy.Module):
        """Comprehensive output refinement system."""
        
        def __init__(self):
            super().__init__()
            self.quality_assessor = dspy.ChainOfThought(QualityAssessment)
            self.content_refiner = dspy.ChainOfThought(ContentRefinement)
            self.response_comparator = dspy.ChainOfThought(ResponseComparison)
            self.style_refiner = dspy.ChainOfThought(StyleRefinement)
            self.quality_metrics = QualityMetrics()
        
        def assess_quality(self, content: str, criteria: str = "clarity, accuracy, completeness, engagement") -> dspy.Prediction:
            """Assess content quality."""
            return self.quality_assessor(content=content, criteria=criteria)
        
        def refine_content(self, content: str, feedback: str, goals: str = "improve overall quality") -> dspy.Prediction:
            """Refine content based on feedback."""
            return self.content_refiner(
                original_content=content,
                feedback=feedback,
                refinement_goals=goals
            )
        
        def best_of_n_selection(self, candidates: List[str], criteria: str = "quality, relevance, clarity") -> dspy.Prediction:
            """Select best response from multiple candidates."""
            candidates_str = "\n---\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(candidates)])
            
            return self.response_comparator(
                responses=candidates_str,
                evaluation_criteria=criteria
            )
        
        def refine_style(self, content: str, target_style: str, constraints: str = "preserve meaning and accuracy") -> dspy.Prediction:
            """Refine content style."""
            return self.style_refiner(
                content=content,
                target_style=target_style,
                constraints=constraints
            )
        
        def iterative_refinement(self, initial_content: str, max_iterations: int = 3, quality_threshold: float = 8.0) -> RefinementResult:
            """Perform iterative refinement until quality threshold is met."""
            current_content = initial_content
            refinement_steps = []
            iteration = 0
            
            while iteration < max_iterations:
                # Assess current quality
                quality_metrics = self.quality_metrics.calculate_overall_quality(current_content)
                
                if quality_metrics["overall"] >= quality_threshold:
                    break
                
                # Get quality assessment
                assessment = self.assess_quality(current_content)
                
                # Refine based on assessment
                refinement = self.refine_content(
                    current_content,
                    f"Weaknesses: {assessment.weaknesses}. Issues: {assessment.specific_issues}",
                    "Address identified weaknesses and improve overall quality"
                )
                
                current_content = refinement.refined_content
                refinement_steps.append(f"Iteration {iteration + 1}: {refinement.changes_made}")
                iteration += 1
            
            final_metrics = self.quality_metrics.calculate_overall_quality(current_content)
            
            return RefinementResult(
                original_output=initial_content,
                refined_output=current_content,
                quality_score=final_metrics["overall"],
                refinement_steps=refinement_steps,
                metadata={
                    "iterations": iteration,
                    "final_metrics": final_metrics,
                    "threshold_met": final_metrics["overall"] >= quality_threshold
                }
            )
        
        def generate_multiple_candidates(self, prompt: str, n: int = 5) -> List[str]:
            """Generate multiple candidate responses."""
            # Simple signature for generating candidates
            class CandidateGeneration(dspy.Signature):
                """Generate a response to the given prompt."""
                prompt = dspy.InputField(desc="The prompt to respond to")
                response = dspy.OutputField(desc="Generated response")
            
            generator = dspy.Predict(CandidateGeneration)
            candidates = []
            
            for i in range(n):
                try:
                    # Add some variation by modifying the prompt slightly
                    varied_prompt = f"{prompt} (Variation {i+1}: Focus on providing a unique perspective)"
                    result = generator(prompt=varied_prompt)
                    candidates.append(result.response)
                except Exception as e:
                    print_error(f"Error generating candidate {i+1}: {e}")
                    candidates.append(f"Error generating candidate {i+1}")
            
            return candidates
    
    # Specialized Refinement Modules
    class SummarizationRefiner(dspy.Module):
        """Specialized refiner for summarization tasks."""
        
        def __init__(self):
            super().__init__()
            
            class SummarizationTask(dspy.Signature):
                """Summarize the given text."""
                text = dspy.InputField(desc="Text to summarize")
                summary = dspy.OutputField(desc="Concise summary of the text")
            
            self.summarizer = dspy.ChainOfThought(SummarizationTask)
            self.refiner = OutputRefiner()
        
        def generate_and_refine_summary(self, text: str) -> RefinementResult:
            """Generate summary and refine it."""
            # Generate initial summary
            initial_summary = self.summarizer(text=text).summary
            
            # Refine the summary
            return self.refiner.iterative_refinement(
                initial_summary,
                max_iterations=2,
                quality_threshold=7.5
            )
    
    class QuestionAnsweringRefiner(dspy.Module):
        """Specialized refiner for Q&A tasks."""
        
        def __init__(self):
            super().__init__()
            
            class QATask(dspy.Signature):
                """Answer the given question."""
                question = dspy.InputField(desc="Question to answer")
                answer = dspy.OutputField(desc="Comprehensive answer")
            
            self.qa_module = dspy.ChainOfThought(QATask)
            self.refiner = OutputRefiner()
        
        def generate_and_refine_answer(self, question: str) -> RefinementResult:
            """Generate answer and refine it."""
            # Generate initial answer
            initial_answer = self.qa_module(question=question).answer
            
            # Refine the answer
            return self.refiner.iterative_refinement(
                initial_answer,
                max_iterations=2,
                quality_threshold=8.0
            )
    
    # Initialize refinement systems
    output_refiner = OutputRefiner()
    summarization_refiner = SummarizationRefiner()
    qa_refiner = QuestionAnsweringRefiner()
    
    print_result("Output refinement systems initialized successfully!")
    
    # Demo 1: Quality Assessment
    print_step("Quality Assessment Demo", "Assessing quality of different text samples")
    
    sample_texts = [
        {
            "content": "AI is good. It helps people. Many companies use it. It's the future.",
            "description": "Simple, low-quality text"
        },
        {
            "content": "Artificial Intelligence represents a transformative technology that is revolutionizing numerous industries. Through machine learning algorithms and sophisticated data processing capabilities, AI systems can analyze complex patterns, make predictions, and automate decision-making processes. Consequently, organizations across various sectors are integrating AI solutions to enhance efficiency, improve customer experiences, and drive innovation in their respective domains.",
            "description": "Well-structured, higher-quality text"
        },
        {
            "content": "The implementation of artificial intelligence technologies presents both unprecedented opportunities and significant challenges for modern enterprises. However, successful deployment requires careful consideration of ethical implications, data privacy concerns, and workforce impact. Therefore, organizations must develop comprehensive strategies that balance technological advancement with responsible innovation practices.",
            "description": "High-quality, well-reasoned text"
        }
    ]
    
    for i, sample in enumerate(sample_texts, 1):
        try:
            # DSPy quality assessment
            assessment = output_refiner.assess_quality(sample["content"])
            
            # Metrics-based assessment
            metrics = output_refiner.quality_metrics.calculate_overall_quality(sample["content"], "artificial intelligence")
            
            print(f"\n--- Sample {i}: {sample['description']} ---")
            print_result(f"Text: {sample['content'][:100]}...", "Content")
            print_result(f"DSPy Quality Score: {assessment.quality_score}/10", "DSPy Assessment")
            print_result(f"Metrics Score: {metrics['overall']:.1f}/10", "Metrics Assessment")
            print_result(f"Strengths: {assessment.strengths}", "Strengths")
            print_result(f"Weaknesses: {assessment.weaknesses}", "Weaknesses")
            print_result(f"Detailed Metrics: {metrics}", "Detailed Metrics")
            
        except Exception as e:
            print_error(f"Error assessing sample {i}: {e}")
    
    # Demo 2: Best-of-N Selection
    print_step("Best-of-N Selection Demo", "Selecting best response from multiple candidates")
    
    test_prompt = "Explain the benefits of renewable energy for the environment"
    
    try:
        # Generate multiple candidates
        print("Generating multiple candidate responses...")
        candidates = output_refiner.generate_multiple_candidates(test_prompt, n=4)
        
        print_result(f"Generated {len(candidates)} candidates", "Generation Status")
        
        # Show candidates
        for i, candidate in enumerate(candidates, 1):
            print(f"\nCandidate {i}: {candidate[:150]}...")
        
        # Select best candidate
        best_selection = output_refiner.best_of_n_selection(candidates)
        
        print_result(f"Best Response: {best_selection.best_response}", "Selected Best")
        print_result(f"Ranking: {best_selection.ranking}", "Full Ranking")
        print_result(f"Reasoning: {best_selection.selection_reasoning}", "Selection Reasoning")
        
    except Exception as e:
        print_error(f"Error in best-of-n selection: {e}")
    
    # Demo 3: Iterative Refinement
    print_step("Iterative Refinement Demo", "Improving content through multiple iterations")
    
    initial_content = "Climate change is bad. It affects weather. People should do something about it. Governments need to act."
    
    try:
        refinement_result = output_refiner.iterative_refinement(
            initial_content,
            max_iterations=3,
            quality_threshold=7.5
        )
        
        print_result(f"Original: {refinement_result.original_output}", "Original Content")
        print_result(f"Refined: {refinement_result.refined_output}", "Refined Content")
        print_result(f"Quality Score: {refinement_result.quality_score:.1f}/10", "Final Quality")
        print_result(f"Iterations: {refinement_result.metadata['iterations']}", "Refinement Process")
        
        print("\nRefinement Steps:")
        for step in refinement_result.refinement_steps:
            print(f"  - {step}")
        
        print_result(f"Threshold Met: {refinement_result.metadata['threshold_met']}", "Success Status")
        
    except Exception as e:
        print_error(f"Error in iterative refinement: {e}")
    
    # Demo 4: Style Refinement
    print_step("Style Refinement Demo", "Adapting content to different styles")
    
    base_content = "Machine learning algorithms can process large amounts of data quickly. They find patterns that humans might miss. This technology is used in many applications today."
    
    style_targets = [
        {
            "style": "academic research paper",
            "description": "Formal academic style"
        },
        {
            "style": "casual blog post for general audience",
            "description": "Casual, accessible style"
        },
        {
            "style": "executive business summary",
            "description": "Professional business style"
        },
        {
            "style": "educational content for children",
            "description": "Simple, engaging style"
        }
    ]
    
    for i, target in enumerate(style_targets, 1):
        try:
            refined = output_refiner.refine_style(
                base_content,
                target["style"],
                "maintain technical accuracy while adapting style"
            )
            
            print(f"\n--- Style {i}: {target['description']} ---")
            print_result(f"Original: {base_content}", "Original")
            print_result(f"Refined: {refined.styled_content}", "Styled Version")
            print_result(f"Changes: {refined.style_changes}", "Style Changes")
            print_result(f"Preserved: {refined.preserved_elements}", "Preserved Elements")
            
        except Exception as e:
            print_error(f"Error refining style {i}: {e}")
    
    # Demo 5: Specialized Refinement (Summarization)
    print_step("Specialized Summarization Refinement", "Refining summarization outputs")
    
    long_text = """
    Artificial intelligence (AI) and machine learning (ML) have emerged as transformative technologies that are reshaping industries across the globe. These technologies enable computers to learn from data, recognize patterns, and make decisions with minimal human intervention. The applications of AI and ML are vast and varied, spanning from healthcare and finance to transportation and entertainment.
    
    In healthcare, AI is being used to develop more accurate diagnostic tools, personalized treatment plans, and drug discovery processes. Machine learning algorithms can analyze medical images, predict disease outcomes, and assist doctors in making more informed decisions. For example, AI-powered systems can detect early signs of cancer in mammograms with greater accuracy than human radiologists.
    
    The financial sector has also embraced AI and ML for fraud detection, algorithmic trading, risk assessment, and customer service automation. Banks and financial institutions use these technologies to analyze transaction patterns, identify suspicious activities, and provide personalized financial advice to customers.
    
    In transportation, autonomous vehicles represent one of the most visible applications of AI. Self-driving cars use a combination of sensors, cameras, and machine learning algorithms to navigate roads, avoid obstacles, and make real-time driving decisions. This technology promises to reduce traffic accidents, improve traffic flow, and provide mobility solutions for people who cannot drive traditional vehicles.
    
    However, the rapid adoption of AI and ML also raises important ethical and societal concerns. Issues such as job displacement, privacy, algorithmic bias, and the concentration of power in tech companies need to be carefully addressed. As these technologies continue to evolve, it's crucial to develop appropriate governance frameworks that ensure their benefits are distributed fairly while minimizing potential harms.
    """
    
    try:
        summary_result = summarization_refiner.generate_and_refine_summary(long_text)
        
        print_result(f"Original Length: {len(long_text)} characters", "Input")
        print_result(f"Initial Summary: {summary_result.original_output}", "Initial Summary")
        print_result(f"Refined Summary: {summary_result.refined_output}", "Refined Summary")
        print_result(f"Quality Score: {summary_result.quality_score:.1f}/10", "Quality")
        print_result(f"Refinement Process: {len(summary_result.refinement_steps)} steps", "Process")
        
    except Exception as e:
        print_error(f"Error in summarization refinement: {e}")
    
    # Demo 6: Question Answering Refinement
    print_step("Q&A Refinement Demo", "Improving question-answering responses")
    
    test_questions = [
        "What are the main advantages of using renewable energy sources?",
        "How does quantum computing differ from classical computing?",
        "What factors contribute to climate change and what can individuals do to help?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        try:
            qa_result = qa_refiner.generate_and_refine_answer(question)
            
            print(f"\n--- Q&A Refinement {i} ---")
            print_result(f"Question: {question}", "Question")
            print_result(f"Initial Answer: {qa_result.original_output[:200]}...", "Initial Answer")
            print_result(f"Refined Answer: {qa_result.refined_output[:200]}...", "Refined Answer")
            print_result(f"Quality Score: {qa_result.quality_score:.1f}/10", "Quality")
            
            if qa_result.refinement_steps:
                print("Refinement Steps:")
                for step in qa_result.refinement_steps:
                    print(f"  - {step}")
        
        except Exception as e:
            print_error(f"Error in Q&A refinement {i}: {e}")
    
    # Demo 7: Comparative Refinement Analysis
    print_step("Comparative Analysis", "Comparing refinement effectiveness")
    
    test_cases = [
        {
            "content": "AI is good for business. It saves time and money. Companies should use it more.",
            "expected_improvement": "High - very simple content with room for improvement"
        },
        {
            "content": "Artificial intelligence offers significant advantages for modern enterprises, including enhanced efficiency, cost reduction, and improved decision-making capabilities through data-driven insights.",
            "expected_improvement": "Medium - already decent quality"
        },
        {
            "content": "The strategic implementation of artificial intelligence technologies within organizational frameworks necessitates comprehensive evaluation of operational paradigms, resource allocation methodologies, and stakeholder engagement protocols to optimize performance metrics while ensuring ethical compliance and sustainable value creation.",
            "expected_improvement": "Low - already high quality, may need simplification"
        }
    ]
    
    refinement_results = []
    
    for i, case in enumerate(test_cases, 1):
        try:
            initial_metrics = output_refiner.quality_metrics.calculate_overall_quality(case["content"])
            refinement_result = output_refiner.iterative_refinement(case["content"], max_iterations=2)
            final_metrics = refinement_result.metadata["final_metrics"]
            
            improvement = final_metrics["overall"] - initial_metrics["overall"]
            
            refinement_results.append({
                "case": i,
                "initial_quality": initial_metrics["overall"],
                "final_quality": final_metrics["overall"],
                "improvement": improvement,
                "expected": case["expected_improvement"]
            })
            
            print(f"\n--- Case {i} Analysis ---")
            print_result(f"Initial Quality: {initial_metrics['overall']:.1f}/10", "Before")
            print_result(f"Final Quality: {final_metrics['overall']:.1f}/10", "After")
            print_result(f"Improvement: {improvement:.1f} points", "Change")
            print_result(f"Expected: {case['expected_improvement']}", "Expectation")
            
        except Exception as e:
            print_error(f"Error analyzing case {i}: {e}")
    
    # Summary of refinement effectiveness
    if refinement_results:
        avg_improvement = statistics.mean(result["improvement"] for result in refinement_results)
        print_result(f"Average Quality Improvement: {avg_improvement:.1f} points", "Overall Performance")
    
    print("\n" + "="*70)
    print("OUTPUT REFINEMENT COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy output refinement techniques!")

if __name__ == "__main__":
    main()
