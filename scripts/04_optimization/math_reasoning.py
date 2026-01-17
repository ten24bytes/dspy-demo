#!/usr/bin/env python3
"""
Mathematical Reasoning with DSPy

This script demonstrates how to build mathematical reasoning systems using DSPy.
It covers different approaches for solving math problems, optimization, and evaluation.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import re
import random
import math
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error

@dataclass
class MathProblem:
    """Represents a math problem with its solution."""
    problem: str
    answer: Union[float, int, str]
    difficulty: str = "medium"
    category: str = "general"

class BasicMathReasoning(dspy.Signature):
    """Solve mathematical problems step by step."""
    
    problem = dspy.InputField(desc="The mathematical problem to solve")
    reasoning = dspy.OutputField(desc="Step-by-step solution process")
    answer = dspy.OutputField(desc="The final numerical answer")

class AlgebraReasoning(dspy.Signature):
    """Solve algebra problems with detailed steps."""
    
    problem = dspy.InputField(desc="The algebra problem to solve")
    equation_setup = dspy.OutputField(desc="Setting up the equation(s)")
    solution_steps = dspy.OutputField(desc="Step-by-step algebraic solution")
    verification = dspy.OutputField(desc="Verification of the answer")
    answer = dspy.OutputField(desc="The final answer")

class GeometryReasoning(dspy.Signature):
    """Solve geometry problems with visual reasoning."""
    
    problem = dspy.InputField(desc="The geometry problem to solve")
    diagram_analysis = dspy.OutputField(desc="Analysis of the geometric setup")
    formula_selection = dspy.OutputField(desc="Relevant formulas and theorems")
    calculation = dspy.OutputField(desc="Step-by-step calculation")
    answer = dspy.OutputField(desc="The final answer with units")

class WordProblemReasoning(dspy.Signature):
    """Solve word problems by extracting mathematical components."""
    
    problem = dspy.InputField(desc="The word problem to solve")
    key_information = dspy.OutputField(desc="Important numbers and relationships")
    mathematical_model = dspy.OutputField(desc="Mathematical representation")
    solution = dspy.OutputField(desc="Step-by-step solution")
    answer = dspy.OutputField(desc="The final answer in context")

class BasicMathSolver(dspy.Module):
    """Basic mathematical reasoning solver."""
    
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(BasicMathReasoning)
    
    def forward(self, problem: str) -> dspy.Prediction:
        return self.solve(problem=problem)

class ProgramOfThought(dspy.Module):
    """Program-aided mathematical reasoning."""
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought("problem -> analysis, approach")
        self.generate_code = dspy.ChainOfThought("problem, analysis -> python_code")
        self.solve = dspy.ChainOfThought("problem, code_result -> final_answer")
    
    def forward(self, problem: str) -> dspy.Prediction:
        # Step 1: Analyze the problem
        analysis = self.analyze(problem=problem)
        
        # Step 2: Generate Python code
        code_gen = self.generate_code(
            problem=problem,
            analysis=analysis.analysis
        )
        
        # Step 3: Execute code (simplified simulation)
        try:
            # This is a simplified version - in practice, you'd use safe code execution
            code_result = self._execute_safe_code(code_gen.python_code)
        except Exception as e:
            code_result = f"Error executing code: {e}"
        
        # Step 4: Generate final answer
        solution = self.solve(
            problem=problem,
            code_result=code_result
        )
        
        return dspy.Prediction(
            problem=problem,
            analysis=analysis.analysis,
            approach=analysis.approach,
            python_code=code_gen.python_code,
            code_result=code_result,
            final_answer=solution.final_answer
        )
    
    def _execute_safe_code(self, code: str) -> str:
        """Safely execute simple mathematical code."""
        try:
            # Very basic and safe execution for demo
            # In practice, use proper sandboxing
            if "import" in code or "exec" in code or "eval" in code:
                return "Code execution not allowed for safety"
            
            # Allow only basic math operations
            allowed_globals = {
                '__builtins__': {},
                'math': math,
                'abs': abs,
                'round': round,
                'pow': pow,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'pi': math.pi,
                'e': math.e
            }
            
            # Execute the code
            local_vars = {}
            exec(code, allowed_globals, local_vars)
            
            # Return the result
            if 'result' in local_vars:
                return str(local_vars['result'])
            else:
                return "No result variable found"
                
        except Exception as e:
            return f"Execution error: {e}"

class MultiStepMathSolver(dspy.Module):
    """Multi-step mathematical solver with verification."""
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought("problem -> problem_type, difficulty")
        self.algebra_solver = dspy.ChainOfThought(AlgebraReasoning)
        self.geometry_solver = dspy.ChainOfThought(GeometryReasoning)
        self.word_solver = dspy.ChainOfThought(WordProblemReasoning)
        self.verify = dspy.ChainOfThought("problem, answer -> verification, confidence")
    
    def forward(self, problem: str) -> dspy.Prediction:
        # Step 1: Classify problem type
        classification = self.classify(problem=problem)
        
        # Step 2: Solve based on type
        problem_type = classification.problem_type.lower()
        
        if "algebra" in problem_type:
            solution = self.algebra_solver(problem=problem)
            solver_used = "algebra"
        elif "geometry" in problem_type:
            solution = self.geometry_solver(problem=problem)
            solver_used = "geometry"
        elif "word" in problem_type:
            solution = self.word_solver(problem=problem)
            solver_used = "word_problem"
        else:
            # Default to basic solver
            basic_solver = BasicMathSolver()
            solution = basic_solver(problem=problem)
            solver_used = "basic"
        
        # Step 3: Verify answer
        verification = self.verify(
            problem=problem,
            answer=solution.answer
        )
        
        return dspy.Prediction(
            problem=problem,
            problem_type=classification.problem_type,
            difficulty=classification.difficulty,
            solver_used=solver_used,
            solution=solution,
            verification=verification.verification,
            confidence=verification.confidence
        )

def create_sample_problems() -> List[MathProblem]:
    """Create sample math problems for testing."""
    
    problems = [
        # Arithmetic
        MathProblem(
            "What is 15% of 240?",
            36,
            "easy",
            "arithmetic"
        ),
        MathProblem(
            "If a train travels 120 miles in 2 hours, what is its average speed?",
            60,
            "easy",
            "word_problem"
        ),
        
        # Algebra
        MathProblem(
            "Solve for x: 2x + 5 = 17",
            6,
            "medium",
            "algebra"
        ),
        MathProblem(
            "If 3x - 7 = 2x + 9, what is the value of x?",
            16,
            "medium",
            "algebra"
        ),
        
        # Geometry
        MathProblem(
            "Find the area of a circle with radius 5 units.",
            78.54,  # π * 5² ≈ 78.54
            "medium",
            "geometry"
        ),
        MathProblem(
            "What is the perimeter of a rectangle with length 8 and width 6?",
            28,
            "easy",
            "geometry"
        ),
        
        # Word problems
        MathProblem(
            "Sarah has 3 times as many apples as John. If John has 12 apples, how many apples does Sarah have?",
            36,
            "easy",
            "word_problem"
        ),
        MathProblem(
            "A store is having a 25% off sale. If an item originally costs $80, what is the sale price?",
            60,
            "medium",
            "word_problem"
        ),
        
        # Advanced
        MathProblem(
            "Find the quadratic formula solution for x² - 5x + 6 = 0",
            "x = 2 or x = 3",
            "hard",
            "algebra"
        ),
        MathProblem(
            "What is the volume of a cylinder with radius 3 and height 10?",
            282.74,  # π * 3² * 10 ≈ 282.74
            "hard",
            "geometry"
        )
    ]
    
    return problems

def evaluate_math_solver(solver, test_problems: List[MathProblem]) -> Dict[str, Any]:
    """Evaluate mathematical reasoning performance."""
    
    results = {
        'total': len(test_problems),
        'correct': 0,
        'by_category': {},
        'by_difficulty': {},
        'details': []
    }
    
    for problem in test_problems:
        try:
            # Get prediction
            prediction = solver(problem=problem.problem)
            
            # Extract numerical answer
            predicted_answer = extract_numerical_answer(prediction.answer if hasattr(prediction, 'answer') else str(prediction))
            expected_answer = problem.answer
            
            # Check if correct
            is_correct = check_math_answer(predicted_answer, expected_answer)
            
            if is_correct:
                results['correct'] += 1
            
            # Track by category
            category = problem.category
            if category not in results['by_category']:
                results['by_category'][category] = {'total': 0, 'correct': 0}
            results['by_category'][category]['total'] += 1
            if is_correct:
                results['by_category'][category]['correct'] += 1
            
            # Track by difficulty
            difficulty = problem.difficulty
            if difficulty not in results['by_difficulty']:
                results['by_difficulty'][difficulty] = {'total': 0, 'correct': 0}
            results['by_difficulty'][difficulty]['total'] += 1
            if is_correct:
                results['by_difficulty'][difficulty]['correct'] += 1
            
            # Store details
            results['details'].append({
                'problem': problem.problem,
                'expected': expected_answer,
                'predicted': predicted_answer,
                'correct': is_correct,
                'category': category,
                'difficulty': difficulty
            })
            
        except Exception as e:
            print_error(f"Error solving problem '{problem.problem}': {e}")
            results['details'].append({
                'problem': problem.problem,
                'expected': problem.answer,
                'predicted': 'ERROR',
                'correct': False,
                'category': problem.category,
                'difficulty': problem.difficulty
            })
    
    # Calculate accuracy
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    return results

def extract_numerical_answer(answer_text: str) -> Union[float, str]:
    """Extract numerical answer from text."""
    
    if not answer_text:
        return None
    
    # Try to find numbers in the text
    numbers = re.findall(r'-?\d+\.?\d*', str(answer_text))
    
    if numbers:
        try:
            # Return the last number found (often the final answer)
            return float(numbers[-1])
        except ValueError:
            pass
    
    # Return original text if no number found
    return answer_text.strip()

def check_math_answer(predicted, expected, tolerance=0.01) -> bool:
    """Check if mathematical answer is correct within tolerance."""
    
    try:
        # Handle string answers
        if isinstance(expected, str) or isinstance(predicted, str):
            return str(predicted).strip().lower() == str(expected).strip().lower()
        
        # Handle numerical answers
        if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
            return abs(float(predicted) - float(expected)) <= tolerance
        
        return False
        
    except Exception:
        return False

def demonstrate_basic_math():
    """Demonstrate basic mathematical reasoning."""
    
    print_step("Basic Mathematical Reasoning")
    
    solver = BasicMathSolver()
    
    problems = [
        "What is 25 × 16?",
        "Calculate 144 ÷ 12",
        "Find the square root of 169",
        "What is 2³ + 3²?"
    ]
    
    for problem in problems:
        try:
            result = solver(problem=problem)
            print_result(f"Problem: {problem}")
            print_result(f"Reasoning: {result.reasoning}")
            print_result(f"Answer: {result.answer}")
            print("-" * 50)
        except Exception as e:
            print_error(f"Error solving problem: {e}")

def demonstrate_program_of_thought():
    """Demonstrate program-aided mathematical reasoning."""
    
    print_step("Program-Aided Mathematical Reasoning")
    
    solver = ProgramOfThought()
    
    problems = [
        "Calculate the compound interest on $1000 at 5% annual rate for 3 years",
        "Find the Fibonacci number at position 10",
        "Calculate the area of a triangle with sides 3, 4, and 5"
    ]
    
    for problem in problems:
        try:
            result = solver(problem=problem)
            print_result(f"Problem: {problem}")
            print_result(f"Analysis: {result.analysis}")
            print_result(f"Approach: {result.approach}")
            print_result(f"Python Code: {result.python_code}")
            print_result(f"Code Result: {result.code_result}")
            print_result(f"Final Answer: {result.final_answer}")
            print("-" * 50)
        except Exception as e:
            print_error(f"Error in program-aided reasoning: {e}")

def demonstrate_multi_step_solving():
    """Demonstrate multi-step mathematical problem solving."""
    
    print_step("Multi-Step Mathematical Problem Solving")
    
    solver = MultiStepMathSolver()
    
    problem = ("A rectangular garden has a length that is 3 meters more than twice its width. "
              "If the perimeter of the garden is 36 meters, what are the dimensions of the garden?")
    
    try:
        result = solver(problem=problem)
        print_result(f"Problem: {problem}")
        print_result(f"Problem Type: {result.problem_type}")
        print_result(f"Difficulty: {result.difficulty}")
        print_result(f"Solver Used: {result.solver_used}")
        print_result(f"Solution: {result.solution}")
        print_result(f"Verification: {result.verification}")
        print_result(f"Confidence: {result.confidence}")
        
    except Exception as e:
        print_error(f"Error in multi-step solving: {e}")

def demonstrate_evaluation():
    """Demonstrate evaluation of mathematical reasoning."""
    
    print_step("Mathematical Reasoning Evaluation")
    
    # Create test problems
    test_problems = create_sample_problems()
    
    # Test basic solver
    print_result("Evaluating Basic Math Solver:")
    basic_solver = BasicMathSolver()
    basic_results = evaluate_math_solver(basic_solver, test_problems[:5])  # Use subset for demo
    
    print_result(f"Accuracy: {basic_results['accuracy']:.2%}")
    print_result(f"Correct: {basic_results['correct']}/{basic_results['total']}")
    
    # Show results by category
    print_result("Results by Category:")
    for category, stats in basic_results['by_category'].items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print_result(f"  {category}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Show some examples
    print_result("Example Results:")
    for detail in basic_results['details'][:3]:
        status = "✓" if detail['correct'] else "✗"
        print_result(f"  {status} {detail['problem']}")
        print_result(f"    Expected: {detail['expected']}, Got: {detail['predicted']}")

def main():
    """Main function demonstrating mathematical reasoning."""
    
    print("=" * 60)
    print("DSPy Mathematical Reasoning Demo")
    print("=" * 60)
    
    # Setup language model
    lm = setup_default_lm()
    if not lm:
        return
    
    try:
        # Basic mathematical reasoning
        demonstrate_basic_math()
        
        # Program-aided reasoning
        demonstrate_program_of_thought()
        
        # Multi-step solving
        demonstrate_multi_step_solving()
        
        # Evaluation
        demonstrate_evaluation()
        
        print_step("Mathematical Reasoning Complete!")
        
    except Exception as e:
        print_error(f"Error in mathematical reasoning demo: {e}")

if __name__ == "__main__":
    main()
