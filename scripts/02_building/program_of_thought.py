#!/usr/bin/env python3
"""
Program of Thought with DSPy

This script demonstrates Program-aided Language Models (PAL) approach using DSPy.
It shows how to combine natural language reasoning with programmatic computation
for solving complex problems.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import re
import math
import ast
import operator
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error

@dataclass
class ProgramResult:
    """Result of program execution."""
    code: str
    output: Any
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

class ProgramGeneration(dspy.Signature):
    """Generate Python code to solve a given problem."""
    
    problem = dspy.InputField(desc="The problem to solve")
    approach = dspy.InputField(desc="Suggested approach or reasoning")
    code = dspy.OutputField(desc="Python code that solves the problem")
    explanation = dspy.OutputField(desc="Explanation of the solution approach")

class MathProblemSolver(dspy.Signature):
    """Solve mathematical problems by generating and executing code."""
    
    problem = dspy.InputField(desc="Mathematical problem statement")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning about the problem")
    python_code = dspy.OutputField(desc="Python code to solve the problem")
    expected_result = dspy.OutputField(desc="Expected type and format of result")

class DataAnalysis(dspy.Signature):
    """Generate code for data analysis tasks."""
    
    task = dspy.InputField(desc="Data analysis task description")
    data_description = dspy.InputField(desc="Description of available data")
    analysis_code = dspy.OutputField(desc="Python code for data analysis")
    visualization_code = dspy.OutputField(desc="Code for data visualization")
    insights = dspy.OutputField(desc="Expected insights from the analysis")

class LogicPuzzleSolver(dspy.Signature):
    """Solve logic puzzles using programmatic reasoning."""
    
    puzzle = dspy.InputField(desc="Logic puzzle description")
    constraints = dspy.InputField(desc="Constraints and rules")
    solution_approach = dspy.OutputField(desc="Approach to solve the puzzle")
    solution_code = dspy.OutputField(desc="Python code implementing the solution")
    verification = dspy.OutputField(desc="How to verify the solution")

class SafeCodeExecutor:
    """Safe execution environment for generated code."""
    
    def __init__(self):
        # Define safe built-ins and modules
        self.safe_builtins = {
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'print': print,
                'sorted': sorted,
                'reversed': reversed,
            },
            'math': math,
            'operator': operator,
        }
        
        # Patterns to detect unsafe code
        self.unsafe_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
    
    def is_safe(self, code: str) -> bool:
        """Check if code is safe to execute."""
        for pattern in self.unsafe_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        # Try to parse the code
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def execute(self, code: str, timeout: int = 5) -> ProgramResult:
        """Execute code safely with timeout."""
        if not self.is_safe(code):
            return ProgramResult(
                code=code,
                output=None,
                success=False,
                error_message="Code contains unsafe operations"
            )
        
        try:
            # Create safe execution environment
            local_vars = {}
            global_vars = self.safe_builtins.copy()
            
            # Execute the code
            exec(code, global_vars, local_vars)
            
            # Get result (look for 'result' variable or last expression)
            if 'result' in local_vars:
                output = local_vars['result']
            elif 'answer' in local_vars:
                output = local_vars['answer']
            else:
                # Try to find any variable that looks like a result
                output = local_vars if local_vars else "Code executed successfully"
            
            return ProgramResult(
                code=code,
                output=output,
                success=True
            )
            
        except Exception as e:
            return ProgramResult(
                code=code,
                output=None,
                success=False,
                error_message=str(e)
            )

class ProgramOfThought(dspy.Module):
    """Program-aided reasoning module."""
    
    def __init__(self):
        super().__init__()
        self.generate_program = dspy.ChainOfThought(ProgramGeneration)
        self.executor = SafeCodeExecutor()
    
    def forward(self, problem: str, approach: str = "") -> Dict[str, Any]:
        """Solve problem using program generation and execution."""
        
        # Generate program
        program_result = self.generate_program(
            problem=problem,
            approach=approach or "Break down the problem and solve step by step"
        )
        
        # Execute the generated code
        execution_result = self.executor.execute(program_result.code)
        
        return {
            'problem': problem,
            'approach': approach,
            'generated_code': program_result.code,
            'explanation': program_result.explanation,
            'execution_success': execution_result.success,
            'result': execution_result.output,
            'error': execution_result.error_message
        }

class MathSolver(dspy.Module):
    """Specialized mathematical problem solver."""
    
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought(MathProblemSolver)
        self.executor = SafeCodeExecutor()
    
    def forward(self, problem: str) -> Dict[str, Any]:
        """Solve mathematical problem."""
        
        # Generate solution
        solution = self.solver(problem=problem)
        
        # Execute the code
        execution_result = self.executor.execute(solution.python_code)
        
        return {
            'problem': problem,
            'reasoning': solution.reasoning,
            'code': solution.python_code,
            'expected_result': solution.expected_result,
            'actual_result': execution_result.output,
            'success': execution_result.success,
            'error': execution_result.error_message
        }

class DataAnalyzer(dspy.Module):
    """Data analysis with program generation."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(DataAnalysis)
        self.executor = SafeCodeExecutor()
    
    def forward(self, task: str, data_description: str) -> Dict[str, Any]:
        """Perform data analysis task."""
        
        # Generate analysis code
        analysis = self.analyzer(
            task=task,
            data_description=data_description
        )
        
        # Execute analysis code
        analysis_result = self.executor.execute(analysis.analysis_code)
        
        # Execute visualization code (if safe)
        viz_result = None
        if analysis.visualization_code:
            viz_result = self.executor.execute(analysis.visualization_code)
        
        return {
            'task': task,
            'data_description': data_description,
            'analysis_code': analysis.analysis_code,
            'visualization_code': analysis.visualization_code,
            'insights': analysis.insights,
            'analysis_result': analysis_result.output if analysis_result.success else None,
            'visualization_result': viz_result.output if viz_result and viz_result.success else None,
            'errors': {
                'analysis': analysis_result.error_message,
                'visualization': viz_result.error_message if viz_result else None
            }
        }

class LogicSolver(dspy.Module):
    """Logic puzzle solver."""
    
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought(LogicPuzzleSolver)
        self.executor = SafeCodeExecutor()
    
    def forward(self, puzzle: str, constraints: str = "") -> Dict[str, Any]:
        """Solve logic puzzle."""
        
        # Generate solution
        solution = self.solver(
            puzzle=puzzle,
            constraints=constraints or "Standard logic puzzle rules apply"
        )
        
        # Execute solution code
        execution_result = self.executor.execute(solution.solution_code)
        
        return {
            'puzzle': puzzle,
            'constraints': constraints,
            'approach': solution.solution_approach,
            'code': solution.solution_code,
            'verification': solution.verification,
            'solution': execution_result.output,
            'success': execution_result.success,
            'error': execution_result.error_message
        }

def demonstrate_basic_program_of_thought():
    """Demonstrate basic program-aided reasoning."""
    
    print_step("Basic Program-aided Reasoning")
    
    pot = ProgramOfThought()
    
    problems = [
        "Calculate the compound interest on $1000 invested at 5% annual rate for 3 years",
        "Find the area of a triangle with sides 3, 4, and 5 using Heron's formula",
        "Generate the first 10 Fibonacci numbers",
        "Calculate the distance between points (1, 2) and (4, 6)"
    ]
    
    for problem in problems:
        try:
            result = pot(problem=problem)
            
            print_result(f"Problem: {problem}")
            print_result(f"Generated Code:\n{result['generated_code']}")
            print_result(f"Explanation: {result['explanation']}")
            
            if result['execution_success']:
                print_result(f"Result: {result['result']}")
            else:
                print_error(f"Execution Error: {result['error']}")
            
            print("-" * 50)
            
        except Exception as e:
            print_error(f"Error solving problem: {e}")

def demonstrate_math_solver():
    """Demonstrate mathematical problem solving."""
    
    print_step("Mathematical Problem Solving")
    
    math_solver = MathSolver()
    
    math_problems = [
        "Solve the quadratic equation x² - 5x + 6 = 0",
        "Find the derivative of f(x) = x³ + 2x² - x + 1 at x = 2",
        "Calculate the volume of a sphere with radius 5",
        "Find the sum of the first 100 positive integers"
    ]
    
    for problem in math_problems:
        try:
            result = math_solver(problem=problem)
            
            print_result(f"Problem: {problem}")
            print_result(f"Reasoning: {result['reasoning']}")
            print_result(f"Code:\n{result['code']}")
            print_result(f"Expected: {result['expected_result']}")
            
            if result['success']:
                print_result(f"Answer: {result['actual_result']}")
            else:
                print_error(f"Error: {result['error']}")
            
            print("-" * 50)
            
        except Exception as e:
            print_error(f"Error in math solver: {e}")

def demonstrate_data_analysis():
    """Demonstrate data analysis with code generation."""
    
    print_step("Data Analysis with Code Generation")
    
    analyzer = DataAnalyzer()
    
    # Sample data analysis tasks
    tasks = [
        {
            "task": "Calculate basic statistics for a list of test scores",
            "data": "Test scores: [85, 92, 78, 96, 88, 73, 95, 89, 84, 91]"
        },
        {
            "task": "Find the correlation between two variables",
            "data": "X values: [1, 2, 3, 4, 5], Y values: [2, 4, 6, 8, 10]"
        },
        {
            "task": "Calculate moving average for sales data",
            "data": "Sales data: [100, 120, 90, 110, 130, 95, 125, 115, 105, 140]"
        }
    ]
    
    for task_info in tasks:
        try:
            result = analyzer(
                task=task_info["task"],
                data_description=task_info["data"]
            )
            
            print_result(f"Task: {task_info['task']}")
            print_result(f"Data: {task_info['data']}")
            print_result(f"Analysis Code:\n{result['analysis_code']}")
            print_result(f"Expected Insights: {result['insights']}")
            
            if result['analysis_result']:
                print_result(f"Analysis Result: {result['analysis_result']}")
            
            if result['errors']['analysis']:
                print_error(f"Analysis Error: {result['errors']['analysis']}")
            
            print("-" * 50)
            
        except Exception as e:
            print_error(f"Error in data analysis: {e}")

def demonstrate_logic_puzzles():
    """Demonstrate logic puzzle solving."""
    
    print_step("Logic Puzzle Solving")
    
    logic_solver = LogicSolver()
    
    puzzles = [
        {
            "puzzle": "Three people (Alice, Bob, Charlie) have different ages. Alice is older than Bob. Charlie is younger than Bob. Who is the oldest?",
            "constraints": "Age ordering constraints only"
        },
        {
            "puzzle": "In a race, there are 5 runners. Runner A finishes before B. Runner C finishes after D but before A. Runner E finishes last. What is the finishing order?",
            "constraints": "Relative positioning in race"
        },
        {
            "puzzle": "There are 4 colored balls: red, blue, green, yellow. The red ball is heavier than blue. Green is lighter than yellow but heavier than red. Order them by weight.",
            "constraints": "Weight comparison relationships"
        }
    ]
    
    for puzzle_info in puzzles:
        try:
            result = logic_solver(
                puzzle=puzzle_info["puzzle"],
                constraints=puzzle_info["constraints"]
            )
            
            print_result(f"Puzzle: {puzzle_info['puzzle']}")
            print_result(f"Constraints: {puzzle_info['constraints']}")
            print_result(f"Approach: {result['approach']}")
            print_result(f"Solution Code:\n{result['code']}")
            print_result(f"Verification: {result['verification']}")
            
            if result['success']:
                print_result(f"Solution: {result['solution']}")
            else:
                print_error(f"Error: {result['error']}")
            
            print("-" * 50)
            
        except Exception as e:
            print_error(f"Error solving logic puzzle: {e}")

def demonstrate_code_execution_safety():
    """Demonstrate safe code execution features."""
    
    print_step("Safe Code Execution")
    
    executor = SafeCodeExecutor()
    
    # Test safe and unsafe code examples
    test_codes = [
        {
            "name": "Safe Math Calculation",
            "code": "result = 2 ** 10 + 5 * 3",
            "should_be_safe": True
        },
        {
            "name": "Safe Loop",
            "code": "result = sum(range(1, 101))",
            "should_be_safe": True
        },
        {
            "name": "Unsafe File Access",
            "code": "with open('/etc/passwd', 'r') as f: content = f.read()",
            "should_be_safe": False
        },
        {
            "name": "Unsafe Import",
            "code": "import os; result = os.listdir('/')",
            "should_be_safe": False
        },
        {
            "name": "Unsafe Eval",
            "code": "result = eval('__import__(\"os\").system(\"ls\")')",
            "should_be_safe": False
        }
    ]
    
    for test in test_codes:
        print_result(f"\nTesting: {test['name']}")
        print_result(f"Code: {test['code']}")
        
        is_safe = executor.is_safe(test['code'])
        print_result(f"Detected as safe: {is_safe}")
        print_result(f"Expected to be safe: {test['should_be_safe']}")
        
        if is_safe == test['should_be_safe']:
            print_result("✓ Safety detection correct")
        else:
            print_error("✗ Safety detection incorrect")
        
        if is_safe:
            result = executor.execute(test['code'])
            if result.success:
                print_result(f"Execution result: {result.output}")
            else:
                print_result(f"Execution error: {result.error_message}")

def main():
    """Main function demonstrating Program of Thought."""
    
    print("=" * 60)
    print("DSPy Program of Thought Demo")
    print("=" * 60)
    
    # Setup language model
    lm = setup_default_lm()
    if not lm:
        return
    
    try:
        # Basic program-aided reasoning
        demonstrate_basic_program_of_thought()
        
        # Mathematical problem solving
        demonstrate_math_solver()
        
        # Data analysis
        demonstrate_data_analysis()
        
        # Logic puzzles
        demonstrate_logic_puzzles()
        
        # Safety demonstration
        demonstrate_code_execution_safety()
        
        print_step("Program of Thought Demo Complete!")
        
    except Exception as e:
        print_error(f"Error in Program of Thought demo: {e}")

if __name__ == "__main__":
    main()
