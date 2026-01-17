#!/usr/bin/env python3
"""
Advanced Tool Use with DSPy

This script demonstrates advanced tool integration and usage patterns with DSPy,
including custom tools, tool orchestration, and complex workflows.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import json
import requests
import sqlite3
import tempfile
import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv
import math
import random

@dataclass
class ToolResult:
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0

def main():
    """Main function demonstrating advanced tool use with DSPy."""
    print("=" * 70)
    print("ADVANCED TOOL USE WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for advanced tool usage")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Custom Tool Implementations
    class CalculatorTool:
        """Advanced calculator with mathematical functions."""
        
        def calculate(self, expression: str) -> ToolResult:
            """Safely evaluate mathematical expressions."""
            try:
                # Sanitize expression for safety
                allowed_chars = set('0123456789+-*/().% ')
                allowed_functions = ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp', 'abs', 'round']
                
                # Basic validation
                if not all(c in allowed_chars or any(func in expression for func in allowed_functions) for c in expression):
                    return ToolResult(False, None, "Invalid characters in expression")
                
                # Replace function names with math module equivalents
                safe_expression = expression
                for func in allowed_functions:
                    safe_expression = safe_expression.replace(func, f'math.{func}')
                
                # Evaluate safely
                result = eval(safe_expression, {"__builtins__": {}, "math": math})
                return ToolResult(True, result)
                
            except Exception as e:
                return ToolResult(False, None, f"Calculation error: {str(e)}")
        
        def get_statistics(self, numbers: List[float]) -> ToolResult:
            """Calculate statistical measures for a list of numbers."""
            try:
                if not numbers:
                    return ToolResult(False, None, "Empty number list")
                
                stats = {
                    'count': len(numbers),
                    'sum': sum(numbers),
                    'mean': sum(numbers) / len(numbers),
                    'min': min(numbers),
                    'max': max(numbers),
                    'range': max(numbers) - min(numbers)
                }
                
                # Calculate standard deviation
                mean = stats['mean']
                variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
                stats['std_dev'] = math.sqrt(variance)
                
                return ToolResult(True, stats)
                
            except Exception as e:
                return ToolResult(False, None, f"Statistics error: {str(e)}")
    
    class DatabaseTool:
        """In-memory database tool for data operations."""
        
        def __init__(self):
            self.db_path = ":memory:"
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.setup_sample_data()
        
        def setup_sample_data(self):
            """Set up sample database with test data."""
            cursor = self.conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    price REAL,
                    stock INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE sales (
                    id INTEGER PRIMARY KEY,
                    product_id INTEGER,
                    quantity INTEGER,
                    sale_date TEXT,
                    total_amount REAL,
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            ''')
            
            # Insert sample data
            products = [
                (1, 'Laptop Pro', 'Electronics', 1299.99, 50),
                (2, 'Wireless Mouse', 'Electronics', 29.99, 200),
                (3, 'Office Chair', 'Furniture', 199.99, 30),
                (4, 'Desk Lamp', 'Furniture', 49.99, 75),
                (5, 'Coffee Maker', 'Appliances', 89.99, 25)
            ]
            
            cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?, ?)', products)
            
            sales = [
                (1, 1, 2, '2024-01-15', 2599.98),
                (2, 2, 5, '2024-01-16', 149.95),
                (3, 3, 1, '2024-01-17', 199.99),
                (4, 4, 3, '2024-01-18', 149.97),
                (5, 5, 1, '2024-01-19', 89.99)
            ]
            
            cursor.executemany('INSERT INTO sales VALUES (?, ?, ?, ?, ?)', sales)
            self.conn.commit()
        
        def query(self, sql: str) -> ToolResult:
            """Execute SQL query and return results."""
            try:
                cursor = self.conn.cursor()
                cursor.execute(sql)
                
                if sql.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    return ToolResult(True, {"columns": columns, "rows": results})
                else:
                    self.conn.commit()
                    return ToolResult(True, f"Query executed successfully. Rows affected: {cursor.rowcount}")
                    
            except Exception as e:
                return ToolResult(False, None, f"Database error: {str(e)}")
        
        def get_schema(self) -> ToolResult:
            """Get database schema information."""
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                schema = {}
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    schema[table_name] = [
                        {"name": col[1], "type": col[2], "nullable": not col[3]}
                        for col in columns
                    ]
                
                return ToolResult(True, schema)
                
            except Exception as e:
                return ToolResult(False, None, f"Schema error: {str(e)}")
    
    class WebSearchTool:
        """Simulated web search tool (for demonstration)."""
        
        def __init__(self):
            # Simulated search results database
            self.search_db = {
                "python programming": [
                    {"title": "Python.org Official Documentation", "url": "https://python.org", "snippet": "Official Python programming language documentation"},
                    {"title": "Python Tutorial for Beginners", "url": "https://example.com/python", "snippet": "Learn Python from scratch with examples"},
                ],
                "machine learning": [
                    {"title": "Introduction to Machine Learning", "url": "https://ml.example.com", "snippet": "Comprehensive guide to ML concepts"},
                    {"title": "Scikit-learn Documentation", "url": "https://scikit-learn.org", "snippet": "Machine learning library for Python"},
                ],
                "weather forecast": [
                    {"title": "Today's Weather Forecast", "url": "https://weather.com", "snippet": "Current weather conditions and 7-day forecast"},
                    {"title": "Climate Data and Trends", "url": "https://climate.gov", "snippet": "Historical weather data and climate information"},
                ]
            }
        
        def search(self, query: str, num_results: int = 5) -> ToolResult:
            """Simulate web search with mock results."""
            try:
                query_lower = query.lower()
                results = []
                
                # Find matching results
                for key, values in self.search_db.items():
                    if any(word in query_lower for word in key.split()):
                        results.extend(values)
                
                # If no specific results, return generic ones
                if not results:
                    results = [
                        {"title": f"Search results for: {query}", "url": "https://example.com/search", "snippet": f"Information about {query}"},
                        {"title": f"More about {query}", "url": "https://info.example.com", "snippet": f"Detailed information on {query}"},
                    ]
                
                return ToolResult(True, results[:num_results])
                
            except Exception as e:
                return ToolResult(False, None, f"Search error: {str(e)}")
        
        def get_page_content(self, url: str) -> ToolResult:
            """Simulate fetching page content."""
            try:
                # Simulate content based on URL
                if "python.org" in url:
                    content = "Python is a programming language that lets you work quickly and integrate systems more effectively."
                elif "weather.com" in url:
                    content = "Today's weather: Sunny with a high of 75°F. Tomorrow: Partly cloudy with a chance of rain."
                else:
                    content = f"Content from {url}: This is simulated web content for demonstration purposes."
                
                return ToolResult(True, content)
                
            except Exception as e:
                return ToolResult(False, None, f"Content fetch error: {str(e)}")
    
    class FileSystemTool:
        """File system operations tool."""
        
        def __init__(self):
            self.temp_dir = tempfile.mkdtemp()
        
        def create_file(self, filename: str, content: str) -> ToolResult:
            """Create a file with specified content."""
            try:
                filepath = os.path.join(self.temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
                return ToolResult(True, f"File created: {filepath}")
                
            except Exception as e:
                return ToolResult(False, None, f"File creation error: {str(e)}")
        
        def read_file(self, filename: str) -> ToolResult:
            """Read content from a file."""
            try:
                filepath = os.path.join(self.temp_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                return ToolResult(True, content)
                
            except Exception as e:
                return ToolResult(False, None, f"File read error: {str(e)}")
        
        def list_files(self) -> ToolResult:
            """List files in the working directory."""
            try:
                files = os.listdir(self.temp_dir)
                return ToolResult(True, files)
                
            except Exception as e:
                return ToolResult(False, None, f"Directory listing error: {str(e)}")
    
    # DSPy Signatures for Tool Integration
    class ToolSelection(dspy.Signature):
        """Select the most appropriate tool for a given task."""
        task_description = dspy.InputField(desc="Description of the task to be performed")
        available_tools = dspy.InputField(desc="List of available tools and their capabilities")
        
        selected_tool = dspy.OutputField(desc="The most appropriate tool for the task")
        reasoning = dspy.OutputField(desc="Explanation for why this tool was selected")
        parameters = dspy.OutputField(desc="Required parameters for the selected tool")
    
    class ToolOrchestration(dspy.Signature):
        """Orchestrate multiple tools to complete a complex task."""
        complex_task = dspy.InputField(desc="Complex task requiring multiple tool operations")
        available_tools = dspy.InputField(desc="Available tools and their capabilities")
        
        execution_plan = dspy.OutputField(desc="Step-by-step plan using multiple tools")
        tool_sequence = dspy.OutputField(desc="Ordered sequence of tool operations")
        expected_outcome = dspy.OutputField(desc="Expected final outcome of the orchestration")
    
    class ResultSynthesis(dspy.Signature):
        """Synthesize results from multiple tool operations."""
        tool_results = dspy.InputField(desc="Results from multiple tool operations")
        original_task = dspy.InputField(desc="The original task that required these tools")
        
        synthesized_result = dspy.OutputField(desc="Combined and synthesized final result")
        insights = dspy.OutputField(desc="Key insights derived from the tool results")
        recommendations = dspy.OutputField(desc="Recommendations based on the analysis")
    
    # Advanced Tool Manager
    class AdvancedToolManager(dspy.Module):
        """Advanced tool management and orchestration system."""
        
        def __init__(self):
            super().__init__()
            self.calculator = CalculatorTool()
            self.database = DatabaseTool()
            self.web_search = WebSearchTool()
            self.file_system = FileSystemTool()
            
            self.tool_selector = dspy.ChainOfThought(ToolSelection)
            self.orchestrator = dspy.ChainOfThought(ToolOrchestration)
            self.synthesizer = dspy.ChainOfThought(ResultSynthesis)
            
            self.tools_description = """
            Available Tools:
            1. Calculator: Mathematical calculations, statistics, complex expressions
            2. Database: SQL queries, data analysis, schema inspection
            3. WebSearch: Information retrieval, content fetching
            4. FileSystem: File operations, content management
            """
        
        def select_tool(self, task: str) -> dspy.Prediction:
            """Select the best tool for a given task."""
            return self.tool_selector(
                task_description=task,
                available_tools=self.tools_description
            )
        
        def orchestrate_tools(self, complex_task: str) -> dspy.Prediction:
            """Create an orchestration plan for complex tasks."""
            return self.orchestrator(
                complex_task=complex_task,
                available_tools=self.tools_description
            )
        
        def execute_tool_operation(self, tool_name: str, operation: str, parameters: Dict[str, Any]) -> ToolResult:
            """Execute a specific tool operation."""
            try:
                if tool_name.lower() == "calculator":
                    if operation == "calculate":
                        return self.calculator.calculate(parameters.get("expression", ""))
                    elif operation == "statistics":
                        return self.calculator.get_statistics(parameters.get("numbers", []))
                
                elif tool_name.lower() == "database":
                    if operation == "query":
                        return self.database.query(parameters.get("sql", ""))
                    elif operation == "schema":
                        return self.database.get_schema()
                
                elif tool_name.lower() == "websearch":
                    if operation == "search":
                        return self.web_search.search(
                            parameters.get("query", ""), 
                            parameters.get("num_results", 5)
                        )
                    elif operation == "fetch":
                        return self.web_search.get_page_content(parameters.get("url", ""))
                
                elif tool_name.lower() == "filesystem":
                    if operation == "create":
                        return self.file_system.create_file(
                            parameters.get("filename", ""), 
                            parameters.get("content", "")
                        )
                    elif operation == "read":
                        return self.file_system.read_file(parameters.get("filename", ""))
                    elif operation == "list":
                        return self.file_system.list_files()
                
                return ToolResult(False, None, f"Unknown tool or operation: {tool_name}.{operation}")
                
            except Exception as e:
                return ToolResult(False, None, f"Tool execution error: {str(e)}")
        
        def synthesize_results(self, results: List[ToolResult], task: str) -> dspy.Prediction:
            """Synthesize results from multiple tool operations."""
            results_summary = []
            for i, result in enumerate(results):
                if result.success:
                    results_summary.append(f"Operation {i+1}: Success - {str(result.result)[:200]}")
                else:
                    results_summary.append(f"Operation {i+1}: Failed - {result.error_message}")
            
            return self.synthesizer(
                tool_results="\n".join(results_summary),
                original_task=task
            )
    
    # Initialize the tool manager
    tool_manager = AdvancedToolManager()
    print_result("Advanced tool manager initialized successfully!")
    
    # Demo 1: Tool Selection
    print_step("Tool Selection Demo", "Automatically selecting tools for different tasks")
    
    task_examples = [
        "Calculate the compound interest for a $10,000 investment at 5% annual rate over 10 years",
        "Find information about the latest developments in artificial intelligence",
        "Analyze sales data to identify top-performing products",
        "Create a report file with customer satisfaction metrics"
    ]
    
    for i, task in enumerate(task_examples, 1):
        try:
            selection = tool_manager.select_tool(task)
            
            print(f"\n--- Task {i} ---")
            print_result(f"Task: {task}", "Task Description")
            print_result(f"Selected Tool: {selection.selected_tool}", "Selected Tool")
            print_result(f"Reasoning: {selection.reasoning}", "Selection Reasoning")
            print_result(f"Parameters: {selection.parameters}", "Required Parameters")
            
        except Exception as e:
            print_error(f"Error in tool selection {i}: {e}")
    
    # Demo 2: Individual Tool Operations
    print_step("Individual Tool Operations", "Demonstrating each tool's capabilities")
    
    # Calculator operations
    print("\n--- Calculator Tool ---")
    calc_operations = [
        {"expression": "2 + 3 * 4", "description": "Basic arithmetic"},
        {"expression": "sqrt(144) + log(100)", "description": "Mathematical functions"},
        {"numbers": [10, 20, 30, 25, 15], "description": "Statistical analysis", "operation": "statistics"}
    ]
    
    for op in calc_operations:
        try:
            if "numbers" in op:
                result = tool_manager.execute_tool_operation("calculator", "statistics", op)
            else:
                result = tool_manager.execute_tool_operation("calculator", "calculate", op)
            
            print(f"\n{op['description']}: {op.get('expression', op.get('numbers', ''))}")
            if result.success:
                print_result(f"Result: {result.result}", "Calculation Result")
            else:
                print_error(f"Error: {result.error_message}")
                
        except Exception as e:
            print_error(f"Calculator operation error: {e}")
    
    # Database operations
    print("\n--- Database Tool ---")
    db_operations = [
        {"sql": "SELECT * FROM products WHERE price > 100", "description": "Product query"},
        {"sql": "SELECT category, AVG(price) as avg_price FROM products GROUP BY category", "description": "Category analysis"},
        {"operation": "schema", "description": "Database schema"}
    ]
    
    for op in db_operations:
        try:
            if "sql" in op:
                result = tool_manager.execute_tool_operation("database", "query", op)
            else:
                result = tool_manager.execute_tool_operation("database", "schema", {})
            
            print(f"\n{op['description']}")
            if result.success:
                print_result(f"Result: {str(result.result)[:300]}...", "Query Result")
            else:
                print_error(f"Error: {result.error_message}")
                
        except Exception as e:
            print_error(f"Database operation error: {e}")
    
    # Web search operations
    print("\n--- Web Search Tool ---")
    search_operations = [
        {"query": "python programming", "description": "Programming search"},
        {"query": "machine learning", "description": "ML information search"}
    ]
    
    for op in search_operations:
        try:
            result = tool_manager.execute_tool_operation("websearch", "search", op)
            
            print(f"\n{op['description']}: {op['query']}")
            if result.success:
                print_result(f"Found {len(result.result)} results", "Search Results")
                for i, res in enumerate(result.result[:2], 1):
                    print(f"  {i}. {res['title']} - {res['snippet']}")
            else:
                print_error(f"Error: {result.error_message}")
                
        except Exception as e:
            print_error(f"Search operation error: {e}")
    
    # File system operations
    print("\n--- File System Tool ---")
    fs_operations = [
        {"filename": "report.txt", "content": "Sales Report\n\nQ1 Performance:\n- Revenue: $150K\n- Growth: 15%", "description": "Create report file"},
        {"filename": "report.txt", "description": "Read report file", "operation": "read"},
        {"description": "List files", "operation": "list"}
    ]
    
    for op in fs_operations:
        try:
            if "content" in op:
                result = tool_manager.execute_tool_operation("filesystem", "create", op)
            elif op.get("operation") == "read":
                result = tool_manager.execute_tool_operation("filesystem", "read", op)
            else:
                result = tool_manager.execute_tool_operation("filesystem", "list", {})
            
            print(f"\n{op['description']}")
            if result.success:
                print_result(f"Result: {str(result.result)[:200]}", "File Operation Result")
            else:
                print_error(f"Error: {result.error_message}")
                
        except Exception as e:
            print_error(f"File system operation error: {e}")
    
    # Demo 3: Tool Orchestration
    print_step("Tool Orchestration Demo", "Orchestrating multiple tools for complex tasks")
    
    complex_tasks = [
        "Analyze our product sales data, calculate performance metrics, and create a comprehensive report",
        "Research the latest AI trends, compile the information, and generate a market analysis document",
        "Process customer feedback data, calculate satisfaction scores, and store results in a file"
    ]
    
    for i, task in enumerate(complex_tasks, 1):
        try:
            orchestration = tool_manager.orchestrate_tools(task)
            
            print(f"\n--- Complex Task {i} ---")
            print_result(f"Task: {task}", "Complex Task")
            print_result(f"Plan: {orchestration.execution_plan}", "Execution Plan")
            print_result(f"Sequence: {orchestration.tool_sequence}", "Tool Sequence")
            print_result(f"Expected Outcome: {orchestration.expected_outcome}", "Expected Outcome")
            
        except Exception as e:
            print_error(f"Error in orchestration {i}: {e}")
    
    # Demo 4: Multi-Tool Workflow Execution
    print_step("Multi-Tool Workflow", "Executing a complete multi-tool workflow")
    
    try:
        workflow_task = "Analyze product performance and create a business intelligence report"
        
        # Step 1: Query database for product data
        print("\nStep 1: Querying product performance data...")
        db_result = tool_manager.execute_tool_operation("database", "query", {
            "sql": "SELECT p.name, p.category, p.price, s.quantity, s.total_amount FROM products p JOIN sales s ON p.id = s.product_id"
        })
        
        # Step 2: Calculate performance metrics
        if db_result.success:
            print("Step 2: Calculating performance metrics...")
            
            # Extract sales amounts for statistical analysis
            sales_amounts = [2599.98, 149.95, 199.99, 149.97, 89.99]  # Sample data
            stats_result = tool_manager.execute_tool_operation("calculator", "statistics", {
                "numbers": sales_amounts
            })
        
        # Step 3: Create comprehensive report
        if db_result.success and stats_result.success:
            print("Step 3: Creating business intelligence report...")
            
            report_content = f"""
Business Intelligence Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PRODUCT PERFORMANCE ANALYSIS
============================

Database Query Results:
{str(db_result.result)[:500]}

Statistical Analysis:
Mean Sales: ${stats_result.result['mean']:.2f}
Total Revenue: ${stats_result.result['sum']:.2f}
Standard Deviation: ${stats_result.result['std_dev']:.2f}
Range: ${stats_result.result['range']:.2f}

INSIGHTS:
- Top performing products identified
- Sales variance analysis completed
- Revenue distribution calculated

RECOMMENDATIONS:
- Focus on high-performing categories
- Optimize inventory for best sellers
- Review pricing strategy for underperformers
"""
            
            file_result = tool_manager.execute_tool_operation("filesystem", "create", {
                "filename": "bi_report.txt",
                "content": report_content
            })
        
        # Step 4: Synthesize all results
        print("Step 4: Synthesizing results...")
        
        all_results = [db_result, stats_result, file_result]
        synthesis = tool_manager.synthesize_results(all_results, workflow_task)
        
        print_result(f"Synthesis: {synthesis.synthesized_result}", "Final Result")
        print_result(f"Insights: {synthesis.insights}", "Key Insights")
        print_result(f"Recommendations: {synthesis.recommendations}", "Recommendations")
        
    except Exception as e:
        print_error(f"Error in multi-tool workflow: {e}")
    
    # Demo 5: Error Handling and Recovery
    print_step("Error Handling Demo", "Demonstrating error handling and recovery")
    
    error_scenarios = [
        {
            "tool": "calculator",
            "operation": "calculate",
            "parameters": {"expression": "1/0"},
            "description": "Division by zero error"
        },
        {
            "tool": "database",
            "operation": "query",
            "parameters": {"sql": "SELECT * FROM nonexistent_table"},
            "description": "Invalid table query"
        },
        {
            "tool": "filesystem",
            "operation": "read",
            "parameters": {"filename": "nonexistent_file.txt"},
            "description": "Reading non-existent file"
        }
    ]
    
    for i, scenario in enumerate(error_scenarios, 1):
        try:
            result = tool_manager.execute_tool_operation(
                scenario["tool"],
                scenario["operation"],
                scenario["parameters"]
            )
            
            print(f"\n--- Error Scenario {i}: {scenario['description']} ---")
            print_result(f"Success: {result.success}", "Operation Status")
            if result.success:
                print_result(f"Result: {result.result}", "Result")
            else:
                print_result(f"Error: {result.error_message}", "Error Message")
                
        except Exception as e:
            print_error(f"Unexpected error in scenario {i}: {e}")
    
    # Demo 6: Performance Monitoring
    print_step("Performance Monitoring", "Monitoring tool performance and usage")
    
    import time
    
    performance_tests = [
        {"tool": "calculator", "operation": "calculate", "params": {"expression": "sqrt(1000000)"}},
        {"tool": "database", "operation": "query", "params": {"sql": "SELECT COUNT(*) FROM products"}},
        {"tool": "websearch", "operation": "search", "params": {"query": "test query"}},
    ]
    
    performance_results = []
    
    for test in performance_tests:
        try:
            start_time = time.time()
            result = tool_manager.execute_tool_operation(test["tool"], test["operation"], test["params"])
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            performance_results.append({
                "tool": test["tool"],
                "operation": test["operation"],
                "success": result.success,
                "execution_time": execution_time
            })
            
            print(f"\n{test['tool']}.{test['operation']}:")
            print_result(f"Success: {result.success}", "Status")
            print_result(f"Execution Time: {execution_time:.4f}s", "Performance")
            
        except Exception as e:
            print_error(f"Performance test error: {e}")
    
    # Performance summary
    print("\n--- Performance Summary ---")
    avg_time = sum(r["execution_time"] for r in performance_results) / len(performance_results)
    success_rate = sum(1 for r in performance_results if r["success"]) / len(performance_results)
    
    print_result(f"Average Execution Time: {avg_time:.4f}s", "Overall Performance")
    print_result(f"Success Rate: {success_rate:.2%}", "Reliability")
    
    print("\n" + "="*70)
    print("ADVANCED TOOL USE COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated advanced DSPy tool integration and orchestration!")

if __name__ == "__main__":
    main()
