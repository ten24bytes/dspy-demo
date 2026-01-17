#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Integration with DSPy

This script demonstrates how to integrate DSPy with the Model Context Protocol (MCP)
for enhanced context management and model interoperability.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

# MCP Protocol Data Structures
@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None

@dataclass
class MCPContext:
    """Represents MCP context information."""
    session_id: str
    timestamp: str
    resources: List[MCPResource]
    tools: List[MCPTool]
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

def main():
    """Main function demonstrating MCP integration with DSPy."""
    print("=" * 70)
    print("MODEL CONTEXT PROTOCOL (MCP) INTEGRATION WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for MCP integration")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # DSPy Signatures for MCP Integration
    class ContextAnalysis(dspy.Signature):
        """Analyze MCP context to understand available resources and capabilities."""
        context_info = dspy.InputField(desc="MCP context information including resources and tools")
        user_request = dspy.InputField(desc="User's request or query")
        
        relevant_resources = dspy.OutputField(desc="List of relevant resources for the request")
        applicable_tools = dspy.OutputField(desc="Tools that can be used to fulfill the request")
        execution_strategy = dspy.OutputField(desc="Strategy for using resources and tools")
        context_gaps = dspy.OutputField(desc="Any missing context or resources needed")
    
    class ResourceUtilization(dspy.Signature):
        """Determine how to best utilize available MCP resources."""
        available_resources = dspy.InputField(desc="Available MCP resources with descriptions")
        task_requirements = dspy.InputField(desc="Requirements for the current task")
        
        resource_selection = dspy.OutputField(desc="Selected resources and how to use them")
        access_strategy = dspy.OutputField(desc="Strategy for accessing and processing resources")
        expected_outcomes = dspy.OutputField(desc="Expected outcomes from resource utilization")
    
    class ToolOrchestration(dspy.Signature):
        """Orchestrate MCP tools to complete complex tasks."""
        available_tools = dspy.InputField(desc="Available MCP tools and their capabilities")
        task_description = dspy.InputField(desc="Description of the task to be completed")
        context_state = dspy.InputField(desc="Current context state and available data")
        
        tool_sequence = dspy.OutputField(desc="Sequence of tools to execute")
        data_flow = dspy.OutputField(desc="How data flows between tools")
        coordination_strategy = dspy.OutputField(desc="Strategy for coordinating tool execution")
    
    class ContextSynthesis(dspy.Signature):
        """Synthesize information from multiple MCP sources."""
        source_data = dspy.InputField(desc="Data from multiple MCP resources and tools")
        synthesis_goal = dspy.InputField(desc="Goal for synthesizing the information")
        
        synthesized_result = dspy.OutputField(desc="Unified result from all sources")
        confidence_assessment = dspy.OutputField(desc="Assessment of result confidence and reliability")
        source_attribution = dspy.OutputField(desc="Attribution of information to specific sources")
    
    # MCP Client Implementation
    class MCPClient:
        """Simulated MCP client for demonstration purposes."""
        
        def __init__(self):
            self.session_id = f"mcp_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.resources = self._initialize_resources()
            self.tools = self._initialize_tools()
            self.conversation_history = []
        
        def _initialize_resources(self) -> List[MCPResource]:
            """Initialize sample MCP resources."""
            return [
                MCPResource(
                    uri="mcp://documents/company_policy",
                    name="Company Policy Document",
                    description="Internal company policies and procedures",
                    mime_type="application/json",
                    content=json.dumps({
                        "vacation_policy": "Employees get 20 days annually",
                        "remote_work": "Hybrid model with 3 days in office",
                        "expense_policy": "Pre-approval required for expenses over $100"
                    }),
                    metadata={"last_updated": "2024-01-15", "version": "2.1"}
                ),
                MCPResource(
                    uri="mcp://data/sales_report",
                    name="Q4 Sales Report",
                    description="Quarterly sales performance data",
                    mime_type="application/json",
                    content=json.dumps({
                        "total_revenue": 2500000,
                        "growth_rate": 15.2,
                        "top_products": ["Product A", "Product B", "Product C"],
                        "regions": {
                            "North": 1000000,
                            "South": 800000,
                            "East": 400000,
                            "West": 300000
                        }
                    }),
                    metadata={"period": "Q4 2023", "currency": "USD"}
                ),
                MCPResource(
                    uri="mcp://knowledge/faq",
                    name="Customer FAQ Database",
                    description="Frequently asked questions and answers",
                    mime_type="application/json",
                    content=json.dumps({
                        "shipping": "We offer free shipping on orders over $50",
                        "returns": "30-day return policy for unused items",
                        "support": "24/7 customer support via chat and email",
                        "warranty": "1-year warranty on all electronic products"
                    }),
                    metadata={"last_reviewed": "2024-01-10"}
                ),
                MCPResource(
                    uri="mcp://api/weather_data",
                    name="Weather Data API",
                    description="Current weather and forecast information",
                    mime_type="application/json",
                    content=json.dumps({
                        "current": {"temperature": 72, "condition": "sunny", "humidity": 45},
                        "forecast": [
                            {"day": "tomorrow", "high": 75, "low": 58, "condition": "partly cloudy"},
                            {"day": "day_after", "high": 68, "low": 52, "condition": "rainy"}
                        ]
                    }),
                    metadata={"location": "San Francisco", "last_updated": datetime.now().isoformat()}
                )
            ]
        
        def _initialize_tools(self) -> List[MCPTool]:
            """Initialize sample MCP tools."""
            return [
                MCPTool(
                    name="data_analyzer",
                    description="Analyze numerical data and generate insights",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "items": {"type": "number"}},
                            "analysis_type": {"type": "string", "enum": ["descriptive", "trend", "comparison"]}
                        },
                        "required": ["data", "analysis_type"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "insights": {"type": "array", "items": {"type": "string"}},
                            "metrics": {"type": "object"}
                        }
                    }
                ),
                MCPTool(
                    name="document_search",
                    description="Search through document collections",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "document_types": {"type": "array", "items": {"type": "string"}},
                            "max_results": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                ),
                MCPTool(
                    name="notification_sender",
                    description="Send notifications to users or systems",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string"},
                            "message": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                            "delivery_method": {"type": "string", "enum": ["email", "sms", "push"]}
                        },
                        "required": ["recipient", "message"]
                    }
                ),
                MCPTool(
                    name="report_generator",
                    description="Generate formatted reports from data",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "data_sources": {"type": "array", "items": {"type": "string"}},
                            "report_type": {"type": "string", "enum": ["summary", "detailed", "executive"]},
                            "format": {"type": "string", "enum": ["pdf", "html", "json"]}
                        },
                        "required": ["data_sources", "report_type"]
                    }
                )
            ]
        
        def get_context(self) -> MCPContext:
            """Get current MCP context."""
            return MCPContext(
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                resources=self.resources,
                tools=self.tools,
                conversation_history=self.conversation_history,
                metadata={
                    "client_version": "1.0.0",
                    "protocol_version": "1.0",
                    "capabilities": ["resource_access", "tool_execution", "context_management"]
                }
            )
        
        def get_resource(self, uri: str) -> Optional[MCPResource]:
            """Get a specific resource by URI."""
            return next((r for r in self.resources if r.uri == uri), None)
        
        def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Execute an MCP tool with given parameters."""
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                return {"error": f"Tool '{tool_name}' not found"}
            
            # Simulate tool execution
            if tool_name == "data_analyzer":
                data = parameters.get("data", [])
                analysis_type = parameters.get("analysis_type", "descriptive")
                
                if analysis_type == "descriptive":
                    return {
                        "summary": f"Analysis of {len(data)} data points",
                        "insights": [
                            f"Average: {sum(data)/len(data):.2f}" if data else "No data",
                            f"Range: {max(data) - min(data):.2f}" if data else "No range",
                            "Data shows normal distribution pattern"
                        ],
                        "metrics": {"count": len(data), "sum": sum(data)}
                    }
                
            elif tool_name == "document_search":
                query = parameters.get("query", "")
                return {
                    "results": [
                        {"title": f"Document about {query}", "relevance": 0.95, "uri": "mcp://docs/result1"},
                        {"title": f"Related information on {query}", "relevance": 0.87, "uri": "mcp://docs/result2"}
                    ],
                    "total_found": 2
                }
            
            elif tool_name == "notification_sender":
                return {
                    "status": "sent",
                    "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "delivery_time": datetime.now().isoformat()
                }
            
            elif tool_name == "report_generator":
                return {
                    "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "status": "generated",
                    "download_url": "mcp://reports/latest_report.pdf",
                    "size": "1.2MB"
                }
            
            return {"result": "Tool executed successfully"}
        
        def add_to_history(self, interaction: Dict[str, Any]):
            """Add interaction to conversation history."""
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                **interaction
            })
    
    # DSPy Module for MCP Integration
    class MCPIntegrator(dspy.Module):
        """DSPy module that integrates with MCP for enhanced context management."""
        
        def __init__(self, mcp_client: MCPClient):
            super().__init__()
            self.mcp_client = mcp_client
            self.context_analyzer = dspy.ChainOfThought(ContextAnalysis)
            self.resource_utilizer = dspy.ChainOfThought(ResourceUtilization)
            self.tool_orchestrator = dspy.ChainOfThought(ToolOrchestration)
            self.context_synthesizer = dspy.ChainOfThought(ContextSynthesis)
        
        def analyze_context(self, user_request: str) -> dspy.Prediction:
            """Analyze MCP context for a user request."""
            context = self.mcp_client.get_context()
            
            context_info = f"""
            Available Resources:
            {[f"{r.name}: {r.description}" for r in context.resources]}
            
            Available Tools:
            {[f"{t.name}: {t.description}" for t in context.tools]}
            
            Session ID: {context.session_id}
            History Length: {len(context.conversation_history)}
            """
            
            return self.context_analyzer(
                context_info=context_info,
                user_request=user_request
            )
        
        def utilize_resources(self, task_requirements: str) -> dspy.Prediction:
            """Determine how to best utilize available resources."""
            context = self.mcp_client.get_context()
            
            resources_info = "\n".join([
                f"{r.name} ({r.uri}): {r.description} - {r.mime_type}"
                for r in context.resources
            ])
            
            return self.resource_utilizer(
                available_resources=resources_info,
                task_requirements=task_requirements
            )
        
        def orchestrate_tools(self, task_description: str) -> dspy.Prediction:
            """Orchestrate MCP tools to complete a task."""
            context = self.mcp_client.get_context()
            
            tools_info = "\n".join([
                f"{t.name}: {t.description}\nInput: {t.input_schema}"
                for t in context.tools
            ])
            
            context_state = f"Session: {context.session_id}, Resources: {len(context.resources)}, History: {len(context.conversation_history)}"
            
            return self.tool_orchestrator(
                available_tools=tools_info,
                task_description=task_description,
                context_state=context_state
            )
        
        def synthesize_context(self, source_data: List[str], goal: str) -> dspy.Prediction:
            """Synthesize information from multiple MCP sources."""
            combined_data = "\n---\n".join(source_data)
            
            return self.context_synthesizer(
                source_data=combined_data,
                synthesis_goal=goal
            )
        
        def process_request(self, user_request: str) -> Dict[str, Any]:
            """Process a complete user request using MCP integration."""
            # Step 1: Analyze context
            analysis = self.analyze_context(user_request)
            
            # Step 2: Plan resource utilization
            resource_plan = self.utilize_resources(user_request)
            
            # Step 3: Orchestrate tools if needed
            tool_plan = self.orchestrate_tools(user_request)
            
            # Step 4: Execute planned actions
            execution_results = []
            
            # Simulate resource access
            for resource in self.mcp_client.resources:
                if any(keyword in user_request.lower() for keyword in [
                    resource.name.lower(), 
                    resource.description.lower().split()
                ]):
                    execution_results.append(f"Accessed: {resource.name} - {resource.content[:100]}...")
            
            # Simulate tool execution based on request
            if "analyze" in user_request.lower() or "data" in user_request.lower():
                tool_result = self.mcp_client.execute_tool("data_analyzer", {
                    "data": [100, 150, 200, 175, 225],
                    "analysis_type": "descriptive"
                })
                execution_results.append(f"Data Analysis: {tool_result}")
            
            if "search" in user_request.lower() or "find" in user_request.lower():
                tool_result = self.mcp_client.execute_tool("document_search", {
                    "query": user_request,
                    "max_results": 5
                })
                execution_results.append(f"Search Results: {tool_result}")
            
            # Step 5: Synthesize results
            if execution_results:
                synthesis = self.synthesize_context(execution_results, f"Respond to: {user_request}")
                final_result = synthesis.synthesized_result
            else:
                final_result = f"Analyzed request but no specific actions were taken. Analysis: {analysis.execution_strategy}"
            
            # Add to conversation history
            self.mcp_client.add_to_history({
                "type": "user_request",
                "request": user_request,
                "analysis": analysis.execution_strategy,
                "resources_used": resource_plan.resource_selection,
                "tools_planned": tool_plan.tool_sequence,
                "final_result": final_result
            })
            
            return {
                "request": user_request,
                "context_analysis": analysis.execution_strategy,
                "resource_strategy": resource_plan.resource_selection,
                "tool_strategy": tool_plan.tool_sequence,
                "execution_results": execution_results,
                "final_result": final_result,
                "session_id": self.mcp_client.session_id
            }
    
    # Initialize MCP components
    mcp_client = MCPClient()
    mcp_integrator = MCPIntegrator(mcp_client)
    print_result("MCP client and integrator initialized successfully!")
    
    # Demo 1: MCP Context Overview
    print_step("MCP Context Overview", "Exploring available resources and tools")
    
    context = mcp_client.get_context()
    
    print_result(f"Session ID: {context.session_id}", "Session Information")
    print_result(f"Resources: {len(context.resources)}", "Available Resources")
    print_result(f"Tools: {len(context.tools)}", "Available Tools")
    
    print("\n--- Available Resources ---")
    for i, resource in enumerate(context.resources, 1):
        print(f"{i}. {resource.name}")
        print(f"   URI: {resource.uri}")
        print(f"   Type: {resource.mime_type}")
        print(f"   Description: {resource.description}")
        print()
    
    print("--- Available Tools ---")
    for i, tool in enumerate(context.tools, 1):
        print(f"{i}. {tool.name}")
        print(f"   Description: {tool.description}")
        print(f"   Input Schema: {tool.input_schema.get('properties', {}).keys()}")
        print()
    
    # Demo 2: Context Analysis
    print_step("Context Analysis Demo", "Analyzing user requests against MCP context")
    
    sample_requests = [
        "What is our company's vacation policy?",
        "Can you analyze our sales performance this quarter?",
        "I need to find information about our return policy",
        "Generate a summary report of our business performance",
        "What's the weather forecast for tomorrow?"
    ]
    
    for i, request in enumerate(sample_requests, 1):
        try:
            analysis = mcp_integrator.analyze_context(request)
            
            print(f"\n--- Request {i}: {request} ---")
            print_result(f"Relevant Resources: {analysis.relevant_resources}", "Resources")
            print_result(f"Applicable Tools: {analysis.applicable_tools}", "Tools")
            print_result(f"Strategy: {analysis.execution_strategy}", "Execution Strategy")
            print_result(f"Context Gaps: {analysis.context_gaps}", "Missing Context")
            
        except Exception as e:
            print_error(f"Error analyzing request {i}: {e}")
    
    # Demo 3: Resource Utilization
    print_step("Resource Utilization Demo", "Planning resource usage for tasks")
    
    task_scenarios = [
        "Create a comprehensive business report using all available data",
        "Answer customer inquiries about company policies",
        "Provide weather-based recommendations for outdoor activities",
        "Analyze sales trends and identify growth opportunities"
    ]
    
    for i, task in enumerate(task_scenarios, 1):
        try:
            resource_plan = mcp_integrator.utilize_resources(task)
            
            print(f"\n--- Task {i}: {task} ---")
            print_result(f"Resource Selection: {resource_plan.resource_selection}", "Selected Resources")
            print_result(f"Access Strategy: {resource_plan.access_strategy}", "Access Strategy")
            print_result(f"Expected Outcomes: {resource_plan.expected_outcomes}", "Expected Outcomes")
            
        except Exception as e:
            print_error(f"Error planning resources for task {i}: {e}")
    
    # Demo 4: Tool Orchestration
    print_step("Tool Orchestration Demo", "Coordinating multiple tools for complex tasks")
    
    complex_tasks = [
        "Search for sales data, analyze it, and generate a formatted report",
        "Find company policies, extract key information, and send notifications to relevant teams",
        "Analyze customer feedback data and create actionable insights report",
        "Process multiple data sources and create executive summary"
    ]
    
    for i, task in enumerate(complex_tasks, 1):
        try:
            orchestration = mcp_integrator.orchestrate_tools(task)
            
            print(f"\n--- Complex Task {i}: {task} ---")
            print_result(f"Tool Sequence: {orchestration.tool_sequence}", "Tool Sequence")
            print_result(f"Data Flow: {orchestration.data_flow}", "Data Flow")
            print_result(f"Coordination: {orchestration.coordination_strategy}", "Coordination Strategy")
            
        except Exception as e:
            print_error(f"Error orchestrating task {i}: {e}")
    
    # Demo 5: Complete Request Processing
    print_step("Complete Request Processing", "End-to-end request handling with MCP")
    
    full_requests = [
        "What is our company's remote work policy and how does it compare to industry standards?",
        "Analyze our Q4 sales performance and identify the top performing regions",
        "I need help with returns - what's the process and timeline?",
        "Create a weather-based activity recommendation for this weekend",
        "Generate a comprehensive business intelligence report for stakeholders"
    ]
    
    for i, request in enumerate(full_requests, 1):
        try:
            result = mcp_integrator.process_request(request)
            
            print(f"\n--- Complete Request {i} ---")
            print_result(f"Request: {result['request']}", "User Request")
            print_result(f"Context Analysis: {result['context_analysis']}", "Analysis")
            print_result(f"Resource Strategy: {result['resource_strategy']}", "Resources")
            print_result(f"Tool Strategy: {result['tool_strategy']}", "Tools")
            print_result(f"Final Result: {result['final_result']}", "Final Response")
            
        except Exception as e:
            print_error(f"Error processing request {i}: {e}")
    
    # Demo 6: Conversation History and Context Continuity
    print_step("Context Continuity Demo", "Demonstrating conversation history usage")
    
    # Show conversation history
    history = mcp_client.conversation_history
    
    print_result(f"Conversation History Length: {len(history)}", "History Status")
    
    if history:
        print("\n--- Recent Interactions ---")
        for i, interaction in enumerate(history[-3:], 1):  # Show last 3 interactions
            print(f"{i}. {interaction.get('timestamp', 'Unknown time')}")
            print(f"   Type: {interaction.get('type', 'Unknown')}")
            print(f"   Request: {interaction.get('request', 'N/A')[:100]}...")
            print()
    
    # Demonstrate context-aware follow-up
    follow_up_request = "Can you elaborate on the sales analysis from my previous question?"
    
    try:
        context_aware_result = mcp_integrator.process_request(follow_up_request)
        
        print("--- Context-Aware Follow-up ---")
        print_result(f"Follow-up: {follow_up_request}", "Follow-up Request")
        print_result(f"Response: {context_aware_result['final_result']}", "Context-Aware Response")
        
    except Exception as e:
        print_error(f"Error in context-aware processing: {e}")
    
    # Demo 7: MCP Resource Updates
    print_step("Dynamic Resource Management", "Adding and updating MCP resources")
    
    try:
        # Add a new resource dynamically
        new_resource = MCPResource(
            uri="mcp://data/realtime_metrics",
            name="Real-time Business Metrics",
            description="Live dashboard data with current KPIs",
            mime_type="application/json",
            content=json.dumps({
                "active_users": 1250,
                "revenue_today": 15000,
                "conversion_rate": 3.2,
                "server_uptime": "99.8%",
                "support_tickets": 23
            }),
            metadata={"update_frequency": "1_minute", "source": "dashboard_api"}
        )
        
        mcp_client.resources.append(new_resource)
        
        print_result("New resource added successfully", "Resource Update")
        print_result(f"Total resources: {len(mcp_client.resources)}", "Resource Count")
        
        # Test with the new resource
        test_request = "What are our current business metrics?"
        result = mcp_integrator.process_request(test_request)
        
        print_result(f"New Resource Test: {result['final_result'][:200]}...", "Test Result")
        
    except Exception as e:
        print_error(f"Error in resource management: {e}")
    
    print("\n" + "="*70)
    print("MCP INTEGRATION COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy integration with Model Context Protocol!")

if __name__ == "__main__":
    main()
