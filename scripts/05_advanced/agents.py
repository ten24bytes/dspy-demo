#!/usr/bin/env python3
"""
Agents with DSPy

This script demonstrates how to build intelligent agents using DSPy.
It covers different agent architectures, memory systems, and tool integration.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import json
import time
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from utils import setup_default_lm, print_step, print_result, print_error

@dataclass
class AgentMemory:
    """Agent memory for storing experiences and knowledge."""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: List[Dict[str, Any]] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    max_short_term: int = 10
    max_long_term: int = 100

@dataclass
class AgentGoal:
    """Represents an agent goal."""
    description: str
    priority: int
    deadline: Optional[datetime] = None
    completed: bool = False
    progress: float = 0.0

@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    action_type: str
    parameters: Dict[str, Any]
    timestamp: datetime
    result: Optional[Any] = None
    success: bool = True

class AgentPlanning(dspy.Signature):
    """Plan actions to achieve a goal."""
    
    goal = dspy.InputField(desc="The goal to achieve")
    current_state = dspy.InputField(desc="Current state and available information")
    memory = dspy.InputField(desc="Relevant past experiences and knowledge")
    plan = dspy.OutputField(desc="Step-by-step plan to achieve the goal")
    next_action = dspy.OutputField(desc="The immediate next action to take")

class AgentReasoning(dspy.Signature):
    """Reason about the current situation and decide on actions."""
    
    situation = dspy.InputField(desc="Current situation description")
    goal = dspy.InputField(desc="Current goal or objective")
    available_actions = dspy.InputField(desc="List of available actions")
    memory = dspy.InputField(desc="Relevant memory and context")
    reasoning = dspy.OutputField(desc="Reasoning about the situation")
    decision = dspy.OutputField(desc="Decision on what action to take")
    confidence = dspy.OutputField(desc="Confidence level in the decision (0-1)")

class AgentReflection(dspy.Signature):
    """Reflect on past actions and learn from experience."""
    
    action_taken = dspy.InputField(desc="The action that was taken")
    result = dspy.InputField(desc="The result of the action")
    goal = dspy.InputField(desc="The goal that was being pursued")
    success = dspy.InputField(desc="Whether the action was successful")
    lesson_learned = dspy.OutputField(desc="What was learned from this experience")
    improvement = dspy.OutputField(desc="How to improve future actions")

class ToolUsage(dspy.Signature):
    """Decide which tool to use and how to use it."""
    
    task = dspy.InputField(desc="The task to accomplish")
    available_tools = dspy.InputField(desc="List of available tools and their descriptions")
    context = dspy.InputField(desc="Current context and state")
    tool_selection = dspy.OutputField(desc="Which tool to use")
    tool_parameters = dspy.OutputField(desc="Parameters to pass to the tool")
    expected_outcome = dspy.OutputField(desc="Expected outcome from using the tool")

class SimpleAgent(dspy.Module):
    """Simple reactive agent with basic reasoning."""
    
    def __init__(self, name: str = "SimpleAgent"):
        super().__init__()
        self.name = name
        self.reason = dspy.ChainOfThought(AgentReasoning)
        self.memory = AgentMemory()
        self.goals = []
        self.actions_taken = []
    
    def add_goal(self, goal: AgentGoal):
        """Add a goal to the agent."""
        self.goals.append(goal)
    
    def act(self, situation: str, available_actions: List[str]) -> dspy.Prediction:
        """Take action based on current situation."""
        
        # Get relevant memory
        relevant_memory = self._get_relevant_memory(situation)
        
        # Get current goal
        current_goal = self._get_current_goal()
        
        # Reason about the situation
        reasoning_result = self.reason(
            situation=situation,
            goal=current_goal.description if current_goal else "No specific goal",
            available_actions=str(available_actions),
            memory=str(relevant_memory)
        )
        
        # Store in memory
        self._store_memory({
            'type': 'reasoning',
            'situation': situation,
            'decision': reasoning_result.decision,
            'confidence': reasoning_result.confidence,
            'timestamp': datetime.now()
        })
        
        return reasoning_result
    
    def _get_relevant_memory(self, context: str) -> List[Dict[str, Any]]:
        """Get memory relevant to current context."""
        # Simple implementation - return recent memories
        return self.memory.short_term[-5:] + self.memory.working_memory.get('relevant', [])
    
    def _get_current_goal(self) -> Optional[AgentGoal]:
        """Get the current highest priority goal."""
        active_goals = [g for g in self.goals if not g.completed]
        if active_goals:
            return max(active_goals, key=lambda x: x.priority)
        return None
    
    def _store_memory(self, memory_item: Dict[str, Any]):
        """Store item in agent memory."""
        self.memory.short_term.append(memory_item)
        if len(self.memory.short_term) > self.memory.max_short_term:
            # Move old memory to long-term
            old_memory = self.memory.short_term.pop(0)
            self.memory.long_term.append(old_memory)
            if len(self.memory.long_term) > self.memory.max_long_term:
                self.memory.long_term.pop(0)

class PlanningAgent(dspy.Module):
    """Agent with planning and reflection capabilities."""
    
    def __init__(self, name: str = "PlanningAgent"):
        super().__init__()
        self.name = name
        self.plan = dspy.ChainOfThought(AgentPlanning)
        self.reason = dspy.ChainOfThought(AgentReasoning)
        self.reflect = dspy.ChainOfThought(AgentReflection)
        self.memory = AgentMemory()
        self.goals = []
        self.current_plan = None
        self.actions_taken = []
    
    def add_goal(self, goal: AgentGoal):
        """Add a goal to the agent."""
        self.goals.append(goal)
        # Trigger replanning
        self.current_plan = None
    
    def act(self, situation: str, available_actions: List[str]) -> Dict[str, Any]:
        """Take action with planning and reflection."""
        
        current_goal = self._get_current_goal()
        if not current_goal:
            return {'action': 'wait', 'reason': 'No active goals'}
        
        # Plan if needed
        if not self.current_plan:
            self._create_plan(current_goal, situation)
        
        # Reason about immediate action
        reasoning_result = self.reason(
            situation=situation,
            goal=current_goal.description,
            available_actions=str(available_actions),
            memory=str(self._get_relevant_memory(situation))
        )
        
        # Take action
        action = AgentAction(
            action_type=reasoning_result.decision,
            parameters={'confidence': reasoning_result.confidence},
            timestamp=datetime.now()
        )
        
        # Simulate action execution
        action.result = self._execute_action(action, situation)
        self.actions_taken.append(action)
        
        # Reflect on action
        self._reflect_on_action(action, current_goal)
        
        return {
            'action': action.action_type,
            'reasoning': reasoning_result.reasoning,
            'confidence': reasoning_result.confidence,
            'result': action.result
        }
    
    def _create_plan(self, goal: AgentGoal, current_state: str):
        """Create a plan to achieve the goal."""
        try:
            planning_result = self.plan(
                goal=goal.description,
                current_state=current_state,
                memory=str(self._get_relevant_memory(goal.description))
            )
            
            self.current_plan = {
                'goal': goal,
                'plan': planning_result.plan,
                'next_action': planning_result.next_action,
                'created_at': datetime.now()
            }
            
            self._store_memory({
                'type': 'planning',
                'goal': goal.description,
                'plan': planning_result.plan,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            print_error(f"Error creating plan: {e}")
    
    def _execute_action(self, action: AgentAction, situation: str) -> str:
        """Simulate action execution."""
        # This is a simulation - in a real agent, this would interface with the environment
        action_outcomes = [
            "Action completed successfully",
            "Action partially completed",
            "Action failed - trying alternative approach",
            "Action successful with unexpected results"
        ]
        return random.choice(action_outcomes)
    
    def _reflect_on_action(self, action: AgentAction, goal: AgentGoal):
        """Reflect on the taken action and learn."""
        try:
            reflection_result = self.reflect(
                action_taken=action.action_type,
                result=action.result,
                goal=goal.description,
                success=str(action.success)
            )
            
            self._store_memory({
                'type': 'reflection',
                'action': action.action_type,
                'lesson': reflection_result.lesson_learned,
                'improvement': reflection_result.improvement,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            print_error(f"Error during reflection: {e}")
    
    def _get_current_goal(self) -> Optional[AgentGoal]:
        """Get the current highest priority goal."""
        active_goals = [g for g in self.goals if not g.completed]
        if active_goals:
            return max(active_goals, key=lambda x: x.priority)
        return None
    
    def _get_relevant_memory(self, context: str) -> List[Dict[str, Any]]:
        """Get memory relevant to current context."""
        # Simple relevance based on recency and type
        relevant = []
        
        # Add recent memories
        relevant.extend(self.memory.short_term[-3:])
        
        # Add planning memories
        planning_memories = [m for m in self.memory.long_term if m.get('type') == 'planning']
        relevant.extend(planning_memories[-2:])
        
        return relevant
    
    def _store_memory(self, memory_item: Dict[str, Any]):
        """Store item in agent memory."""
        self.memory.short_term.append(memory_item)
        if len(self.memory.short_term) > self.memory.max_short_term:
            old_memory = self.memory.short_term.pop(0)
            self.memory.long_term.append(old_memory)
            if len(self.memory.long_term) > self.memory.max_long_term:
                self.memory.long_term.pop(0)

class ToolUsingAgent(dspy.Module):
    """Agent that can use tools to accomplish tasks."""
    
    def __init__(self, name: str = "ToolAgent"):
        super().__init__()
        self.name = name
        self.tool_selector = dspy.ChainOfThought(ToolUsage)
        self.reason = dspy.ChainOfThought(AgentReasoning)
        self.memory = AgentMemory()
        self.tools = {}
        self.goals = []
    
    def add_tool(self, name: str, description: str, function: Callable):
        """Add a tool that the agent can use."""
        self.tools[name] = {
            'description': description,
            'function': function
        }
    
    def add_goal(self, goal: AgentGoal):
        """Add a goal to the agent."""
        self.goals.append(goal)
    
    def act(self, task: str, context: str = "") -> Dict[str, Any]:
        """Perform a task using available tools."""
        
        # Select appropriate tool
        tool_decision = self.tool_selector(
            task=task,
            available_tools=str({name: tool['description'] for name, tool in self.tools.items()}),
            context=context
        )
        
        tool_name = tool_decision.tool_selection
        
        # Execute tool if available
        if tool_name in self.tools:
            try:
                # Parse parameters (simplified)
                params = self._parse_tool_parameters(tool_decision.tool_parameters)
                
                # Execute tool
                result = self.tools[tool_name]['function'](**params)
                
                self._store_memory({
                    'type': 'tool_usage',
                    'task': task,
                    'tool': tool_name,
                    'parameters': params,
                    'result': str(result),
                    'success': True,
                    'timestamp': datetime.now()
                })
                
                return {
                    'tool_used': tool_name,
                    'parameters': params,
                    'result': result,
                    'expected_outcome': tool_decision.expected_outcome,
                    'success': True
                }
                
            except Exception as e:
                self._store_memory({
                    'type': 'tool_usage',
                    'task': task,
                    'tool': tool_name,
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now()
                })
                
                return {
                    'tool_used': tool_name,
                    'error': str(e),
                    'success': False
                }
        else:
            return {
                'error': f"Tool '{tool_name}' not available",
                'available_tools': list(self.tools.keys()),
                'success': False
            }
    
    def _parse_tool_parameters(self, param_string: str) -> Dict[str, Any]:
        """Parse tool parameters from string."""
        try:
            # Try to parse as JSON
            return json.loads(param_string)
        except:
            # Fallback to simple key=value parsing
            params = {}
            for part in param_string.split(','):
                if '=' in part:
                    key, value = part.split('=', 1)
                    params[key.strip()] = value.strip()
            return params
    
    def _store_memory(self, memory_item: Dict[str, Any]):
        """Store item in agent memory."""
        self.memory.short_term.append(memory_item)
        if len(self.memory.short_term) > self.memory.max_short_term:
            old_memory = self.memory.short_term.pop(0)
            self.memory.long_term.append(old_memory)

# Sample tools for demonstration
def calculator_tool(operation: str, a: float, b: float) -> float:
    """Simple calculator tool."""
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else float('inf')
    }
    return operations.get(operation, lambda x, y: 0)(float(a), float(b))

def weather_tool(location: str) -> str:
    """Mock weather tool."""
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    return f"Weather in {location}: {condition}, {temperature}°C"

def note_tool(action: str, content: str = "") -> str:
    """Simple note-taking tool."""
    if action == "create":
        return f"Note created: {content}"
    elif action == "read":
        return f"Note content: {content}"
    else:
        return "Unknown note action"

def demonstrate_simple_agent():
    """Demonstrate simple reactive agent."""
    
    print_step("Simple Reactive Agent")
    
    agent = SimpleAgent("ReactiveBot")
    
    # Add goals
    agent.add_goal(AgentGoal("Learn about the environment", priority=5))
    agent.add_goal(AgentGoal("Help users with tasks", priority=3))
    
    # Simulate interactions
    scenarios = [
        ("User asks for help with math problem", ["solve_math", "ask_clarification", "search_info"]),
        ("New information about environment available", ["read_info", "store_info", "ignore"]),
        ("User requests weather information", ["check_weather", "ask_location", "provide_general_info"])
    ]
    
    for situation, actions in scenarios:
        try:
            result = agent.act(situation, actions)
            print_result(f"Situation: {situation}")
            print_result(f"Available Actions: {actions}")
            print_result(f"Decision: {result.decision}")
            print_result(f"Reasoning: {result.reasoning}")
            print_result(f"Confidence: {result.confidence}")
            print("-" * 50)
        except Exception as e:
            print_error(f"Error in simple agent: {e}")

def demonstrate_planning_agent():
    """Demonstrate planning agent with reflection."""
    
    print_step("Planning Agent with Reflection")
    
    agent = PlanningAgent("PlannerBot")
    
    # Add a complex goal
    agent.add_goal(AgentGoal("Organize a virtual team meeting", priority=8))
    
    # Simulate multiple action steps
    scenarios = [
        ("Need to organize team meeting", ["schedule_calendar", "send_invites", "prepare_agenda"]),
        ("Calendar conflicts detected", ["find_alternative_time", "prioritize_attendees", "reschedule"]),
        ("Meeting agenda needs preparation", ["gather_topics", "create_slides", "send_materials"])
    ]
    
    for situation, actions in scenarios:
        try:
            result = agent.act(situation, actions)
            print_result(f"Situation: {situation}")
            print_result(f"Action Taken: {result['action']}")
            print_result(f"Reasoning: {result['reasoning']}")
            print_result(f"Confidence: {result['confidence']}")
            print_result(f"Result: {result['result']}")
            print("-" * 50)
        except Exception as e:
            print_error(f"Error in planning agent: {e}")

def demonstrate_tool_using_agent():
    """Demonstrate agent with tool usage capabilities."""
    
    print_step("Tool-Using Agent")
    
    agent = ToolUsingAgent("ToolBot")
    
    # Add tools
    agent.add_tool("calculator", "Perform mathematical calculations", calculator_tool)
    agent.add_tool("weather", "Get weather information for a location", weather_tool)
    agent.add_tool("notes", "Create and manage notes", note_tool)
    
    # Add goal
    agent.add_goal(AgentGoal("Assist with various tasks using tools", priority=7))
    
    # Test different tasks
    tasks = [
        ("Calculate 15 + 27", "User needs math help"),
        ("Check weather in New York", "User planning trip"),
        ("Create a note about meeting agenda", "User organizing meeting")
    ]
    
    for task, context in tasks:
        try:
            result = agent.act(task, context)
            print_result(f"Task: {task}")
            print_result(f"Context: {context}")
            
            if result['success']:
                print_result(f"Tool Used: {result['tool_used']}")
                print_result(f"Parameters: {result.get('parameters', {})}")
                print_result(f"Result: {result['result']}")
                print_result(f"Expected: {result.get('expected_outcome', 'N/A')}")
            else:
                print_result(f"Error: {result.get('error', 'Unknown error')}")
            
            print("-" * 50)
            
        except Exception as e:
            print_error(f"Error in tool-using agent: {e}")

def demonstrate_multi_agent_system():
    """Demonstrate multiple agents working together."""
    
    print_step("Multi-Agent System")
    
    # Create specialized agents
    planner = PlanningAgent("Planner")
    tool_user = ToolUsingAgent("ToolUser")
    
    # Add tools to tool user
    tool_user.add_tool("calculator", "Perform calculations", calculator_tool)
    tool_user.add_tool("weather", "Get weather info", weather_tool)
    
    # Add goals
    planner.add_goal(AgentGoal("Plan a project timeline", priority=9))
    tool_user.add_goal(AgentGoal("Provide data and calculations", priority=6))
    
    # Simulate collaboration
    print_result("Planner creates project plan:")
    plan_result = planner.act(
        "Need to plan project timeline for Q1",
        ["create_timeline", "assign_tasks", "set_milestones"]
    )
    print_result(f"Plan Action: {plan_result['action']}")
    print_result(f"Plan Reasoning: {plan_result['reasoning']}")
    
    print_result("\nTool User provides supporting data:")
    tool_result = tool_user.act(
        "Calculate total project budget: 15000 + 8500 + 12000",
        "Supporting the project planning"
    )
    
    if tool_result['success']:
        print_result(f"Calculation Result: {tool_result['result']}")
        print_result(f"Tool Used: {tool_result['tool_used']}")

def main():
    """Main function demonstrating different agent architectures."""
    
    print("=" * 60)
    print("DSPy Agents Demo")
    print("=" * 60)
    
    # Setup language model
    lm = setup_default_lm()
    if not lm:
        return
    
    try:
        # Simple reactive agent
        demonstrate_simple_agent()
        
        # Planning agent with reflection
        demonstrate_planning_agent()
        
        # Tool-using agent
        demonstrate_tool_using_agent()
        
        # Multi-agent system
        demonstrate_multi_agent_system()
        
        print_step("Agent Demonstrations Complete!")
        
    except Exception as e:
        print_error(f"Error in agent demo: {e}")

if __name__ == "__main__":
    main()
