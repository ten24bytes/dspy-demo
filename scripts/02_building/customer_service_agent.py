#!/usr/bin/env python3
"""
Building AI Agents with DSPy: Customer Service Agent - Python Script Version

This script demonstrates building intelligent customer service agents using DSPy:
- Multi-step reasoning for customer queries
- Intent classification and response generation
- Tool integration for order management
- Memory and context management
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class MockOrderSystem:
    """Mock order management system."""

    def __init__(self):
        self.orders = {
            "ORD-12345": {
                "id": "ORD-12345",
                "customer_id": "CUST-001",
                "status": "shipped",
                "items": ["Laptop", "Mouse"],
                "total": 1299.99,
                "order_date": "2024-01-15",
                "tracking_number": "TRK-98765"
            },
            "ORD-67890": {
                "id": "ORD-67890",
                "customer_id": "CUST-002",
                "status": "processing",
                "items": ["Headphones"],
                "total": 199.99,
                "order_date": "2024-01-20",
                "tracking_number": None
            }
        }

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order details by order ID."""
        return self.orders.get(order_id)

    def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = status
            return True
        return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders and self.orders[order_id]["status"] in ["processing", "pending"]:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False


class MockCustomerDB:
    """Mock customer database."""

    def __init__(self):
        self.customers = {
            "john.doe@email.com": {
                "id": "CUST-001",
                "name": "John Doe",
                "email": "john.doe@email.com",
                "tier": "premium",
                "orders": ["ORD-12345"]
            },
            "jane.smith@email.com": {
                "id": "CUST-002",
                "name": "Jane Smith",
                "email": "jane.smith@email.com",
                "tier": "standard",
                "orders": ["ORD-67890"]
            }
        }

    def get_customer(self, email: str) -> Optional[Dict]:
        """Get customer details by email."""
        return self.customers.get(email)


class MockKnowledgeBase:
    """Mock knowledge base for FAQs and policies."""

    def __init__(self):
        self.articles = {
            "shipping_policy": "We offer free shipping on orders over $50. Standard shipping takes 3-5 business days.",
            "return_policy": "Items can be returned within 30 days of purchase for a full refund.",
            "warranty_info": "All electronics come with a 1-year manufacturer warranty.",
            "payment_methods": "We accept credit cards, PayPal, and bank transfers."
        }

    def search(self, query: str) -> str:
        """Search knowledge base for relevant information."""
        query_lower = query.lower()
        for key, content in self.articles.items():
            if any(word in query_lower for word in key.split('_')):
                return content
        return "I don't have specific information about that. Please contact our support team for more details."


class CustomerServiceAgent(dspy.Module):
    """Intelligent customer service agent using DSPy."""

    def __init__(self, order_system, customer_db, knowledge_base):
        super().__init__()

        # Backend systems
        self.order_system = order_system
        self.customer_db = customer_db
        self.knowledge_base = knowledge_base

        # Define signatures
        class ClassifyIntent(dspy.Signature):
            """Classify the customer's intent from their message."""
            customer_message = dspy.InputField(desc="The customer's message or query")
            intent = dspy.OutputField(desc="Intent: order_inquiry, technical_support, billing_question, general_info, complaint, or other")
            confidence = dspy.OutputField(desc="Confidence level (high/medium/low)")

        class ExtractOrderInfo(dspy.Signature):
            """Extract order-related information from customer message."""
            customer_message = dspy.InputField(desc="The customer's message")
            order_id = dspy.OutputField(desc="Order ID if mentioned, otherwise 'none'")
            email = dspy.OutputField(desc="Customer email if mentioned, otherwise 'none'")
            action_requested = dspy.OutputField(desc="What the customer wants to do with their order")

        class GenerateResponse(dspy.Signature):
            """Generate a helpful customer service response."""
            customer_message = dspy.InputField(desc="The customer's original message")
            intent = dspy.InputField(desc="Classified intent")
            context_info = dspy.InputField(desc="Relevant information from systems (orders, customer data, etc.)")
            response = dspy.OutputField(desc="A helpful, professional, and empathetic response")
            suggested_actions = dspy.OutputField(desc="Suggested next steps or actions")

        class EscalationCheck(dspy.Signature):
            """Determine if a customer inquiry should be escalated to human support."""
            customer_message = dspy.InputField(desc="The customer's message")
            intent = dspy.InputField(desc="Classified intent")
            previous_interactions = dspy.InputField(desc="Summary of previous interactions")
            should_escalate = dspy.OutputField(desc="true if should escalate to human, false otherwise")
            escalation_reason = dspy.OutputField(desc="Reason for escalation if applicable")

        # DSPy modules
        self.classify_intent = dspy.Predict(ClassifyIntent)
        self.extract_order_info = dspy.Predict(ExtractOrderInfo)
        self.generate_response = dspy.ChainOfThought(GenerateResponse)
        self.check_escalation = dspy.Predict(EscalationCheck)

        # Conversation memory
        self.conversation_history = []

    def get_order_context(self, order_id: str, email: str) -> str:
        """Get order and customer context information."""
        context_parts = []

        # Get order info
        if order_id != 'none':
            order = self.order_system.get_order(order_id)
            if order:
                context_parts.append(f"Order {order_id}: Status={order['status']}, Items={order['items']}, Total=${order['total']}, Date={order['order_date']}")
            else:
                context_parts.append(f"Order {order_id} not found in system.")

        # Get customer info
        if email != 'none':
            customer = self.customer_db.get_customer(email)
            if customer:
                context_parts.append(f"Customer: {customer['name']} ({customer['tier']} tier), Orders: {customer['orders']}")
            else:
                context_parts.append(f"Customer with email {email} not found.")

        return " | ".join(context_parts) if context_parts else "No specific order or customer context available."

    def handle_order_action(self, order_id: str, action: str) -> str:
        """Handle order-related actions."""
        if "cancel" in action.lower():
            if self.order_system.cancel_order(order_id):
                return f"Order {order_id} has been successfully cancelled."
            else:
                return f"Unable to cancel order {order_id}. It may already be shipped or completed."

        return "Action processed."

    def forward(self, customer_message: str, customer_email: str = None):
        """Process customer message and generate response."""

        # Step 1: Classify intent
        intent_result = self.classify_intent(customer_message=customer_message)

        # Step 2: Extract order information if relevant
        order_info = self.extract_order_info(customer_message=customer_message)

        # Step 3: Gather context information
        context_info = self.get_order_context(order_info.order_id, order_info.email or customer_email or 'none')

        # Step 4: Add knowledge base information for general questions
        if intent_result.intent in ['general_info', 'technical_support']:
            kb_info = self.knowledge_base.search(customer_message)
            context_info += f" | Knowledge Base: {kb_info}"

        # Step 5: Handle order actions if needed
        action_result = ""
        if intent_result.intent == 'order_inquiry' and order_info.order_id != 'none':
            action_result = self.handle_order_action(order_info.order_id, order_info.action_requested)
            context_info += f" | Action Result: {action_result}"

        # Step 6: Generate response
        response_result = self.generate_response(
            customer_message=customer_message,
            intent=intent_result.intent,
            context_info=context_info
        )

        # Step 7: Check for escalation
        escalation_result = self.check_escalation(
            customer_message=customer_message,
            intent=intent_result.intent,
            previous_interactions=str(self.conversation_history[-3:])  # Last 3 interactions
        )

        # Step 8: Update conversation history
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "customer_message": customer_message,
            "intent": intent_result.intent,
            "response": response_result.response
        }
        self.conversation_history.append(interaction)

        return dspy.Prediction(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            order_id=order_info.order_id,
            extracted_email=order_info.email,
            action_requested=order_info.action_requested,
            context_info=context_info,
            reasoning=response_result.reasoning,
            response=response_result.response,
            suggested_actions=response_result.suggested_actions,
            should_escalate=escalation_result.should_escalate,
            escalation_reason=escalation_result.escalation_reason,
            action_result=action_result
        )


def test_agent_scenario(agent, message: str, email: str = None, scenario_name: str = ""):
    """Test the agent with a specific scenario."""
    print(f"\\n🎯 Scenario: {scenario_name}")
    print(f"Customer Message: {message}")
    if email:
        print(f"Customer Email: {email}")
    print("-" * 60)

    result = agent(customer_message=message, customer_email=email)

    print(f"Intent: {result.intent} (Confidence: {result.confidence})")
    print(f"Response: {result.response}")
    print(f"Suggested Actions: {result.suggested_actions}")

    if result.should_escalate == "true":
        print(f"🚨 ESCALATION NEEDED: {result.escalation_reason}")

    if result.action_result:
        print(f"Action Performed: {result.action_result}")

    print("=" * 60)


def simulate_conversation(agent, messages: List[str], customer_email: str = None):
    """Simulate a multi-turn conversation."""
    print(f"\\n💬 Starting conversation with {customer_email or 'anonymous customer'}")
    print("=" * 70)

    for i, message in enumerate(messages, 1):
        print(f"\\nTurn {i}:")
        print(f"Customer: {message}")

        result = agent(customer_message=message, customer_email=customer_email)

        print(f"Agent ({result.intent}): {result.response}")

        if result.should_escalate == "true":
            print(f"🚨 [ESCALATION TRIGGERED: {result.escalation_reason}]")

        print("-" * 50)


def analyze_agent_performance(agent):
    """Analyze the agent's conversation history."""
    history = agent.conversation_history

    if not history:
        print("No conversation history available.")
        return

    # Intent distribution
    intents = [interaction['intent'] for interaction in history]
    intent_counts = {}
    for intent in intents:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print(f"Total Interactions: {len(history)}")
    print(f"Intent Distribution: {intent_counts}")

    # Recent interactions
    print("\\nRecent Interactions:")
    for interaction in history[-3:]:
        print(f"- {interaction['timestamp']}: {interaction['intent']} - {interaction['customer_message'][:50]}...")

    # Time analysis
    if len(history) > 1:
        first_time = datetime.fromisoformat(history[0]['timestamp'])
        last_time = datetime.fromisoformat(history[-1]['timestamp'])
        duration = last_time - first_time
        print(f"\\nConversation Duration: {duration.total_seconds():.1f} seconds")


def main():
    """Main function demonstrating customer service agent with DSPy."""
    print("🤖 Customer Service Agent with DSPy")
    print("=" * 50)

    # Load environment variables
    load_dotenv('.env')

    # Configure Language Model
    print_step("Configuring Language Model", "Setting up DSPy with OpenAI")

    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=1500)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return

    # Initialize mock systems
    print_step("Initializing Backend Systems", "Setting up mock order, customer, and knowledge systems")

    order_system = MockOrderSystem()
    customer_db = MockCustomerDB()
    knowledge_base = MockKnowledgeBase()

    print_result("Mock backend systems initialized!")

    # Create the customer service agent
    print_step("Creating Customer Service Agent", "Building the intelligent agent")

    agent = CustomerServiceAgent(order_system, customer_db, knowledge_base)
    print_result("Customer service agent created successfully!")

    # Test scenarios
    print_step("Testing Customer Service Agent", "Running various customer scenarios")

    test_scenarios = [
        {
            "message": "Hi, I want to check the status of my order ORD-12345",
            "email": "john.doe@email.com",
            "scenario": "Order Status Inquiry"
        },
        {
            "message": "I need to cancel my order ORD-67890 immediately!",
            "email": "jane.smith@email.com",
            "scenario": "Order Cancellation Request"
        },
        {
            "message": "What's your return policy? I'm not happy with my purchase.",
            "scenario": "Policy Question"
        },
        {
            "message": "This is terrible! I've been waiting 3 weeks and still no product! I want my money back NOW!",
            "scenario": "Angry Customer Complaint"
        },
        {
            "message": "Do you offer free shipping? I'm thinking of placing a large order.",
            "scenario": "Shipping Inquiry"
        }
    ]

    for scenario in test_scenarios:
        test_agent_scenario(
            agent,
            message=scenario["message"],
            email=scenario.get("email"),
            scenario_name=scenario["scenario"]
        )

    # Multi-turn conversation
    print_step("Testing Conversation Flow", "Multi-turn conversation with context")

    conversation_messages = [
        "Hi, I placed an order last week but haven't received any updates",
        "My order number is ORD-12345",
        "Great! When will it arrive?",
        "Actually, I need to change my delivery address. Is that possible?"
    ]

    simulate_conversation(agent, conversation_messages, "john.doe@email.com")

    # Analytics
    print_step("Agent Analytics", "Analyzing conversation history and performance")

    analyze_agent_performance(agent)

    # Show conversation history
    print("\\n📋 Full Conversation History:")
    for i, interaction in enumerate(agent.conversation_history, 1):
        print(f"{i}. [{interaction['intent']}] {interaction['customer_message']}")
        print(f"   Response: {interaction['response'][:100]}...")
        print()

    print("\\n🎉 Customer Service Agent demonstration completed!")
    print("\\nKey Features Demonstrated:")
    print("- Intent classification and confidence scoring")
    print("- Order and customer context integration")
    print("- Knowledge base search for policies")
    print("- Multi-turn conversation handling")
    print("- Escalation logic for complex issues")
    print("- Conversation analytics and history tracking")


if __name__ == "__main__":
    main()
