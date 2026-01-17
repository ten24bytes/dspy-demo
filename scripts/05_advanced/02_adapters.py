#!/usr/bin/env python3
"""
DSPy Adapters - Formatting and Structuring Model Outputs

This script demonstrates how to use DSPy 3.x adapters to control
how prompts are formatted and how outputs are parsed:
- ChatAdapter: For chat-style conversational interfaces
- JSONAdapter: For structured JSON outputs
- XMLAdapter: For XML-structured outputs

What You'll Learn:
- How to use ChatAdapter for better conversational prompts
- How to use JSONAdapter to ensure valid JSON outputs
- How to use XMLAdapter for XML-structured data
- When to use each adapter type
- How to combine adapters with modules
"""

from dotenv import load_dotenv
from utils import (
    setup_default_lm,
    configure_dspy,
    create_chat_adapter,
    create_json_adapter,
    create_xml_adapter,
    print_step,
    print_result,
    print_error
)
import dspy
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy adapters."""
    print("=== DSPy Adapters Tutorial ===")
    print("Formatting and structuring model outputs with adapters")
    print("=" * 80)

    # Load environment variables
    load_dotenv('.env')

    # ========== PART 1: ChatAdapter ==========
    print_step(
        "Part 1: ChatAdapter",
        "Using ChatAdapter for conversational interfaces"
    )

    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=500)

        # Configure with ChatAdapter
        # ChatAdapter formats messages in chat format, which is more natural for conversational AI
        chat_adapter = create_chat_adapter()
        configure_dspy(lm=lm, adapter=chat_adapter)

        print_result("Language model configured with ChatAdapter!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return

    # Example 1.1: Basic Chat
    print("\n  Example 1.1: Basic Conversational Interface")

    class ChatAssistant(dspy.Signature):
        """A friendly conversational assistant."""
        user_message = dspy.InputField(desc="The user's message")
        assistant_response = dspy.OutputField(desc="A helpful and friendly response")

    chat_bot = dspy.Predict(ChatAssistant)

    messages = [
        "Hi! How are you?",
        "Can you help me understand what DSPy is?",
        "That's interesting! What are adapters used for?"
    ]

    for msg in messages:
        result = chat_bot(user_message=msg)
        print(f"\n  User: {msg}")
        print(f"  Assistant: {result.assistant_response}")

    # Example 1.2: Multi-turn Conversation
    print("\n  Example 1.2: Multi-turn Conversation with Context")

    class ContextualChat(dspy.Module):
        """A chatbot that maintains conversation context."""
        def __init__(self):
            super().__init__()

            class GenerateResponse(dspy.Signature):
                """Generate a contextual response."""
                conversation_history = dspy.InputField(desc="Previous conversation")
                user_message = dspy.InputField(desc="Current user message")
                response = dspy.OutputField(desc="Contextual response")

            self.respond = dspy.Predict(GenerateResponse)

        def forward(self, history, message):
            """Generate a response considering conversation history."""
            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])
            result = self.respond(
                conversation_history=history_text,
                user_message=message
            )
            return dspy.Prediction(response=result.response)

    contextual_bot = ContextualChat()

    conversation = []
    test_messages = [
        "What's the capital of France?",
        "What's the population of that city?",
        "What are some famous landmarks there?"
    ]

    for msg in test_messages:
        result = contextual_bot(history=conversation, message=msg)
        print(f"\n  User: {msg}")
        print(f"  Bot: {result.response}")

        conversation.append({"role": "user", "content": msg})
        conversation.append({"role": "assistant", "content": result.response})

    # ========== PART 2: JSONAdapter ==========
    print_step(
        "Part 2: JSONAdapter",
        "Using JSONAdapter for structured JSON outputs"
    )

    # Reconfigure with JSONAdapter
    json_adapter = create_json_adapter()
    lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=1000)
    configure_dspy(lm=lm, adapter=json_adapter)
    print_result("Reconfigured with JSONAdapter", "Status")

    # Example 2.1: Structured Data Extraction
    print("\n  Example 2.1: Extract Structured Data as JSON")

    class ExtractPersonInfo(dspy.Signature):
        """Extract person information in JSON format."""
        text = dspy.InputField(desc="Text containing person information")
        person_json = dspy.OutputField(desc="JSON with fields: name, age, occupation, location")

    extractor = dspy.Predict(ExtractPersonInfo)

    sample_text = """
    John Smith is a 35-year-old software engineer living in San Francisco.
    He has been working in the tech industry for over 10 years.
    """

    result = extractor(text=sample_text)
    print(f"\n  Input text: {sample_text.strip()}")
    print(f"\n  Extracted JSON:")
    try:
        # Try to parse and pretty-print the JSON
        parsed = json.loads(result.person_json)
        print(f"  {json.dumps(parsed, indent=2)}")
        print(f"\n  ✓ Valid JSON extracted!")
    except json.JSONDecodeError:
        print(f"  {result.person_json}")
        print(f"\n  ⚠ Output may not be valid JSON (adapter helps but doesn't guarantee)")

    # Example 2.2: Complex JSON Structure
    print("\n  Example 2.2: Generate Complex JSON Schema")

    class GenerateProductCatalog(dspy.Signature):
        """Generate a product catalog in JSON format."""
        category = dspy.InputField(desc="Product category")
        num_products = dspy.InputField(desc="Number of products to generate")
        catalog_json = dspy.OutputField(
            desc="JSON array of products, each with: id, name, price, description, features (array)"
        )

    catalog_gen = dspy.Predict(GenerateProductCatalog)

    result = catalog_gen(category="Laptops", num_products="3")
    print(f"\n  Generated catalog JSON:")
    try:
        catalog = json.loads(result.catalog_json)
        print(f"  {json.dumps(catalog, indent=2)}")
        print(f"\n  ✓ Generated {len(catalog)} products")
    except json.JSONDecodeError:
        print(f"  {result.catalog_json}")

    # Example 2.3: JSON Validation and Correction
    print("\n  Example 2.3: JSON Schema Validation")

    class ValidateAndFixJSON(dspy.Module):
        """Validate JSON against a schema and fix if needed."""
        def __init__(self):
            super().__init__()

            class FixJSON(dspy.Signature):
                """Fix invalid JSON to match schema."""
                invalid_json = dspy.InputField(desc="Potentially invalid JSON")
                schema_description = dspy.InputField(desc="Description of required schema")
                fixed_json = dspy.OutputField(desc="Valid JSON matching the schema")

            self.fixer = dspy.Predict(FixJSON)

        def forward(self, data, schema):
            """Validate and fix JSON data."""
            # Try to parse
            try:
                parsed = json.loads(data)
                return dspy.Prediction(
                    is_valid=True,
                    data=data,
                    message="JSON is valid"
                )
            except json.JSONDecodeError:
                # Fix invalid JSON
                result = self.fixer(
                    invalid_json=data,
                    schema_description=schema
                )
                return dspy.Prediction(
                    is_valid=False,
                    data=result.fixed_json,
                    message="JSON was fixed"
                )

    validator = ValidateAndFixJSON()

    invalid_json = '{"name": "John", "age": 30, occupation: "Engineer"}'  # Missing quotes
    schema = "Object with name (string), age (number), occupation (string)"

    result = validator(data=invalid_json, schema=schema)
    print(f"\n  Original: {invalid_json}")
    print(f"  Fixed: {result.data}")
    print(f"  Status: {result.message}")

    # ========== PART 3: XMLAdapter ==========
    print_step(
        "Part 3: XMLAdapter",
        "Using XMLAdapter for XML-structured outputs"
    )

    # Reconfigure with XMLAdapter
    xml_adapter = create_xml_adapter()
    lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=1000)
    configure_dspy(lm=lm, adapter=xml_adapter)
    print_result("Reconfigured with XMLAdapter", "Status")

    # Example 3.1: Generate XML Document
    print("\n  Example 3.1: Generate XML Configuration")

    class GenerateXMLConfig(dspy.Signature):
        """Generate XML configuration."""
        app_name = dspy.InputField(desc="Application name")
        settings = dspy.InputField(desc="Configuration settings to include")
        xml_config = dspy.OutputField(desc="XML configuration document")

    xml_gen = dspy.Predict(GenerateXMLConfig)

    result = xml_gen(
        app_name="MyApp",
        settings="database connection, API endpoints, logging level"
    )
    print(f"\n  Generated XML:")
    print(f"  {result.xml_config}")

    # Example 3.2: Convert Data to XML
    print("\n  Example 3.2: Convert Structured Data to XML")

    class DataToXML(dspy.Signature):
        """Convert structured data to XML format."""
        data_description = dspy.InputField(desc="Description of data")
        xml_output = dspy.OutputField(desc="XML representation of the data")

    converter = dspy.Predict(DataToXML)

    data_desc = """
    A company with name 'TechCorp', employees: [
        {name: 'Alice', role: 'Engineer', salary: 100000},
        {name: 'Bob', role: 'Manager', salary: 120000}
    ]
    """

    result = converter(data_description=data_desc)
    print(f"\n  Data: {data_desc.strip()}")
    print(f"\n  XML output:")
    print(f"  {result.xml_output}")

    # ========== PART 4: Comparing Adapters ==========
    print_step(
        "Part 4: Adapter Comparison",
        "Understanding when to use each adapter"
    )

    print("\n  ChatAdapter:")
    print("  ✓ Use for: Conversational interfaces, chatbots, dialogue systems")
    print("  ✓ Benefits: More natural conversation flow, better context handling")
    print("  ✓ Best with: Chat-optimized models (GPT-4, Claude)")
    print()

    print("  JSONAdapter:")
    print("  ✓ Use for: APIs, structured data extraction, configuration generation")
    print("  ✓ Benefits: Encourages valid JSON outputs, easier parsing")
    print("  ✓ Best with: Any model, especially for data-heavy applications")
    print()

    print("  XMLAdapter:")
    print("  ✓ Use for: Legacy systems, document generation, configuration files")
    print("  ✓ Benefits: Hierarchical structure, good for complex nested data")
    print("  ✓ Best with: Any model, especially when XML is required")
    print()

    print("  No Adapter (Default):")
    print("  ✓ Use for: General text generation, simple Q&A, open-ended tasks")
    print("  ✓ Benefits: Maximum flexibility, no format constraints")
    print("  ✓ Best with: Any model, for most general purposes")

    # ========== PART 5: Advanced Adapter Usage ==========
    print_step(
        "Part 5: Advanced Patterns",
        "Combining adapters with complex workflows"
    )

    class MultiFormatProcessor(dspy.Module):
        """
        Process data and output in multiple formats using different adapters.
        """
        def __init__(self):
            super().__init__()

            class ProcessData(dspy.Signature):
                """Process and format data."""
                input_data = dspy.InputField()
                format_type = dspy.InputField(desc="json or xml")
                output = dspy.OutputField(desc="Formatted output")

            self.processor = dspy.Predict(ProcessData)

        def forward(self, data, format="json"):
            """Process data in specified format."""
            # In practice, you would reconfigure with appropriate adapter here
            result = self.processor(input_data=data, format_type=format)
            return dspy.Prediction(output=result.output, format=format)

    processor = MultiFormatProcessor()

    print("\n  This pattern demonstrates:")
    print("  ✓ Dynamic adapter selection based on requirements")
    print("  ✓ Multi-format output generation")
    print("  ✓ Flexible data processing pipelines")
    print("\n  Use cases:")
    print("  - API services that support multiple output formats")
    print("  - Data transformation pipelines")
    print("  - Format conversion tools")

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. Adapters control how prompts are formatted and parsed")
    print("2. ChatAdapter: Best for conversational interfaces")
    print("3. JSONAdapter: Best for structured data and APIs")
    print("4. XMLAdapter: Best for hierarchical data and legacy systems")
    print("5. Use configure_dspy(adapter=...) to set the adapter")
    print("6. Adapters improve output reliability but don't guarantee format")
    print("7. Choose adapters based on your application's needs")


if __name__ == "__main__":
    main()
