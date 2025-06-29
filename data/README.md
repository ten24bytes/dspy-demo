# Sample Data Directory

This directory contains sample datasets used throughout the DSPy demo project examples.

## Contents

### Core Datasets

- `qa_dataset.json` - Question-answering pairs for basic examples
- `classification_dataset.json` - Text classification examples
- `entity_extraction_dataset.json` - Named entity recognition examples

### Knowledge Bases

- `company_knowledge/` - Sample company information for RAG examples
- `scientific_papers/` - Sample research papers for advanced RAG
- `customer_support/` - FAQ and support documentation

### Real-World Examples

- `financial_data/` - Sample financial data for Yahoo Finance examples
- `email_samples/` - Sample emails for email processing examples
- `game_content/` - Content for AI text game examples

### Training Data

- `optimization_examples/` - Examples for DSPy optimization tutorials
- `math_problems/` - Mathematical reasoning problems
- `conversation_history/` - Sample conversations for agent training

## Usage

These datasets are automatically loaded by the utility functions in `utils/datasets.py`.
You can also load them directly in your own examples:

```python
import json

# Load QA dataset
with open('data/qa_dataset.json', 'r') as f:
    qa_data = json.load(f)

# Load classification data
with open('data/classification_dataset.json', 'r') as f:
    classification_data = json.load(f)
```

## Data Sources

All sample data is either:

- Synthetically generated for educational purposes
- Public domain information
- Anonymized and simplified real-world examples

No proprietary or sensitive data is included.
