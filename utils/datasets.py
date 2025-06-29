"""
Sample datasets for DSPy examples.
"""

import dspy
from typing import List, Dict, Any
import json
import os

def get_sample_qa_data() -> List[dspy.Example]:
    """Get sample question-answering data from JSON file."""
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "qa_dataset.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        return [dspy.Example(
            question=item["question"], 
            answer=item["answer"],
            context=item.get("context", "")
        ).with_inputs('question', 'context') for item in qa_pairs]
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load QA dataset from file ({e}). Using fallback data.")
        # Fallback to hardcoded data
        qa_pairs = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
            {"question": "In what year did World War II end?", "answer": "1945"}
        ]
        
        return [dspy.Example(question=item["question"], answer=item["answer"]) 
                for item in qa_pairs]

def get_sample_classification_data() -> List[dspy.Example]:
    """Get sample text classification data."""
    texts = [
        {
            "text": "I love this product! It's amazing and works perfectly.",
            "sentiment": "positive"
        },
        {
            "text": "This is terrible. I hate it and want my money back.",
            "sentiment": "negative"
        },
        {
            "text": "It's okay, nothing special but does the job.",
            "sentiment": "neutral"
        },
        {
            "text": "Fantastic quality and great customer service!",
            "sentiment": "positive"
        },
        {
            "text": "Worst purchase I've ever made. Complete waste of money.",
            "sentiment": "negative"
        }
    ]
    
    return [dspy.Example(text=item["text"], sentiment=item["sentiment"]) 
            for item in texts]

def get_sample_entity_data() -> List[dspy.Example]:
    """Get sample named entity recognition data."""
    texts = [
        {
            "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
            "entities": ["Apple Inc. (ORG)", "Steve Jobs (PERSON)", "Cupertino (LOC)", "California (LOC)"]
        },
        {
            "text": "The meeting is scheduled for Monday at Google headquarters in Mountain View.",
            "entities": ["Monday (DATE)", "Google (ORG)", "Mountain View (LOC)"]
        },
        {
            "text": "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
            "entities": ["Barack Obama (PERSON)", "44th President (TITLE)", "United States (LOC)", "2009 (DATE)", "2017 (DATE)"]
        }
    ]
    
    return [dspy.Example(text=item["text"], entities=item["entities"]) 
            for item in texts]

def get_sample_math_data() -> List[dspy.Example]:
    """Get sample mathematical reasoning data."""
    problems = [
        {
            "problem": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
            "solution": "Distance = Speed × Time = 60 mph × 2.5 hours = 150 miles",
            "answer": "150 miles"
        },
        {
            "problem": "What is the area of a rectangle with length 8 and width 5?",
            "solution": "Area = Length × Width = 8 × 5 = 40 square units",
            "answer": "40 square units"
        },
        {
            "problem": "Solve for x: 2x + 5 = 13",
            "solution": "2x + 5 = 13\n2x = 13 - 5\n2x = 8\nx = 4",
            "answer": "x = 4"
        }
    ]
    
    return [dspy.Example(problem=item["problem"], solution=item["solution"], answer=item["answer"]) 
            for item in problems]

def get_sample_rag_documents() -> List[str]:
    """Get sample documents for RAG examples."""
    documents = [
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data and make predictions.",
        
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. It combines computational linguistics with statistical and machine learning methods.",
        
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for tasks like image recognition and language processing.",
        
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data."
    ]
    
    return documents

def save_dataset(data: List[dspy.Example], filename: str, data_dir: str = "data"):
    """Save a dataset to JSON file."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert Examples to dictionaries
    dict_data = []
    for example in data:
        if hasattr(example, 'toDict'):
            dict_data.append(example.toDict())
        else:
            # Fallback for different DSPy versions
            dict_data.append({k: v for k, v in example.__dict__.items()})
    
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(dict_data, f, indent=2)
    
    print(f"Dataset saved to {filepath}")

def load_dataset(filename: str, data_dir: str = "data") -> List[dspy.Example]:
    """Load a dataset from JSON file."""
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, 'r') as f:
        dict_data = json.load(f)
    
    return [dspy.Example(**item) for item in dict_data]
