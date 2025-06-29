#!/usr/bin/env python3
"""
Entity Extraction with DSPy

This script demonstrates how to build named entity recognition (NER) and
information extraction systems using DSPy.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error

@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    confidence: Optional[float] = None

@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    text: str
    entities: List[Entity]
    structured_data: Dict[str, Any]

class BasicNER(dspy.Signature):
    """Extract named entities from text."""
    
    text = dspy.InputField(desc="The input text to process")
    entities = dspy.OutputField(desc="List of entities in format: [entity_text:entity_type, ...]")

class DetailedNER(dspy.Signature):
    """Extract detailed named entities with reasoning."""
    
    text = dspy.InputField(desc="The input text to process")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning for entity identification")
    people = dspy.OutputField(desc="Names of people mentioned")
    organizations = dspy.OutputField(desc="Names of organizations mentioned")
    locations = dspy.OutputField(desc="Names of places and locations")
    dates = dspy.OutputField(desc="Dates and time expressions")
    money = dspy.OutputField(desc="Monetary amounts and currencies")
    miscellaneous = dspy.OutputField(desc="Other important entities")

class ContactExtraction(dspy.Signature):
    """Extract contact information from text."""
    
    text = dspy.InputField(desc="The input text to process")
    names = dspy.OutputField(desc="Names of people")
    emails = dspy.OutputField(desc="Email addresses")
    phones = dspy.OutputField(desc="Phone numbers")
    companies = dspy.OutputField(desc="Company names")
    addresses = dspy.OutputField(desc="Physical addresses")

class EventExtraction(dspy.Signature):
    """Extract event information from text."""
    
    text = dspy.InputField(desc="The input text to process")
    event_name = dspy.OutputField(desc="Name of the event")
    date_time = dspy.OutputField(desc="Date and time of the event")
    location = dspy.OutputField(desc="Location of the event")
    participants = dspy.OutputField(desc="People or organizations involved")
    description = dspy.OutputField(desc="Brief description of the event")

class BasicEntityExtractor(dspy.Module):
    """Basic entity extraction using Chain of Thought."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(BasicNER)
    
    def forward(self, text: str) -> dspy.Prediction:
        return self.extract(text=text)

class DetailedEntityExtractor(dspy.Module):
    """Detailed entity extraction with multiple categories."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(DetailedNER)
    
    def forward(self, text: str) -> dspy.Prediction:
        return self.extract(text=text)

class ContactExtractor(dspy.Module):
    """Specialized contact information extractor."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ContactExtraction)
    
    def forward(self, text: str) -> dspy.Prediction:
        return self.extract(text=text)

class EventExtractor(dspy.Module):
    """Specialized event information extractor."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EventExtraction)
    
    def forward(self, text: str) -> dspy.Prediction:
        return self.extract(text=text)

class MultiStepExtractor(dspy.Module):
    """Multi-step extractor combining different approaches."""
    
    def __init__(self):
        super().__init__()
        self.preprocess = dspy.ChainOfThought("text -> cleaned_text, important_sentences")
        self.basic_ner = BasicEntityExtractor()
        self.detailed_ner = DetailedEntityExtractor()
        self.synthesize = dspy.ChainOfThought(
            "basic_entities, detailed_entities -> final_entities, confidence_scores"
        )
    
    def forward(self, text: str) -> dspy.Prediction:
        # Step 1: Preprocess
        preprocessed = self.preprocess(text=text)
        
        # Step 2: Basic extraction
        basic_result = self.basic_ner(text=preprocessed.cleaned_text)
        
        # Step 3: Detailed extraction
        detailed_result = self.detailed_ner(text=preprocessed.cleaned_text)
        
        # Step 4: Synthesize results
        final_result = self.synthesize(
            basic_entities=basic_result.entities,
            detailed_entities=f"People: {detailed_result.people}, "
                            f"Organizations: {detailed_result.organizations}, "
                            f"Locations: {detailed_result.locations}, "
                            f"Dates: {detailed_result.dates}"
        )
        
        return dspy.Prediction(
            text=text,
            cleaned_text=preprocessed.cleaned_text,
            important_sentences=preprocessed.important_sentences,
            basic_entities=basic_result.entities,
            people=detailed_result.people,
            organizations=detailed_result.organizations,
            locations=detailed_result.locations,
            dates=detailed_result.dates,
            money=detailed_result.money,
            final_entities=final_result.final_entities,
            confidence_scores=final_result.confidence_scores,
            reasoning=detailed_result.reasoning
        )

def parse_entities(entity_string: str) -> List[Entity]:
    """Parse entity string into Entity objects."""
    
    entities = []
    if not entity_string:
        return entities
    
    # Try to parse format: [entity_text:entity_type, ...]
    try:
        # Remove brackets and split by comma
        clean_string = entity_string.strip('[]')
        entity_parts = [part.strip() for part in clean_string.split(',')]
        
        for part in entity_parts:
            if ':' in part:
                text, label = part.split(':', 1)
                entities.append(Entity(
                    text=text.strip(),
                    label=label.strip()
                ))
    except Exception as e:
        print_error(f"Error parsing entities: {e}")
    
    return entities

def demonstrate_basic_ner():
    """Demonstrate basic named entity recognition."""
    
    print_step("Basic Named Entity Recognition")
    
    extractor = BasicEntityExtractor()
    
    test_texts = [
        "John Smith works at Microsoft in Seattle and will visit Tokyo next week.",
        "The meeting with Apple Inc. is scheduled for December 15, 2024 at $500 per hour.",
        "Dr. Sarah Johnson from Stanford University published research on climate change.",
        "Amazon announced a $2 billion investment in renewable energy projects."
    ]
    
    for text in test_texts:
        try:
            result = extractor(text=text)
            entities = parse_entities(result.entities)
            
            print_result(f"Text: {text}")
            print_result(f"Entities: {result.entities}")
            
            if entities:
                for entity in entities:
                    print_result(f"  - {entity.text} ({entity.label})")
            
            print("-" * 50)
            
        except Exception as e:
            print_error(f"Error in basic NER: {e}")

def demonstrate_detailed_ner():
    """Demonstrate detailed named entity recognition."""
    
    print_step("Detailed Named Entity Recognition")
    
    extractor = DetailedEntityExtractor()
    
    text = ("Apple Inc. CEO Tim Cook announced that the company will invest $5 billion "
            "in renewable energy projects across California and Texas by March 2025. "
            "The initiative, partnered with Tesla and Google, aims to reduce carbon "
            "emissions by 50% within two years.")
    
    try:
        result = extractor(text=text)
        
        print_result(f"Text: {text}")
        print_result(f"Reasoning: {result.reasoning}")
        print_result(f"People: {result.people}")
        print_result(f"Organizations: {result.organizations}")
        print_result(f"Locations: {result.locations}")
        print_result(f"Dates: {result.dates}")
        print_result(f"Money: {result.money}")
        print_result(f"Miscellaneous: {result.miscellaneous}")
        
    except Exception as e:
        print_error(f"Error in detailed NER: {e}")

def demonstrate_contact_extraction():
    """Demonstrate contact information extraction."""
    
    print_step("Contact Information Extraction")
    
    extractor = ContactExtractor()
    
    text = ("Please contact Dr. Emily Watson at emily.watson@university.edu "
            "or call (555) 123-4567. She works at Stanford Research Institute, "
            "located at 123 Academic Drive, Palo Alto, CA 94301. "
            "Alternative contact: John Doe (john.doe@company.com, 555-987-6543) "
            "from TechCorp Solutions.")
    
    try:
        result = extractor(text=text)
        
        print_result(f"Text: {text}")
        print_result(f"Names: {result.names}")
        print_result(f"Emails: {result.emails}")
        print_result(f"Phones: {result.phones}")
        print_result(f"Companies: {result.companies}")
        print_result(f"Addresses: {result.addresses}")
        
    except Exception as e:
        print_error(f"Error in contact extraction: {e}")

def demonstrate_event_extraction():
    """Demonstrate event information extraction."""
    
    print_step("Event Information Extraction")
    
    extractor = EventExtractor()
    
    text = ("The Annual AI Conference 2024 will be held on November 15-17, 2024 "
            "at the San Francisco Convention Center. Keynote speakers include "
            "Dr. Andrew Ng from Stanford University and Satya Nadella from Microsoft. "
            "The event will focus on advances in machine learning and artificial intelligence.")
    
    try:
        result = extractor(text=text)
        
        print_result(f"Text: {text}")
        print_result(f"Event Name: {result.event_name}")
        print_result(f"Date/Time: {result.date_time}")
        print_result(f"Location: {result.location}")
        print_result(f"Participants: {result.participants}")
        print_result(f"Description: {result.description}")
        
    except Exception as e:
        print_error(f"Error in event extraction: {e}")

def demonstrate_multi_step_extraction():
    """Demonstrate multi-step entity extraction."""
    
    print_step("Multi-Step Entity Extraction")
    
    extractor = MultiStepExtractor()
    
    text = ("Microsoft Corporation announced yesterday that CEO Satya Nadella will "
            "visit the new Seattle campus on January 20, 2024. The $1.5 billion "
            "facility will house 10,000 employees and feature advanced AI research labs. "
            "The event will be attended by government officials from Washington State "
            "and technology leaders from Amazon, Google, and Apple.")
    
    try:
        result = extractor(text=text)
        
        print_result(f"Original Text: {result.text}")
        print_result(f"Cleaned Text: {result.cleaned_text}")
        print_result(f"Important Sentences: {result.important_sentences}")
        print_result(f"Basic Entities: {result.basic_entities}")
        print_result(f"People: {result.people}")
        print_result(f"Organizations: {result.organizations}")
        print_result(f"Locations: {result.locations}")
        print_result(f"Dates: {result.dates}")
        print_result(f"Money: {result.money}")
        print_result(f"Final Entities: {result.final_entities}")
        print_result(f"Confidence Scores: {result.confidence_scores}")
        print_result(f"Reasoning: {result.reasoning}")
        
    except Exception as e:
        print_error(f"Error in multi-step extraction: {e}")

def demonstrate_regex_enhancement():
    """Demonstrate combining DSPy with regex for enhanced extraction."""
    
    print_step("DSPy + Regex Enhancement")
    
    def extract_with_regex(text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns."""
        
        patterns = {
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phones': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+ \d{1,2}, \d{4}\b',
            'money': r'\$[\d,]+\.?\d*'
        }
        
        extracted = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted[entity_type] = matches
        
        return extracted
    
    # Combine with DSPy
    extractor = DetailedEntityExtractor()
    
    text = ("Contact john.doe@company.com (555-123-4567) about the $50,000 project. "
            "Meeting scheduled for 12/15/2024. More info at https://company.com/project")
    
    try:
        # DSPy extraction
        dspy_result = extractor(text=text)
        
        # Regex extraction
        regex_result = extract_with_regex(text)
        
        print_result(f"Text: {text}")
        print_result("DSPy Results:")
        print_result(f"  People: {dspy_result.people}")
        print_result(f"  Organizations: {dspy_result.organizations}")
        print_result(f"  Money: {dspy_result.money}")
        print_result(f"  Dates: {dspy_result.dates}")
        
        print_result("Regex Results:")
        for entity_type, entities in regex_result.items():
            print_result(f"  {entity_type.title()}: {entities}")
        
    except Exception as e:
        print_error(f"Error in enhanced extraction: {e}")

def main():
    """Main function demonstrating entity extraction."""
    
    print("=" * 60)
    print("DSPy Entity Extraction Demo")
    print("=" * 60)
    
    # Setup language model
    lm = setup_default_lm()
    if not lm:
        return
    
    try:
        # Basic NER
        demonstrate_basic_ner()
        
        # Detailed NER
        demonstrate_detailed_ner()
        
        # Contact extraction
        demonstrate_contact_extraction()
        
        # Event extraction
        demonstrate_event_extraction()
        
        # Multi-step extraction
        demonstrate_multi_step_extraction()
        
        # Enhanced with regex
        demonstrate_regex_enhancement()
        
        print_step("Entity Extraction Complete!")
        
    except Exception as e:
        print_error(f"Error in entity extraction demo: {e}")

if __name__ == "__main__":
    main()
