#!/usr/bin/env python3
"""
Email Information Extraction with DSPy

This script demonstrates how to build intelligent email processing and information extraction systems using DSPy.
It covers email classification, entity extraction, sentiment analysis, and automated response generation.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

@dataclass
class EmailData:
    """Structured representation of email data."""
    subject: str
    sender: str
    recipients: List[str]
    body: str
    timestamp: datetime
    attachments: List[str] = None

@dataclass
class ExtractedInfo:
    """Information extracted from email."""
    email_type: str
    priority: str
    sentiment: str
    action_items: List[str]
    key_entities: Dict[str, List[str]]
    summary: str
    next_steps: List[str]
    deadline_mentions: List[str]

@dataclass
class ContactInfo:
    """Contact information extracted from email."""
    name: str
    email: str
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None

def main():
    """Main function demonstrating email information extraction with DSPy."""
    print("=" * 70)
    print("EMAIL INFORMATION EXTRACTION WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for email processing")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Sample email data
    sample_emails = [
        EmailData(
            subject="Project Update - Q4 Launch Timeline",
            sender="sarah.johnson@company.com",
            recipients=["team@company.com"],
            body="""Hi Team,
            
            I wanted to provide an update on our Q4 product launch timeline. Based on the latest development progress, we're on track to complete beta testing by November 15th.
            
            Key milestones:
            - Complete user testing: November 10th
            - Fix critical bugs: November 12th 
            - Final QA review: November 15th
            - Marketing campaign launch: November 20th
            - Public release: December 1st
            
            Please let me know if you have any concerns about these dates. We need to finalize the marketing materials by next Friday.
            
            Best regards,
            Sarah Johnson
            Product Manager
            sarah.johnson@company.com
            (555) 123-4567
            """,
            timestamp=datetime.now() - timedelta(days=1)
        ),
        
        EmailData(
            subject="URGENT: Server Issues - Need Immediate Action",
            sender="alerts@monitoring.com",
            recipients=["devops@company.com", "management@company.com"],
            body="""CRITICAL ALERT
            
            Server cluster prod-web-01 is experiencing high CPU utilization (95%+) and memory issues.
            
            Error details:
            - Memory usage: 98% of 32GB
            - CPU usage: 96% average over last 15 minutes
            - Active connections: 15,000+ (normal: 5,000)
            - Response time: 8.5s (normal: 0.3s)
            
            IMMEDIATE ACTIONS REQUIRED:
            1. Scale up additional server instances
            2. Investigate database query performance
            3. Check for potential DDoS attack
            4. Contact John Smith (DevOps Lead) at john.smith@company.com or (555) 987-6543
            
            This issue is affecting customer experience. Please address within the next 30 minutes.
            
            Monitoring System
            """,
            timestamp=datetime.now() - timedelta(hours=2)
        ),
        
        EmailData(
            subject="Thank you for the amazing service!",
            sender="happy.customer@email.com",
            recipients=["support@company.com"],
            body="""Dear Support Team,
            
            I just wanted to reach out and thank you for the exceptional service I received yesterday. 
            Lisa from your customer success team went above and beyond to help me resolve the billing issue.
            
            Your response time was incredibly fast (less than 2 hours) and the solution was perfect. 
            I've been a customer for 3 years and this level of service is why I continue to recommend 
            your company to my colleagues.
            
            Please pass along my thanks to Lisa and the entire support team.
            
            Best wishes,
            Michael Chen
            CEO, TechStart Inc.
            michael.chen@techstart.com
            """,
            timestamp=datetime.now() - timedelta(hours=6)
        )
    ]
    
    # DSPy Signatures
    class EmailClassification(dspy.Signature):
        """Classify email type and determine priority level."""
        
        email_subject = dspy.InputField(desc="Subject line of the email")
        email_body = dspy.InputField(desc="Main content of the email")
        sender_info = dspy.InputField(desc="Information about the email sender")
        
        email_type = dspy.OutputField(desc="Type of email: business_update, urgent_alert, customer_feedback, meeting_request, support_ticket, etc.")
        priority = dspy.OutputField(desc="Priority level: HIGH, MEDIUM, LOW")
        urgency_reason = dspy.OutputField(desc="Explanation for the assigned priority level")
    
    class ActionItemExtraction(dspy.Signature):
        """Extract action items and tasks from email content."""
        
        email_content = dspy.InputField(desc="Complete email content including subject and body")
        
        action_items = dspy.OutputField(desc="List of specific action items or tasks mentioned in the email")
        deadlines = dspy.OutputField(desc="Any deadlines or time-sensitive items mentioned")
        responsible_parties = dspy.OutputField(desc="People or teams responsible for actions")
        next_steps = dspy.OutputField(desc="Recommended next steps based on email content")
    
    class EntityExtraction(dspy.Signature):
        """Extract entities and contact information from email."""
        
        email_text = dspy.InputField(desc="Email content to analyze for entities")
        
        people = dspy.OutputField(desc="Names of people mentioned in the email")
        companies = dspy.OutputField(desc="Company or organization names")
        dates = dspy.OutputField(desc="Important dates and deadlines")
        contact_info = dspy.OutputField(desc="Phone numbers, email addresses, or other contact details")
        technical_terms = dspy.OutputField(desc="Technical terms, product names, or project names")
    
    class SentimentAnalysis(dspy.Signature):
        """Analyze sentiment and tone of email communication."""
        
        email_content = dspy.InputField(desc="Email content to analyze for sentiment")
        email_context = dspy.InputField(desc="Context about the email type and purpose")
        
        sentiment = dspy.OutputField(desc="Overall sentiment: POSITIVE, NEGATIVE, NEUTRAL")
        tone = dspy.OutputField(desc="Communication tone: formal, casual, urgent, friendly, frustrated, etc.")
        emotional_indicators = dspy.OutputField(desc="Specific words or phrases indicating emotion")
        customer_satisfaction = dspy.OutputField(desc="If customer email, satisfaction level: satisfied, neutral, dissatisfied")
    
    class EmailSummarization(dspy.Signature):
        """Create a comprehensive summary of email content."""
        
        email_subject = dspy.InputField(desc="Email subject line")
        email_body = dspy.InputField(desc="Email body content")
        extracted_entities = dspy.InputField(desc="Previously extracted entities and action items")
        
        summary = dspy.OutputField(desc="Concise summary of the email's main points")
        key_takeaways = dspy.OutputField(desc="Most important information from the email")
        follow_up_required = dspy.OutputField(desc="Whether follow-up action is needed: yes/no")
        suggested_response = dspy.OutputField(desc="Suggested response or reply if appropriate")
    
    # Email Processor Class
    class EmailProcessor(dspy.Module):
        """Comprehensive email processing and information extraction module."""
        
        def __init__(self):
            super().__init__()
            self.classifier = dspy.ChainOfThought(EmailClassification)
            self.action_extractor = dspy.ChainOfThought(ActionItemExtraction)
            self.entity_extractor = dspy.ChainOfThought(EntityExtraction)
            self.sentiment_analyzer = dspy.ChainOfThought(SentimentAnalysis)
            self.summarizer = dspy.ChainOfThought(EmailSummarization)
        
        def preprocess_email(self, email_data: EmailData) -> Dict[str, str]:
            """Preprocess email data for analysis."""
            # Clean and format email content
            clean_body = re.sub(r'\\s+', ' ', email_data.body.strip())
            
            # Extract sender domain for context
            sender_domain = email_data.sender.split('@')[-1] if '@' in email_data.sender else ''
            
            return {
                'subject': email_data.subject,
                'body': clean_body,
                'sender': email_data.sender,
                'sender_domain': sender_domain,
                'full_content': f"Subject: {email_data.subject}\\n\\nFrom: {email_data.sender}\\n\\n{clean_body}"
            }
        
        def extract_contact_info(self, text: str) -> List[ContactInfo]:
            """Extract contact information using regex patterns."""
            contacts = []
            
            # Extract email addresses
            email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
            emails = re.findall(email_pattern, text)
            
            # Extract phone numbers (simple pattern)
            phone_pattern = r'\\(?\\d{3}\\)?[-\\s]?\\d{3}[-\\s]?\\d{4}'
            phones = re.findall(phone_pattern, text)
            
            # Extract names (basic pattern - capitalized words before email/phone)
            name_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)(?=.*(?:@|\\(\\d{3}\\))?)'
            names = re.findall(name_pattern, text)
            
            # Combine information
            for i, email_addr in enumerate(emails):
                name = names[i] if i < len(names) else ""
                phone = phones[i] if i < len(phones) else None
                contacts.append(ContactInfo(name=name, email=email_addr, phone=phone))
            
            return contacts
        
        def process_email(self, email_data: EmailData) -> ExtractedInfo:
            """Process a single email and extract all relevant information."""
            # Preprocess email
            processed = self.preprocess_email(email_data)
            
            # Step 1: Classify email
            classification = self.classifier(
                email_subject=processed['subject'],
                email_body=processed['body'],
                sender_info=f"Sender: {processed['sender']} (Domain: {processed['sender_domain']})"
            )
            
            # Step 2: Extract action items
            actions = self.action_extractor(
                email_content=processed['full_content']
            )
            
            # Step 3: Extract entities
            entities = self.entity_extractor(
                email_text=processed['full_content']
            )
            
            # Step 4: Analyze sentiment
            sentiment = self.sentiment_analyzer(
                email_content=processed['full_content'],
                email_context=f"Email type: {classification.email_type}, Priority: {classification.priority}"
            )
            
            # Step 5: Generate summary
            summary = self.summarizer(
                email_subject=processed['subject'],
                email_body=processed['body'],
                extracted_entities=f"""Action Items: {actions.action_items}
                People: {entities.people}
                Companies: {entities.companies}
                Dates: {entities.dates}"""
            )
            
            # Extract contacts using regex
            contacts = self.extract_contact_info(processed['full_content'])
            
            # Combine all extracted information
            return ExtractedInfo(
                email_type=classification.email_type,
                priority=classification.priority,
                sentiment=sentiment.sentiment,
                action_items=actions.action_items.split('\\n') if actions.action_items else [],
                key_entities={
                    'people': entities.people.split(',') if entities.people else [],
                    'companies': entities.companies.split(',') if entities.companies else [],
                    'dates': entities.dates.split(',') if entities.dates else [],
                    'contacts': [f"{c.name} - {c.email}" for c in contacts]
                },
                summary=summary.summary,
                next_steps=actions.next_steps.split('\\n') if actions.next_steps else [],
                deadline_mentions=actions.deadlines.split(',') if actions.deadlines else []
            )
        
        def generate_report(self, extracted_infos: List[ExtractedInfo]) -> str:
            """Generate a summary report of processed emails."""
            total_emails = len(extracted_infos)
            high_priority = sum(1 for info in extracted_infos if info.priority == 'HIGH')
            action_required = sum(1 for info in extracted_infos if info.action_items)
            
            sentiment_breakdown = {}
            type_breakdown = {}
            
            for info in extracted_infos:
                sentiment_breakdown[info.sentiment] = sentiment_breakdown.get(info.sentiment, 0) + 1
                type_breakdown[info.email_type] = type_breakdown.get(info.email_type, 0) + 1
            
            report = f"""
            EMAIL PROCESSING REPORT
            ======================
            
            Total Emails Processed: {total_emails}
            High Priority Emails: {high_priority}
            Emails Requiring Action: {action_required}
            
            SENTIMENT BREAKDOWN:
            {dict(sentiment_breakdown)}
            
            EMAIL TYPE BREAKDOWN:
            {dict(type_breakdown)}
            """
            
            return report
    
    # Initialize processor
    processor = EmailProcessor()
    print_result("Email processor initialized successfully!")
    
    # Process sample emails
    print_step("Email Processing Demo", "Processing sample emails")
    
    extracted_results = []
    
    for i, email_data in enumerate(sample_emails, 1):
        try:
            print(f"\\n{'='*60}")
            print(f"Processing Email {i}: {email_data.subject[:50]}...")
            print('='*60)
            
            extracted_info = processor.process_email(email_data)
            extracted_results.append(extracted_info)
            
            # Display results
            print_result(f"Type: {extracted_info.email_type}", "Email Classification")
            print_result(f"Priority: {extracted_info.priority}", "Priority Level")
            print_result(f"Sentiment: {extracted_info.sentiment}", "Sentiment Analysis")
            print_result(f"Summary: {extracted_info.summary}", "Email Summary")
            
            if extracted_info.action_items:
                print_result("\\n".join([f"- {item.strip()}" for item in extracted_info.action_items if item.strip()]), "Action Items")
            
            if extracted_info.deadline_mentions:
                print_result(", ".join(extracted_info.deadline_mentions), "Deadlines")
            
            if extracted_info.key_entities['people']:
                people = [p.strip() for p in extracted_info.key_entities['people'] if p.strip()]
                if people:
                    print_result(", ".join(people), "People Mentioned")
            
            if extracted_info.key_entities['contacts']:
                contacts = [c.strip() for c in extracted_info.key_entities['contacts'] if c.strip()]
                if contacts:
                    print_result("\\n".join(contacts), "Contact Information")
            
        except Exception as e:
            print_error(f"Error processing email {i}: {e}")
            continue
    
    # Generate comprehensive report
    print_step("Generating Report", "Creating summary of processing results")
    
    report = processor.generate_report(extracted_results)
    print_result(report, "Processing Report")
    
    # Priority analysis
    print("\\n" + "="*60)
    print("PRIORITY ANALYSIS")
    print("="*60)
    
    high_priority_emails = [info for info in extracted_results if info.priority == 'HIGH']
    print_result(f"High Priority Emails: {len(high_priority_emails)}", "Priority Breakdown")
    
    for i, info in enumerate(high_priority_emails, 1):
        print(f"  {i}. {info.email_type} - {info.summary[:100]}...")
    
    # Action items summary
    print("\\n" + "="*60)
    print("ACTION ITEMS SUMMARY")
    print("="*60)
    
    all_action_items = []
    for info in extracted_results:
        all_action_items.extend(info.action_items)
    
    if all_action_items:
        print_result(f"Total Action Items: {len(all_action_items)}", "Action Summary")
        for i, action in enumerate(all_action_items[:5], 1):  # Show first 5
            if action.strip():
                print(f"  {i}. {action.strip()}")
        
        if len(all_action_items) > 5:
            print(f"  ... and {len(all_action_items) - 5} more action items")
    else:
        print_result("No action items found", "Action Summary")
    
    # Integration examples
    print_step("Integration Examples", "Demonstrating system integrations")
    
    def create_task_from_email(email_data: EmailData, extracted_info: ExtractedInfo) -> Dict[str, Any]:
        """Create a task management system entry from email."""
        task = {
            'title': f"Email: {email_data.subject}",
            'description': extracted_info.summary,
            'priority': extracted_info.priority.lower(),
            'action_items': extracted_info.action_items,
            'due_date': None,  # Would extract from deadline_mentions
            'tags': [extracted_info.email_type, extracted_info.sentiment.lower()],
            'source': 'email',
            'created_at': datetime.now().isoformat()
        }
        return task
    
    # Create tasks for emails with action items
    tasks_created = []
    for email_data, extracted_info in zip(sample_emails, extracted_results):
        if extracted_info.action_items:
            task = create_task_from_email(email_data, extracted_info)
            tasks_created.append(task)
    
    print_result(f"Tasks Created: {len(tasks_created)}", "Task Management Integration")
    for task in tasks_created:
        print(f"  - {task['title']} (Priority: {task['priority']})")
    
    # Performance metrics
    print_step("Performance Analysis", "Calculating processing metrics")
    
    processing_time = datetime.now()
    emails_per_second = len(sample_emails) / 5  # Estimate 5 seconds per email
    
    print_result(f"Emails Processed: {len(sample_emails)}", "Performance Metrics")
    print_result(f"Processing Rate: ~{emails_per_second:.2f} emails/second", "")
    print_result(f"Success Rate: {len(extracted_results)/len(sample_emails)*100:.1f}%", "")
    
    print("\\n" + "="*70)
    print("EMAIL PROCESSING COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy-powered email information extraction!")

if __name__ == "__main__":
    main()
