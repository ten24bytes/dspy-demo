#!/usr/bin/env python3
"""
Privacy-Conscious Delegation with DSPy

This script demonstrates how to build privacy-aware AI systems that can
delegate tasks while protecting sensitive information and maintaining data privacy.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import hashlib
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

class PrivacyLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class SensitiveDataType(Enum):
    PII = "personally_identifiable_information"
    FINANCIAL = "financial_data"
    MEDICAL = "medical_information"
    CREDENTIALS = "authentication_credentials"
    PROPRIETARY = "proprietary_business_data"

@dataclass
class PrivacyPolicy:
    allowed_processing: List[str]
    data_retention_days: int
    anonymization_required: bool
    encryption_required: bool
    access_control_level: PrivacyLevel

@dataclass
class DelegationResult:
    sanitized_query: str
    privacy_analysis: Dict[str, Any]
    delegation_safe: bool
    recommendations: List[str]

def main():
    """Main function demonstrating privacy-conscious delegation with DSPy."""
    print("=" * 70)
    print("PRIVACY-CONSCIOUS DELEGATION WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for privacy-conscious processing")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # DSPy Signatures for Privacy Processing
    class SensitiveDataDetection(dspy.Signature):
        """Detect and classify sensitive data in text."""
        text = dspy.InputField(desc="Text to analyze for sensitive information")
        privacy_context = dspy.InputField(desc="Context about the data source and intended use")
        
        sensitive_elements = dspy.OutputField(desc="List of detected sensitive data elements with types")
        privacy_risk_level = dspy.OutputField(desc="Overall privacy risk level: low, medium, high, critical")
        data_categories = dspy.OutputField(desc="Categories of sensitive data found (PII, financial, medical, etc.)")
        compliance_concerns = dspy.OutputField(desc="Potential compliance issues (GDPR, HIPAA, etc.)")
    
    class DataSanitization(dspy.Signature):
        """Sanitize text while preserving utility for delegation."""
        original_text = dspy.InputField(desc="Original text containing sensitive information")
        sensitive_elements = dspy.InputField(desc="Identified sensitive elements to be sanitized")
        preservation_requirements = dspy.InputField(desc="What information must be preserved for the task")
        
        sanitized_text = dspy.OutputField(desc="Text with sensitive information removed or anonymized")
        sanitization_methods = dspy.OutputField(desc="Methods used for sanitization (redaction, anonymization, etc.)")
        utility_preservation_score = dspy.OutputField(desc="Score indicating how much utility was preserved (1-10)")
        reversibility_info = dspy.OutputField(desc="Information about whether sanitization can be reversed")
    
    class DelegationSafetyAssessment(dspy.Signature):
        """Assess whether a task can be safely delegated."""
        task_description = dspy.InputField(desc="Description of the task to be delegated")
        data_sensitivity = dspy.InputField(desc="Sensitivity level and types of data involved")
        delegation_target = dspy.InputField(desc="Information about the system/service receiving the delegation")
        privacy_policy = dspy.InputField(desc="Applicable privacy policies and constraints")
        
        delegation_recommendation = dspy.OutputField(desc="SAFE, CONDITIONAL, or UNSAFE delegation recommendation")
        risk_factors = dspy.OutputField(desc="Identified risk factors for this delegation")
        mitigation_strategies = dspy.OutputField(desc="Recommended strategies to reduce risks")
        alternative_approaches = dspy.OutputField(desc="Alternative approaches if delegation is unsafe")
    
    class PrivacyAuditGeneration(dspy.Signature):
        """Generate privacy audit trail for delegation decisions."""
        delegation_details = dspy.InputField(desc="Details of the delegation decision and process")
        data_flow = dspy.InputField(desc="Description of how data flows through the delegation")
        privacy_measures = dspy.InputField(desc="Privacy measures and controls applied")
        
        audit_summary = dspy.OutputField(desc="Comprehensive audit summary")
        compliance_verification = dspy.OutputField(desc="Verification of compliance with regulations")
        monitoring_recommendations = dspy.OutputField(desc="Recommendations for ongoing monitoring")
        documentation_requirements = dspy.OutputField(desc="Required documentation for compliance")
    
    # Privacy Processing Classes
    class PrivacyAnalyzer:
        """Analyzes and processes privacy-sensitive data."""
        
        def __init__(self):
            self.pii_patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
                'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            }
        
        def detect_pii_patterns(self, text: str) -> Dict[str, List[str]]:
            """Detect PII using regex patterns."""
            detected = {}
            for data_type, pattern in self.pii_patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    detected[data_type] = matches
            return detected
        
        def anonymize_text(self, text: str, detected_pii: Dict[str, List[str]]) -> str:
            """Anonymize detected PII in text."""
            anonymized = text
            
            for data_type, matches in detected_pii.items():
                for match in matches:
                    if data_type == 'email':
                        anonymized = anonymized.replace(match, '[EMAIL_REDACTED]')
                    elif data_type == 'phone':
                        anonymized = anonymized.replace(match, '[PHONE_REDACTED]')
                    elif data_type == 'ssn':
                        anonymized = anonymized.replace(match, '[SSN_REDACTED]')
                    elif data_type == 'credit_card':
                        anonymized = anonymized.replace(match, '[CARD_REDACTED]')
                    elif data_type == 'ip_address':
                        anonymized = anonymized.replace(match, '[IP_REDACTED]')
            
            return anonymized
        
        def calculate_privacy_hash(self, text: str) -> str:
            """Calculate a privacy-preserving hash of the text."""
            return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    # Privacy-Conscious Delegation Module
    class PrivacyConsciousDelegator(dspy.Module):
        """Comprehensive privacy-conscious delegation system."""
        
        def __init__(self):
            super().__init__()
            self.privacy_analyzer = PrivacyAnalyzer()
            self.sensitive_detector = dspy.ChainOfThought(SensitiveDataDetection)
            self.data_sanitizer = dspy.ChainOfThought(DataSanitization)
            self.safety_assessor = dspy.ChainOfThought(DelegationSafetyAssessment)
            self.audit_generator = dspy.ChainOfThought(PrivacyAuditGeneration)
        
        def analyze_privacy_risks(self, text: str, context: str = "General processing") -> dspy.Prediction:
            """Analyze privacy risks in the given text."""
            return self.sensitive_detector(text=text, privacy_context=context)
        
        def sanitize_for_delegation(self, text: str, sensitive_elements: str, requirements: str) -> dspy.Prediction:
            """Sanitize text for safe delegation."""
            return self.data_sanitizer(
                original_text=text,
                sensitive_elements=sensitive_elements,
                preservation_requirements=requirements
            )
        
        def assess_delegation_safety(self, task: str, sensitivity: str, target: str, policy: str) -> dspy.Prediction:
            """Assess whether delegation is safe."""
            return self.safety_assessor(
                task_description=task,
                data_sensitivity=sensitivity,
                delegation_target=target,
                privacy_policy=policy
            )
        
        def generate_audit_trail(self, delegation_details: str, data_flow: str, measures: str) -> dspy.Prediction:
            """Generate audit trail for delegation."""
            return self.audit_generator(
                delegation_details=delegation_details,
                data_flow=data_flow,
                privacy_measures=measures
            )
        
        def process_delegation_request(self, query: str, target_service: str, privacy_level: PrivacyLevel) -> DelegationResult:
            """Process a complete delegation request with privacy controls."""
            
            # Step 1: Analyze privacy risks
            privacy_analysis = self.analyze_privacy_risks(query, f"Delegation to {target_service}")
            
            # Step 2: Detect PII patterns
            detected_pii = self.privacy_analyzer.detect_pii_patterns(query)
            
            # Step 3: Determine if sanitization is needed
            risk_level = privacy_analysis.privacy_risk_level.lower()
            needs_sanitization = risk_level in ['high', 'critical'] or detected_pii
            
            sanitized_query = query
            if needs_sanitization:
                # Step 4: Sanitize the query
                sanitization_result = self.sanitize_for_delegation(
                    query,
                    privacy_analysis.sensitive_elements,
                    f"Preserve task intent for {target_service}"
                )
                sanitized_query = sanitization_result.sanitized_text
                
                # Also apply pattern-based anonymization
                sanitized_query = self.privacy_analyzer.anonymize_text(sanitized_query, detected_pii)
            
            # Step 5: Assess delegation safety
            policy_description = f"Privacy level: {privacy_level.value}, Anonymization required: {needs_sanitization}"
            safety_assessment = self.assess_delegation_safety(
                f"Process query: {sanitized_query}",
                privacy_analysis.privacy_risk_level,
                target_service,
                policy_description
            )
            
            # Step 6: Make delegation decision
            delegation_safe = safety_assessment.delegation_recommendation.upper() in ['SAFE', 'CONDITIONAL']
            
            return DelegationResult(
                sanitized_query=sanitized_query,
                privacy_analysis={
                    'risk_level': privacy_analysis.privacy_risk_level,
                    'sensitive_elements': privacy_analysis.sensitive_elements,
                    'data_categories': privacy_analysis.data_categories,
                    'compliance_concerns': privacy_analysis.compliance_concerns,
                    'detected_pii': detected_pii
                },
                delegation_safe=delegation_safe,
                recommendations=safety_assessment.mitigation_strategies.split(', ') if hasattr(safety_assessment, 'mitigation_strategies') else []
            )
    
    # Initialize the delegator
    delegator = PrivacyConsciousDelegator()
    print_result("Privacy-conscious delegator initialized successfully!")
    
    # Demo 1: Privacy Risk Analysis
    print_step("Privacy Risk Analysis", "Analyzing different types of sensitive content")
    
    test_cases = [
        {
            "text": "Please analyze the customer feedback for john.doe@email.com who called from 555-123-4567",
            "context": "Customer service analysis"
        },
        {
            "text": "The patient with ID 12345 has a history of diabetes and takes insulin daily",
            "context": "Medical data processing"
        },
        {
            "text": "Our Q4 revenue projections show a 15% increase in the APAC region",
            "context": "Financial reporting"
        },
        {
            "text": "Update the user profile for IP address 192.168.1.1 with new preferences",
            "context": "System administration"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            analysis = delegator.analyze_privacy_risks(case["text"], case["context"])
            
            print(f"\n--- Case {i}: {case['context']} ---")
            print_result(f"Text: {case['text']}", "Input Text")
            print_result(f"Risk Level: {analysis.privacy_risk_level}", "Privacy Risk")
            print_result(f"Sensitive Elements: {analysis.sensitive_elements}", "Sensitive Elements")
            print_result(f"Data Categories: {analysis.data_categories}", "Data Categories")
            print_result(f"Compliance Concerns: {analysis.compliance_concerns}", "Compliance")
            
        except Exception as e:
            print_error(f"Error analyzing case {i}: {e}")
    
    # Demo 2: Data Sanitization
    print_step("Data Sanitization Demo", "Sanitizing sensitive data for delegation")
    
    sanitization_examples = [
        {
            "text": "Send a reminder to sarah.johnson@company.com about her appointment on March 15th",
            "sensitive_elements": "Email address: sarah.johnson@company.com",
            "requirements": "Preserve the reminder intent and date information"
        },
        {
            "text": "The credit card ending in 4567 was charged $299.99 for the premium subscription",
            "sensitive_elements": "Partial credit card number: 4567, Financial amount: $299.99",
            "requirements": "Preserve transaction type and general amount range"
        },
        {
            "text": "Patient ID 789 reported symptoms consistent with condition X, prescribe medication Y",
            "sensitive_elements": "Patient ID: 789, Medical condition: X, Medication: Y",
            "requirements": "Preserve symptom analysis while anonymizing patient information"
        }
    ]
    
    for i, example in enumerate(sanitization_examples, 1):
        try:
            sanitized = delegator.sanitize_for_delegation(
                example["text"],
                example["sensitive_elements"],
                example["requirements"]
            )
            
            print(f"\n--- Sanitization {i} ---")
            print_result(f"Original: {example['text']}", "Original Text")
            print_result(f"Sanitized: {sanitized.sanitized_text}", "Sanitized Text")
            print_result(f"Methods: {sanitized.sanitization_methods}", "Sanitization Methods")
            print_result(f"Utility Score: {sanitized.utility_preservation_score}/10", "Utility Preserved")
            print_result(f"Reversible: {sanitized.reversibility_info}", "Reversibility")
            
        except Exception as e:
            print_error(f"Error in sanitization example {i}: {e}")
    
    # Demo 3: Delegation Safety Assessment
    print_step("Delegation Safety Assessment", "Evaluating delegation scenarios")
    
    delegation_scenarios = [
        {
            "task": "Translate customer feedback from Spanish to English",
            "sensitivity": "Medium - contains customer opinions but no PII",
            "target": "Third-party translation service",
            "policy": "Standard privacy policy, customer data anonymization required"
        },
        {
            "task": "Analyze medical records for pattern recognition",
            "sensitivity": "Critical - contains protected health information",
            "target": "External AI analysis service",
            "policy": "HIPAA compliance required, PHI anonymization mandatory"
        },
        {
            "task": "Generate marketing report from sales data",
            "sensitivity": "High - contains proprietary business metrics",
            "target": "Cloud analytics platform",
            "policy": "Business confidential, encryption in transit required"
        },
        {
            "task": "Process public support forum posts",
            "sensitivity": "Low - public information only",
            "target": "Sentiment analysis API",
            "policy": "Public data policy, no additional restrictions"
        }
    ]
    
    for i, scenario in enumerate(delegation_scenarios, 1):
        try:
            assessment = delegator.assess_delegation_safety(
                scenario["task"],
                scenario["sensitivity"],
                scenario["target"],
                scenario["policy"]
            )
            
            print(f"\n--- Delegation Scenario {i} ---")
            print_result(f"Task: {scenario['task']}", "Task Description")
            print_result(f"Recommendation: {assessment.delegation_recommendation}", "Delegation Recommendation")
            print_result(f"Risk Factors: {assessment.risk_factors}", "Risk Factors")
            print_result(f"Mitigation: {assessment.mitigation_strategies}", "Mitigation Strategies")
            print_result(f"Alternatives: {assessment.alternative_approaches}", "Alternative Approaches")
            
        except Exception as e:
            print_error(f"Error assessing scenario {i}: {e}")
    
    # Demo 4: Complete Delegation Workflow
    print_step("Complete Delegation Workflow", "End-to-end privacy-conscious delegation")
    
    delegation_requests = [
        {
            "query": "Analyze the sentiment of this customer email: 'Hi, I'm John Smith (john.smith@email.com). I'm unhappy with my recent order #12345. Please process my refund to card ending in 4567.'",
            "target_service": "Sentiment Analysis API",
            "privacy_level": PrivacyLevel.CONFIDENTIAL
        },
        {
            "query": "Summarize this meeting transcript: 'Attendees: Alice (alice@company.com), Bob (555-0123). Discussed Q4 budget of $2M and new hire Sarah starting Monday.'",
            "target_service": "Document Summarization Service",
            "privacy_level": PrivacyLevel.INTERNAL
        },
        {
            "query": "Translate this public announcement: 'Our company is pleased to announce record growth this quarter.'",
            "target_service": "Translation Service",
            "privacy_level": PrivacyLevel.PUBLIC
        }
    ]
    
    for i, request in enumerate(delegation_requests, 1):
        try:
            result = delegator.process_delegation_request(
                request["query"],
                request["target_service"],
                request["privacy_level"]
            )
            
            print(f"\n--- Delegation Request {i} ---")
            print_result(f"Original Query: {request['query'][:100]}...", "Original Query")
            print_result(f"Sanitized Query: {result.sanitized_query}", "Sanitized Query")
            print_result(f"Delegation Safe: {result.delegation_safe}", "Safety Assessment")
            print_result(f"Risk Level: {result.privacy_analysis['risk_level']}", "Risk Level")
            print_result(f"Detected PII: {result.privacy_analysis['detected_pii']}", "Detected PII")
            
            if result.recommendations:
                print_result(f"Recommendations: {', '.join(result.recommendations)}", "Recommendations")
            
        except Exception as e:
            print_error(f"Error processing delegation request {i}: {e}")
    
    # Demo 5: Audit Trail Generation
    print_step("Audit Trail Generation", "Creating compliance documentation")
    
    audit_scenarios = [
        {
            "delegation_details": "Customer sentiment analysis delegated to external API after PII removal",
            "data_flow": "Customer email → PII detection → Anonymization → External API → Results returned",
            "privacy_measures": "Email addresses redacted, names anonymized, order numbers hashed"
        },
        {
            "delegation_details": "Medical data analysis rejected due to high privacy risk",
            "data_flow": "Medical records → Privacy analysis → Risk assessment → Delegation rejected",
            "privacy_measures": "No data sent externally, local processing recommended"
        }
    ]
    
    for i, scenario in enumerate(audit_scenarios, 1):
        try:
            audit = delegator.generate_audit_trail(
                scenario["delegation_details"],
                scenario["data_flow"],
                scenario["privacy_measures"]
            )
            
            print(f"\n--- Audit Trail {i} ---")
            print_result(f"Summary: {audit.audit_summary}", "Audit Summary")
            print_result(f"Compliance: {audit.compliance_verification}", "Compliance Verification")
            print_result(f"Monitoring: {audit.monitoring_recommendations}", "Monitoring Recommendations")
            print_result(f"Documentation: {audit.documentation_requirements}", "Documentation Requirements")
            
        except Exception as e:
            print_error(f"Error generating audit trail {i}: {e}")
    
    # Demo 6: Privacy Policy Integration
    print_step("Privacy Policy Integration", "Applying different privacy policies")
    
    privacy_policies = {
        "GDPR_Strict": PrivacyPolicy(
            allowed_processing=["anonymized_analytics", "aggregated_reporting"],
            data_retention_days=30,
            anonymization_required=True,
            encryption_required=True,
            access_control_level=PrivacyLevel.RESTRICTED
        ),
        "HIPAA_Compliant": PrivacyPolicy(
            allowed_processing=["medical_analysis", "treatment_optimization"],
            data_retention_days=7,
            anonymization_required=True,
            encryption_required=True,
            access_control_level=PrivacyLevel.RESTRICTED
        ),
        "Business_Standard": PrivacyPolicy(
            allowed_processing=["business_analytics", "performance_monitoring"],
            data_retention_days=90,
            anonymization_required=False,
            encryption_required=True,
            access_control_level=PrivacyLevel.CONFIDENTIAL
        )
    }
    
    test_query = "Analyze customer satisfaction for user ID 12345 with email test@example.com"
    
    for policy_name, policy in privacy_policies.items():
        try:
            # Simulate policy application
            policy_description = f"""
            Policy: {policy_name}
            Allowed Processing: {', '.join(policy.allowed_processing)}
            Data Retention: {policy.data_retention_days} days
            Anonymization Required: {policy.anonymization_required}
            Encryption Required: {policy.encryption_required}
            Access Level: {policy.access_control_level.value}
            """
            
            assessment = delegator.assess_delegation_safety(
                "Customer satisfaction analysis",
                "Medium - contains customer identifier and email",
                "Analytics service",
                policy_description
            )
            
            print(f"\n--- Policy: {policy_name} ---")
            print_result(f"Recommendation: {assessment.delegation_recommendation}", "Delegation Decision")
            print_result(f"Risk Factors: {assessment.risk_factors}", "Risk Factors")
            print_result(f"Mitigation: {assessment.mitigation_strategies}", "Required Mitigations")
            
        except Exception as e:
            print_error(f"Error applying policy {policy_name}: {e}")
    
    print("\n" + "="*70)
    print("PRIVACY-CONSCIOUS DELEGATION COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy-powered privacy-conscious delegation!")

if __name__ == "__main__":
    main()
