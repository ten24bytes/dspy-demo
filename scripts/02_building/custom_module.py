#!/usr/bin/env python3
"""
Building AI Applications by Customizing DSPy Modules

This script demonstrates how to create custom DSPy modules for specialized AI applications.
It covers module architecture, composition, state management, error handling, and optimization.
"""

import dspy
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

# Data structures
@dataclass
class ProcessingResult:
    content: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    suggestions: List[str]

class ContentAnalyzer(dspy.Module):
    """Analyzes content for sentiment, topics, and quality."""
    
    def __init__(self):
        super().__init__()
        self.sentiment_analyzer = dspy.ChainOfThought("text -> sentiment, confidence_score")
        self.topic_extractor = dspy.ChainOfThought("text -> topics, relevance_scores")
        self.quality_assessor = dspy.ChainOfThought("text -> quality_score, quality_factors")
    
    def forward(self, text: str) -> ProcessingResult:
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer(text=text)
        
        # Extract topics
        topic_result = self.topic_extractor(text=text)
        
        # Assess quality
        quality_result = self.quality_assessor(text=text)
        
        # Combine results
        metadata = {
            'sentiment': sentiment_result.sentiment,
            'sentiment_confidence': float(sentiment_result.confidence_score),
            'topics': topic_result.topics.split(', ') if hasattr(topic_result, 'topics') else [],
            'quality_score': float(quality_result.quality_score),
            'quality_factors': quality_result.quality_factors.split(', ') if hasattr(quality_result, 'quality_factors') else []
        }
        
        return ProcessingResult(
            content=text,
            confidence=metadata['sentiment_confidence'],
            metadata=metadata,
            timestamp=datetime.now()
        )

class ContentValidator(dspy.Module):
    """Validates content for various criteria."""
    
    def __init__(self, validation_rules: List[str] = None):
        super().__init__()
        self.validation_rules = validation_rules or [
            "appropriate language",
            "factual accuracy", 
            "coherence",
            "completeness"
        ]
        
        rules_str = ", ".join(self.validation_rules)
        self.validator = dspy.ChainOfThought(
            f"text, rules: {rules_str} -> is_valid: bool, issues: list, suggestions: list"
        )
    
    def forward(self, text: str) -> ValidationResult:
        rules_str = ", ".join(self.validation_rules)
        result = self.validator(text=text, rules=rules_str)
        
        is_valid = result.is_valid.lower() in ['true', 'yes', '1'] if hasattr(result, 'is_valid') else True
        
        issues = []
        if hasattr(result, 'issues') and result.issues:
            issues = result.issues.split(', ') if isinstance(result.issues, str) else []
        
        suggestions = []
        if hasattr(result, 'suggestions') and result.suggestions:
            suggestions = result.suggestions.split(', ') if isinstance(result.suggestions, str) else []
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions
        )

class ContentEnhancer(dspy.Module):
    """Enhances content based on analysis and validation."""
    
    def __init__(self):
        super().__init__()
        self.enhancer = dspy.ChainOfThought(
            "original_text, issues, suggestions -> enhanced_text, improvements_made"
        )
    
    def forward(self, text: str, validation_result: ValidationResult) -> str:
        if validation_result.is_valid:
            return text
        
        issues_str = "; ".join(validation_result.issues)
        suggestions_str = "; ".join(validation_result.suggestions)
        
        result = self.enhancer(
            original_text=text,
            issues=issues_str,
            suggestions=suggestions_str
        )
        
        return result.enhanced_text if hasattr(result, 'enhanced_text') else text

class ContentProcessingPipeline(dspy.Module):
    """Complete content processing pipeline."""
    
    def __init__(self, validation_rules: List[str] = None):
        super().__init__()
        self.analyzer = ContentAnalyzer()
        self.validator = ContentValidator(validation_rules)
        self.enhancer = ContentEnhancer()
        self.processing_history = []
    
    def forward(self, text: str, enhance_if_needed: bool = True) -> Dict[str, Any]:
        # Analyze content
        analysis_result = self.analyzer(text)
        
        # Validate content
        validation_result = self.validator(text)
        
        # Enhance if needed
        final_text = text
        if enhance_if_needed and not validation_result.is_valid:
            final_text = self.enhancer(text, validation_result)
        
        pipeline_result = {
            'original_text': text,
            'final_text': final_text,
            'was_enhanced': final_text != text,
            'analysis': analysis_result,
            'validation': validation_result,
            'processing_timestamp': datetime.now()
        }
        
        self.processing_history.append(pipeline_result)
        return pipeline_result
    
    def get_processing_stats(self) -> Dict[str, Any]:
        if not self.processing_history:
            return {'total_processed': 0}
        
        total = len(self.processing_history)
        enhanced = sum(1 for result in self.processing_history if result['was_enhanced'])
        avg_confidence = sum(
            result['analysis'].confidence for result in self.processing_history
        ) / total
        
        return {
            'total_processed': total,
            'enhanced_count': enhanced,
            'enhancement_rate': enhanced / total,
            'average_confidence': avg_confidence
        }

class ConversationManager(dspy.Module):
    """Manages conversation state and context."""
    
    def __init__(self, max_history: int = 10):
        super().__init__()
        self.max_history = max_history
        self.conversation_history = []
        self.conversation_summary = ""
        
        self.responder = dspy.ChainOfThought(
            "conversation_history, current_message -> response, confidence"
        )
        self.summarizer = dspy.ChainOfThought("conversation_history -> summary")
    
    def forward(self, message: str, user_id: str = "user") -> Dict[str, Any]:
        # Add message to history
        self.conversation_history.append({
            'user_id': user_id,
            'message': message,
            'timestamp': datetime.now()
        })
        
        # Maintain history size
        if len(self.conversation_history) > self.max_history:
            old_messages = self.conversation_history[:-self.max_history//2]
            old_text = "\\n".join([f"{msg['user_id']}: {msg['message']}" for msg in old_messages])
            
            summary_result = self.summarizer(conversation_history=old_text)
            self.conversation_summary = summary_result.summary if hasattr(summary_result, 'summary') else ""
            
            self.conversation_history = self.conversation_history[-self.max_history//2:]
        
        # Prepare context
        context_parts = []
        if self.conversation_summary:
            context_parts.append(f"Previous conversation summary: {self.conversation_summary}")
        
        recent_messages = "\\n".join([
            f"{msg['user_id']}: {msg['message']}" 
            for msg in self.conversation_history[-5:]
        ])
        context_parts.append(f"Recent conversation:\\n{recent_messages}")
        
        context = "\\n\\n".join(context_parts)
        
        # Generate response
        response_result = self.responder(
            conversation_history=context,
            current_message=message
        )
        
        response = response_result.response if hasattr(response_result, 'response') else "I understand."
        confidence = float(response_result.confidence) if hasattr(response_result, 'confidence') else 0.8
        
        # Add response to history
        self.conversation_history.append({
            'user_id': 'assistant',
            'message': response,
            'timestamp': datetime.now()
        })
        
        return {
            'response': response,
            'confidence': confidence,
            'conversation_length': len(self.conversation_history),
            'has_summary': bool(self.conversation_summary)
        }
    
    def get_conversation_state(self) -> Dict[str, Any]:
        return {
            'history_length': len(self.conversation_history),
            'summary_exists': bool(self.conversation_summary),
            'last_message_time': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }
    
    def reset_conversation(self):
        self.conversation_history.clear()
        self.conversation_summary = ""

class RobustProcessor(dspy.Module):
    """Robust module with comprehensive error handling."""
    
    def __init__(self, retry_attempts: int = 3):
        super().__init__()
        self.retry_attempts = retry_attempts
        self.error_count = 0
        self.success_count = 0
        
        self.primary_processor = dspy.ChainOfThought("input -> processed_output, confidence")
        self.fallback_processor = dspy.Predict("input -> simple_output")
        self.error_handler = dspy.Predict("error_description, input -> recovery_suggestion")
    
    def forward(self, input_text: str, processing_mode: str = "normal") -> Dict[str, Any]:
        result = {
            'input': input_text,
            'output': None,
            'success': False,
            'error': None,
            'attempts': 0,
            'processing_mode': processing_mode,
            'fallback_used': False
        }
        
        # Input validation
        if not input_text or not isinstance(input_text, str):
            result['error'] = "Invalid input: must be non-empty string"
            self.error_count += 1
            return result
        
        # Try primary processing with retries
        for attempt in range(self.retry_attempts):
            result['attempts'] = attempt + 1
            
            try:
                if processing_mode == "detailed":
                    processed = self.primary_processor(input=input_text)
                    result['output'] = processed.processed_output if hasattr(processed, 'processed_output') else str(processed)
                    result['confidence'] = float(processed.confidence) if hasattr(processed, 'confidence') else 0.8
                else:
                    processed = self.fallback_processor(input=input_text)
                    result['output'] = processed.simple_output if hasattr(processed, 'simple_output') else str(processed)
                    result['confidence'] = 0.7
                
                result['success'] = True
                self.success_count += 1
                break
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Processing attempt {attempt + 1} failed: {error_msg}")
                
                if attempt == self.retry_attempts - 1:
                    # Try fallback
                    try:
                        fallback_result = self.fallback_processor(input=input_text)
                        result['output'] = fallback_result.simple_output if hasattr(fallback_result, 'simple_output') else str(fallback_result)
                        result['confidence'] = 0.5
                        result['success'] = True
                        result['fallback_used'] = True
                        self.success_count += 1
                        
                        # Get recovery suggestion
                        try:
                            recovery = self.error_handler(error_description=error_msg, input=input_text)
                            result['recovery_suggestion'] = recovery.recovery_suggestion if hasattr(recovery, 'recovery_suggestion') else None
                        except:
                            pass
                        
                    except Exception as fallback_error:
                        result['error'] = f"All processing failed. Last error: {str(fallback_error)}"
                        self.error_count += 1
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        total = self.success_count + self.error_count
        return {
            'total_requests': total,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / total if total > 0 else 0.0
        }
    
    def reset_stats(self):
        self.success_count = 0
        self.error_count = 0

class OptimizedProcessor(dspy.Module):
    """Optimized module with caching and batching."""
    
    def __init__(self, cache_size: int = 100, batch_size: int = 5):
        super().__init__()
        self.cache_size = cache_size
        self.batch_size = batch_size
        
        self.processing_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.single_processor = dspy.Predict("text -> processed_text")
        self.batch_processor = dspy.Predict("text_batch -> processed_batch")
        
        self._cache = {}
    
    def _get_cache_key(self, text: str) -> str:
        return hash(text.strip().lower())
    
    def _cache_get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        return None
    
    def _cache_set(self, key: str, value: str):
        if len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def forward(self, texts: List[str]) -> List[Dict[str, Any]]:
        start_time = time.time()
        
        results = []
        cache_results = {}
        texts_to_process = []
        
        # Check cache
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_result = self._cache_get(cache_key)
            
            if cached_result:
                cache_results[i] = {
                    'original': text,
                    'processed': cached_result,
                    'cached': True,
                    'processing_time': 0.0
                }
            else:
                texts_to_process.append((i, text, cache_key))
        
        # Process uncached texts
        if texts_to_process:
            if len(texts_to_process) <= self.batch_size:
                # Batch processing
                batch_texts = [item[1] for item in texts_to_process]
                batch_start = time.time()
                
                try:
                    batch_input = "\\n---\\n".join(batch_texts)
                    batch_result = self.batch_processor(text_batch=batch_input)
                    batch_outputs = batch_result.processed_batch.split("\\n---\\n") if hasattr(batch_result, 'processed_batch') else batch_texts
                    
                    batch_time = time.time() - batch_start
                    avg_time = batch_time / len(texts_to_process)
                    
                    for (i, original_text, cache_key), processed_text in zip(texts_to_process, batch_outputs):
                        self._cache_set(cache_key, processed_text)
                        cache_results[i] = {
                            'original': original_text,
                            'processed': processed_text,
                            'cached': False,
                            'processing_time': avg_time,
                            'batch_processed': True
                        }
                
                except Exception as e:
                    # Fallback to individual processing
                    logger.warning(f"Batch processing failed, falling back: {e}")
                    for i, text, cache_key in texts_to_process:
                        individual_start = time.time()
                        try:
                            result = self.single_processor(text=text)
                            processed = result.processed_text if hasattr(result, 'processed_text') else text
                            individual_time = time.time() - individual_start
                            
                            self._cache_set(cache_key, processed)
                            cache_results[i] = {
                                'original': text,
                                'processed': processed,
                                'cached': False,
                                'processing_time': individual_time,
                                'batch_processed': False
                            }
                        except Exception as individual_error:
                            cache_results[i] = {
                                'original': text,
                                'processed': text,
                                'cached': False,
                                'processing_time': time.time() - individual_start,
                                'error': str(individual_error)
                            }
            else:
                # Individual processing for large batches
                for i, text, cache_key in texts_to_process:
                    individual_start = time.time()
                    try:
                        result = self.single_processor(text=text)
                        processed = result.processed_text if hasattr(result, 'processed_text') else text
                        individual_time = time.time() - individual_start
                        
                        self._cache_set(cache_key, processed)
                        cache_results[i] = {
                            'original': text,
                            'processed': processed,
                            'cached': False,
                            'processing_time': individual_time,
                            'batch_processed': False
                        }
                    except Exception as e:
                        cache_results[i] = {
                            'original': text,
                            'processed': text,
                            'cached': False,
                            'processing_time': time.time() - individual_start,
                            'error': str(e)
                        }
        
        # Assemble results
        results = [cache_results[i] for i in range(len(texts))]
        
        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        total_requests = self.cache_hits + self.cache_misses
        
        return {
            'cache_hit_rate': self.cache_hits / total_requests if total_requests > 0 else 0.0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self._cache),
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0,
            'total_batches_processed': len(self.processing_times)
        }
    
    def clear_cache(self):
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

class AdvancedContentManager(dspy.Module):
    """Advanced content management system combining multiple specialized modules."""
    
    def __init__(self):
        super().__init__()
        
        self.content_pipeline = ContentProcessingPipeline([
            "appropriate language", "factual accuracy", "coherence", "completeness"
        ])
        self.conversation_manager = ConversationManager(max_history=8)
        self.robust_processor = RobustProcessor(retry_attempts=2)
        self.optimized_processor = OptimizedProcessor(cache_size=100, batch_size=4)
        
        self.session_data = {
            'total_processed': 0,
            'session_start': datetime.now(),
            'processing_modes': ['conversation', 'batch', 'analysis', 'robust']
        }
    
    def process_conversation(self, message: str, user_id: str = "user") -> Dict[str, Any]:
        conv_result = self.conversation_manager(message, user_id)
        enhanced_result = self.content_pipeline(conv_result['response'], enhance_if_needed=True)
        
        self.session_data['total_processed'] += 1
        
        return {
            'mode': 'conversation',
            'response': enhanced_result['final_text'],
            'original_response': conv_result['response'],
            'was_enhanced': enhanced_result['was_enhanced'],
            'confidence': conv_result['confidence'],
            'conversation_state': self.conversation_manager.get_conversation_state()
        }
    
    def process_batch(self, texts: List[str]) -> Dict[str, Any]:
        batch_results = self.optimized_processor(texts)
        
        analyzed_results = []
        for result in batch_results:
            if not result.get('error'):
                analysis = self.content_pipeline(result['processed'], enhance_if_needed=False)
                analyzed_results.append({
                    **result,
                    'analysis': analysis['analysis'].metadata,
                    'validation': {
                        'is_valid': analysis['validation'].is_valid,
                        'issues_count': len(analysis['validation'].issues)
                    }
                })
            else:
                analyzed_results.append(result)
        
        self.session_data['total_processed'] += len(texts)
        
        return {
            'mode': 'batch',
            'results': analyzed_results,
            'batch_size': len(texts),
            'performance': self.optimized_processor.get_performance_metrics()
        }
    
    def process_robust(self, text: str, processing_mode: str = "detailed") -> Dict[str, Any]:
        robust_result = self.robust_processor(text, processing_mode)
        
        if robust_result['success']:
            analysis_result = self.content_pipeline(robust_result['output'], enhance_if_needed=True)
            robust_result['enhanced_analysis'] = {
                'final_content': analysis_result['final_text'],
                'was_enhanced': analysis_result['was_enhanced'],
                'sentiment': analysis_result['analysis'].metadata.get('sentiment'),
                'quality_score': analysis_result['analysis'].metadata.get('quality_score')
            }
        
        self.session_data['total_processed'] += 1
        
        return {
            'mode': 'robust',
            **robust_result
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            'session_duration': str(datetime.now() - self.session_data['session_start']),
            'total_processed': self.session_data['total_processed'],
            'pipeline_stats': self.content_pipeline.get_processing_stats(),
            'conversation_state': self.conversation_manager.get_conversation_state(),
            'robust_processor_stats': self.robust_processor.get_performance_stats(),
            'optimization_metrics': self.optimized_processor.get_performance_metrics()
        }
    
    def reset_session(self):
        self.session_data['total_processed'] = 0
        self.session_data['session_start'] = datetime.now()
        self.conversation_manager.reset_conversation()
        self.robust_processor.reset_stats()
        self.optimized_processor.clear_cache()

def main():
    """Demonstrate the advanced content manager."""
    print("=== Advanced Content Manager Demo ===")
    
    content_manager = AdvancedContentManager()
    
    # Test conversation mode
    print("\\n1. Conversation Mode:")
    conv_result = content_manager.process_conversation(
        "Hello! I'd like to learn about the latest developments in artificial intelligence."
    )
    print(f"Response: {conv_result['response'][:100]}...")
    print(f"Enhanced: {conv_result['was_enhanced']}")
    print(f"Confidence: {conv_result['confidence']:.2f}")
    
    # Test batch mode
    print("\\n2. Batch Processing Mode:")
    batch_texts = [
        "Explain the concept of machine learning",
        "What are the benefits of cloud computing?",
        "Describe data visualization techniques"
    ]
    batch_result = content_manager.process_batch(batch_texts)
    print(f"Processed {batch_result['batch_size']} texts")
    print(f"Cache hit rate: {batch_result['performance']['cache_hit_rate']:.2f}")
    
    for i, result in enumerate(batch_result['results']):
        print(f"  Text {i+1}: Success={not result.get('error')}, Cached={result.get('cached', False)}")
    
    # Test robust mode
    print("\\n3. Robust Processing Mode:")
    robust_result = content_manager.process_robust(
        "Analyze the environmental impact of renewable energy technologies",
        "detailed"
    )
    print(f"Success: {robust_result['success']}")
    print(f"Attempts: {robust_result['attempts']}")
    if robust_result['success'] and 'enhanced_analysis' in robust_result:
        analysis = robust_result['enhanced_analysis']
        print(f"Enhanced: {analysis['was_enhanced']}")
        print(f"Sentiment: {analysis['sentiment']}")
    
    # System status
    print("\\n4. System Status:")
    status = content_manager.get_system_status()
    print(f"Session duration: {status['session_duration']}")
    print(f"Total processed: {status['total_processed']}")
    print(f"Pipeline enhancement rate: {status['pipeline_stats'].get('enhancement_rate', 0):.2f}")
    print(f"Robust processor success rate: {status['robust_processor_stats']['success_rate']:.2f}")
    print(f"Cache hit rate: {status['optimization_metrics']['cache_hit_rate']:.2f}")

if __name__ == "__main__":
    main()
