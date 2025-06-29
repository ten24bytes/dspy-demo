#!/usr/bin/env python3
"""
Building RAG as Agent with DSPy - Python Script Version

This script demonstrates how to build intelligent agents that combine 
Retrieval-Augmented Generation (RAG) with advanced reasoning capabilities using DSPy.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
from utils import setup_default_lm, print_step, print_result, print_error
from utils.datasets import get_sample_rag_documents
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import random
from datetime import datetime
from collections import defaultdict

def main():
    """Main function demonstrating RAG as Agent with DSPy."""
    
    print("=" * 70)
    print("BUILDING RAG AS AGENT WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for RAG Agent development")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Data structures
    @dataclass
    class AgentMemory:
        """Memory structure for the RAG agent."""
        conversation_history: List[Dict[str, str]]
        retrieved_context: List[str]
        reasoning_traces: List[str]
        action_history: List[Dict[str, Any]]
        learned_facts: List[str]

    @dataclass
    class RetrievalResult:
        """Enhanced retrieval result with metadata."""
        content: str
        relevance_score: float
        source: str
        timestamp: str
        confidence: float
    
    # Document Retrieval System
    class IntelligentRetriever:
        """Advanced retrieval system with semantic understanding."""
        
        def __init__(self, documents: List[str]):
            self.documents = documents
            self.document_metadata = {}
            self._initialize_metadata()
        
        def _initialize_metadata(self):
            """Initialize document metadata for better retrieval."""
            for i, doc in enumerate(self.documents):
                self.document_metadata[i] = {
                    'length': len(doc),
                    'keywords': self._extract_keywords(doc),
                    'topic': self._infer_topic(doc),
                    'complexity': self._estimate_complexity(doc)
                }
        
        def _extract_keywords(self, text: str) -> List[str]:
            """Simple keyword extraction."""
            words = text.lower().split()
            return [word for word in words if len(word) > 4][:10]
        
        def _infer_topic(self, text: str) -> str:
            """Infer document topic."""
            text_lower = text.lower()
            if 'machine learning' in text_lower or 'ai' in text_lower:
                return 'technology'
            elif 'science' in text_lower or 'research' in text_lower:
                return 'science'
            elif 'business' in text_lower or 'market' in text_lower:
                return 'business'
            else:
                return 'general'
        
        def _estimate_complexity(self, text: str) -> float:
            """Estimate text complexity."""
            avg_word_length = np.mean([len(word) for word in text.split()])
            return min(avg_word_length / 10.0, 1.0)
        
        def semantic_search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
            """Perform semantic search with enhanced scoring."""
            results = []
            query_lower = query.lower()
            
            for i, doc in enumerate(self.documents):
                doc_lower = doc.lower()
                
                # Keyword overlap
                query_words = set(query_lower.split())
                doc_words = set(doc_lower.split())
                overlap = len(query_words.intersection(doc_words))
                
                # Basic relevance score
                relevance = overlap / len(query_words) if query_words else 0
                
                # Boost score based on metadata
                if any(keyword in doc_lower for keyword in query_words):
                    relevance *= 1.2
                
                if relevance > 0:
                    results.append(RetrievalResult(
                        content=doc,
                        relevance_score=relevance,
                        source=f"doc_{i}",
                        timestamp=datetime.now().isoformat(),
                        confidence=min(relevance * 0.8, 1.0)
                    ))
            
            # Sort by relevance and return top_k
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:top_k]
        
        def adaptive_retrieve(self, query: str, context: List[str], memory: AgentMemory) -> List[RetrievalResult]:
            """Adaptive retrieval that considers context and memory."""
            # Expand query based on conversation history
            expanded_query = query
            if memory.conversation_history:
                recent_context = " ".join([msg['content'] for msg in memory.conversation_history[-3:]])
                expanded_query = f"{query} {recent_context}"
            
            # Get initial results
            results = self.semantic_search(expanded_query, top_k=5)
            
            # Filter out already retrieved content
            seen_content = set(memory.retrieved_context)
            filtered_results = [r for r in results if r.content not in seen_content]
            
            return filtered_results[:3]
    
    # DSPy Signatures for RAG Agent
    class QueryAnalysis(dspy.Signature):
        """Analyze user query to determine optimal agent strategy."""
        
        query = dspy.InputField(desc="User's question or request")
        conversation_context = dspy.InputField(desc="Previous conversation context")
        
        query_type = dspy.OutputField(desc="Type of query: factual, analytical, creative, or procedural")
        complexity_level = dspy.OutputField(desc="Complexity level: simple, moderate, or complex")
        required_actions = dspy.OutputField(desc="List of actions needed to answer the query")
        retrieval_strategy = dspy.OutputField(desc="Best retrieval strategy for this query")

    class ActionPlanning(dspy.Signature):
        """Plan the sequence of actions to answer a query."""
        
        query = dspy.InputField(desc="User's question")
        query_analysis = dspy.InputField(desc="Analysis of the query")
        available_context = dspy.InputField(desc="Currently available context and information")
        
        action_plan = dspy.OutputField(desc="Step-by-step plan to answer the query")
        priority_actions = dspy.OutputField(desc="Most important actions to take first")
        fallback_strategy = dspy.OutputField(desc="Alternative approach if primary plan fails")

    class InformationSynthesis(dspy.Signature):
        """Synthesize information from multiple sources into a coherent response."""
        
        query = dspy.InputField(desc="Original user query")
        retrieved_docs = dspy.InputField(desc="Retrieved documents and information")
        conversation_memory = dspy.InputField(desc="Relevant conversation history")
        
        synthesized_answer = dspy.OutputField(desc="Comprehensive answer synthesized from sources")
        confidence_level = dspy.OutputField(desc="Confidence level in the answer")
        information_gaps = dspy.OutputField(desc="Identified gaps in available information")
        follow_up_suggestions = dspy.OutputField(desc="Suggested follow-up questions or actions")
    
    # Advanced RAG Agent Implementation
    class AdvancedRAGAgent(dspy.Module):
        """Advanced RAG agent with multi-step reasoning and memory."""
        
        def __init__(self, retriever: IntelligentRetriever):
            super().__init__()
            self.retriever = retriever
            
            # Initialize DSPy modules
            self.query_analyzer = dspy.ChainOfThought(QueryAnalysis)
            self.action_planner = dspy.ChainOfThought(ActionPlanning)
            self.information_synthesizer = dspy.ChainOfThought(InformationSynthesis)
            
            # Agent memory
            self.memory = AgentMemory(
                conversation_history=[],
                retrieved_context=[],
                reasoning_traces=[],
                action_history=[],
                learned_facts=[]
            )
            
            # Performance tracking
            self.interaction_count = 0
            self.success_rate = 0.0
        
        def forward(self, query: str) -> dspy.Prediction:
            """Main agent reasoning loop."""
            
            self.interaction_count += 1
            
            # Step 1: Analyze the query
            conversation_context = self._get_conversation_context()
            query_analysis = self.query_analyzer(
                query=query,
                conversation_context=conversation_context
            )
            
            # Step 2: Plan actions
            available_context = self._get_available_context()
            action_plan = self.action_planner(
                query=query,
                query_analysis=f"Type: {query_analysis.query_type}, Complexity: {query_analysis.complexity_level}",
                available_context=available_context
            )
            
            # Step 3: Execute retrieval strategy
            retrieval_result = self._execute_retrieval(query, query_analysis, action_plan)
            
            # Step 4: Synthesize information
            synthesis = self.information_synthesizer(
                query=query,
                retrieved_docs=retrieval_result['formatted_docs'],
                conversation_memory=conversation_context
            )
            
            # Step 5: Update memory
            self._update_memory(query, synthesis, query_analysis)
            
            # Step 6: Evaluate and learn
            self._evaluate_interaction(query, synthesis)
            
            return dspy.Prediction(
                answer=synthesis.synthesized_answer,
                confidence=synthesis.confidence_level,
                query_type=query_analysis.query_type,
                action_plan=action_plan.action_plan,
                retrieved_sources=retrieval_result['sources'],
                information_gaps=synthesis.information_gaps,
                follow_up_suggestions=synthesis.follow_up_suggestions,
                reasoning_trace=self._get_reasoning_trace(query_analysis, action_plan, synthesis)
            )
        
        def _get_conversation_context(self) -> str:
            """Get recent conversation context."""
            if not self.memory.conversation_history:
                return "No previous conversation context."
            
            recent_history = self.memory.conversation_history[-3:]
            context = "\\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_history
            ])
            return context
        
        def _get_available_context(self) -> str:
            """Get currently available context information."""
            context_parts = []
            
            if self.memory.retrieved_context:
                context_parts.append(f"Retrieved context: {len(self.memory.retrieved_context)} documents")
            
            if self.memory.learned_facts:
                context_parts.append(f"Learned facts: {len(self.memory.learned_facts)} items")
            
            if self.memory.reasoning_traces:
                context_parts.append(f"Previous reasoning: {len(self.memory.reasoning_traces)} traces")
            
            return "; ".join(context_parts) if context_parts else "No available context."
        
        def _execute_retrieval(self, query: str, query_analysis, action_plan) -> Dict[str, Any]:
            """Execute the retrieval strategy."""
            
            # Perform adaptive retrieval
            retrieved_docs = self.retriever.adaptive_retrieve(query, [], self.memory)
            
            # Format documents for synthesis
            formatted_docs = "\\n\\n".join([
                f"Source {i+1} (relevance: {doc.relevance_score:.2f}):\\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Update memory with retrieved context
            for doc in retrieved_docs:
                if doc.content not in self.memory.retrieved_context:
                    self.memory.retrieved_context.append(doc.content)
            
            return {
                'formatted_docs': formatted_docs,
                'sources': [doc.source for doc in retrieved_docs]
            }
        
        def _update_memory(self, query: str, synthesis, query_analysis):
            """Update agent memory with new interaction."""
            
            # Add to conversation history
            self.memory.conversation_history.append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now().isoformat()
            })
            
            self.memory.conversation_history.append({
                'role': 'assistant',
                'content': synthesis.synthesized_answer,
                'timestamp': datetime.now().isoformat()
            })
            
            # Add reasoning trace
            reasoning_trace = f"Query: {query} | Type: {query_analysis.query_type} | Answer: {synthesis.synthesized_answer[:100]}..."
            self.memory.reasoning_traces.append(reasoning_trace)
        
        def _evaluate_interaction(self, query: str, synthesis):
            """Evaluate and learn from the interaction."""
            
            # Simple confidence-based evaluation
            try:
                confidence_value = float(synthesis.confidence_level.split('%')[0] if '%' in synthesis.confidence_level else synthesis.confidence_level)
                success = confidence_value > 70
            except:
                success = len(synthesis.synthesized_answer) > 50  # Fallback heuristic
            
            # Update success rate
            current_success = 1.0 if success else 0.0
            self.success_rate = (self.success_rate * (self.interaction_count - 1) + current_success) / self.interaction_count
            
            # Log action
            self.memory.action_history.append({
                'query': query,
                'success': success,
                'confidence': synthesis.confidence_level,
                'timestamp': datetime.now().isoformat()
            })
        
        def _get_reasoning_trace(self, query_analysis, action_plan, synthesis) -> str:
            """Generate a reasoning trace for transparency."""
            
            trace_parts = [
                f"1. Query Analysis: {query_analysis.query_type} query with {query_analysis.complexity_level} complexity",
                f"2. Action Planning: {action_plan.action_plan}",
                f"3. Information Synthesis: {synthesis.confidence_level} confidence",
                f"4. Identified Gaps: {synthesis.information_gaps}"
            ]
            
            return "\\n".join(trace_parts)
        
        def get_agent_status(self) -> Dict[str, Any]:
            """Get current agent status and performance metrics."""
            
            return {
                'interaction_count': self.interaction_count,
                'success_rate': self.success_rate,
                'memory_size': {
                    'conversation_history': len(self.memory.conversation_history),
                    'retrieved_context': len(self.memory.retrieved_context),
                    'reasoning_traces': len(self.memory.reasoning_traces),
                    'learned_facts': len(self.memory.learned_facts)
                },
                'last_interactions': self.memory.action_history[-5:] if self.memory.action_history else []
            }
    
    # Initialize components
    print_step("Initializing Components", "Setting up retriever and agent")
    
    # Initialize retriever with sample documents
    documents = get_sample_rag_documents()
    retriever = IntelligentRetriever(documents)
    print_result(f"Initialized retriever with {len(documents)} documents")
    
    # Initialize the RAG agent
    rag_agent = AdvancedRAGAgent(retriever)
    print_result("Advanced RAG Agent initialized successfully!")
    
    # Demo 1: Basic RAG Agent Testing
    print_step("Basic RAG Agent Testing", "Demonstrating agent capabilities with various queries")
    
    test_queries = [
        "What is machine learning and how does it work?",
        "Can you explain the business applications of AI?",
        "How do neural networks process information?",
        "What are the ethical considerations in AI development?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        try:
            result = rag_agent(query)
            
            print_result(f"Answer: {result.answer}", "Response")
            print_result(f"Confidence: {result.confidence}", "Confidence")
            print_result(f"Query Type: {result.query_type}", "Analysis")
            print_result(f"Sources: {', '.join(result.retrieved_sources)}", "Sources")
            
            if result.information_gaps:
                print_result(f"Information Gaps: {result.information_gaps}", "Gaps")
            
            if result.follow_up_suggestions:
                print_result(f"Follow-up Suggestions: {result.follow_up_suggestions}", "Suggestions")
            
            print("\\nReasoning Trace:")
            print(result.reasoning_trace)
            
        except Exception as e:
            print_error(f"Error processing query {i}: {e}")
    
    # Demo 2: Multi-turn Conversation
    print_step("Multi-turn Conversation Demo", "Testing context-aware responses")
    
    conversation_queries = [
        "What is machine learning?",
        "Can you give me specific examples?",
        "How does this compare to traditional programming?",
        "What are the main challenges in implementing ML?"
    ]
    
    print("\\nMulti-turn Conversation:")
    print("-" * 40)
    
    for i, query in enumerate(conversation_queries, 1):
        print(f"\\nTurn {i}: {query}")
        result = rag_agent(query)
        print(f"Agent: {result.answer[:150]}...")
        if result.follow_up_suggestions:
            print(f"Suggestions: {result.follow_up_suggestions}")
    
    # Demo 3: Complex Analytical Query
    print_step("Complex Analysis Demo", "Testing analytical capabilities")
    
    complex_query = "Analyze the relationship between artificial intelligence, machine learning, and deep learning. How do they differ and how do they work together in modern applications?"
    result = rag_agent(complex_query)
    
    print_result(f"Query Type: {result.query_type}", "Analysis")
    print_result(f"Answer: {result.answer}", "Response")
    print_result(f"Confidence: {result.confidence}", "Confidence")
    print_result(f"Sources Used: {len(result.retrieved_sources)}", "Sources")
    
    # Demo 4: Information Gap Identification
    print_step("Gap Identification Demo", "Testing limitation recognition")
    
    gap_query = "What are the latest developments in quantum machine learning and their implications for cryptography?"
    result = rag_agent(gap_query)
    
    print_result(f"Answer: {result.answer}", "Response")
    if result.information_gaps:
        print_result(f"Identified Gaps: {result.information_gaps}", "Gaps")
    if result.follow_up_suggestions:
        print_result(f"Suggestions: {result.follow_up_suggestions}", "Suggestions")
    
    # Demo 5: Agent Performance Summary
    print_step("Agent Performance Summary", "Analyzing agent effectiveness")
    
    status = rag_agent.get_agent_status()
    
    print_result(f"Total Interactions: {status['interaction_count']}", "Statistics")
    print_result(f"Success Rate: {status['success_rate']:.2%}", "Performance")
    print_result(f"Memory Usage: {status['memory_size']}", "Memory")
    
    if status['last_interactions']:
        print("\\nRecent Interactions:")
        for interaction in status['last_interactions']:
            print(f"  - {interaction['query'][:50]}... | Success: {interaction['success']} | Confidence: {interaction['confidence']}")
    
    print("\\n🎉 RAG as Agent demonstration completed successfully!")
    print("\\nKey Features Demonstrated:")
    print("- Intelligent document retrieval with relevance scoring")
    print("- Multi-step reasoning with query analysis and action planning")
    print("- Memory management across conversation turns")
    print("- Confidence scoring and gap identification")
    print("- Performance tracking and agent introspection")

if __name__ == "__main__":
    main()
