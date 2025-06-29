#!/usr/bin/env python3
"""
Streaming with DSPy

This script demonstrates streaming responses and real-time text generation with DSPy,
including basic streaming, async streaming, and performance optimization.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import asyncio
import time
from typing import Generator, AsyncGenerator, List, Dict, Any
from datetime import datetime
import threading
from queue import Queue
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

class StreamingTextGenerator:
    """Basic streaming text generator using DSPy."""
    
    def __init__(self):
        self.predictor = dspy.Predict("prompt -> response")
    
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate streaming response for a given prompt."""
        
        try:
            # Get full response first (in real streaming, this would be incremental)
            result = self.predictor(prompt=prompt)
            response_text = result.response
            
            # Simulate streaming by yielding chunks
            words = response_text.split()
            current_chunk = ""
            
            for word in words:
                current_chunk += word + " "
                
                # Yield chunk when we have enough words
                if len(current_chunk.split()) >= 5:
                    yield current_chunk.strip()
                    current_chunk = ""
                    time.sleep(0.1)  # Simulate delay
            
            # Yield remaining chunk
            if current_chunk.strip():
                yield current_chunk.strip()
                
        except Exception as e:
            yield f"Error in streaming: {e}"

class AdvancedStreamingModule(dspy.Module):
    """Advanced DSPy module with streaming capabilities."""
    
    def __init__(self):
        super().__init__()
        self.qa_module = dspy.ChainOfThought("question -> answer")
        self.story_module = dspy.ChainOfThought("prompt -> story")
        self.explanation_module = dspy.ChainOfThought("topic, audience -> explanation")
        self.chunk_size = 5
        self.delay = 0.08
    
    def stream_qa(self, question: str) -> Generator[Dict[str, Any], None, None]:
        """Stream Q&A response with reasoning."""
        
        yield {'type': 'reasoning_start', 'content': 'Analyzing question...'}
        
        try:
            result = self.qa_module(question=question)
            
            # Stream reasoning if available
            if hasattr(result, 'reasoning') and result.reasoning:
                for chunk in self._chunk_text(result.reasoning):
                    yield {'type': 'reasoning', 'content': chunk, 'timestamp': datetime.now().isoformat()}
            
            yield {'type': 'answer_start', 'content': 'Providing answer...'}
            
            # Stream answer
            for chunk in self._chunk_text(result.answer):
                yield {'type': 'answer', 'content': chunk, 'timestamp': datetime.now().isoformat()}
            
            yield {'type': 'complete', 'content': 'Answer complete'}
            
        except Exception as e:
            yield {'type': 'error', 'content': f"Error: {e}"}
    
    def stream_explanation(self, topic: str, audience: str = "general") -> Generator[Dict[str, Any], None, None]:
        """Stream detailed explanation."""
        
        yield {'type': 'explanation_start', 'content': f'Preparing explanation of {topic}...'}
        
        try:
            result = self.explanation_module(topic=topic, audience=audience)
            
            for chunk in self._chunk_text(result.explanation):
                yield {'type': 'explanation', 'content': chunk, 'timestamp': datetime.now().isoformat()}
            
            yield {'type': 'explanation_complete', 'content': 'Explanation complete'}
            
        except Exception as e:
            yield {'type': 'error', 'content': f"Error: {e}"}
    
    def _chunk_text(self, text: str) -> Generator[str, None, None]:
        """Break text into streaming chunks."""
        words = text.split()
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            yield chunk
            time.sleep(self.delay)

class AsyncStreamingModule(dspy.Module):
    """Asynchronous streaming implementation for better performance."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("prompt -> response")
        self.chunk_size = 4
        self.delay = 0.05
    
    async def async_stream_response(self, prompt: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate asynchronous streaming response."""
        
        yield {
            'type': 'start',
            'content': 'Processing your request...',
            'timestamp': datetime.now().isoformat()
        }
        
        await asyncio.sleep(0.1)
        
        try:
            # Generate response (in practice, this would be async)
            result = self.predictor(prompt=prompt)
            
            # Stream reasoning if available
            if hasattr(result, 'reasoning') and result.reasoning:
                yield {
                    'type': 'reasoning_start',
                    'content': 'Thinking...',
                    'timestamp': datetime.now().isoformat()
                }
                
                async for chunk in self._async_chunk_text(result.reasoning):
                    yield {
                        'type': 'reasoning',
                        'content': chunk,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Stream response
            yield {
                'type': 'response_start',
                'content': 'Generating response...',
                'timestamp': datetime.now().isoformat()
            }
            
            async for chunk in self._async_chunk_text(result.response):
                yield {
                    'type': 'response',
                    'content': chunk,
                    'timestamp': datetime.now().isoformat()
                }
            
            yield {
                'type': 'complete',
                'content': 'Response complete',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            yield {
                'type': 'error',
                'content': f"Error: {e}",
                'timestamp': datetime.now().isoformat()
            }
    
    async def _async_chunk_text(self, text: str) -> AsyncGenerator[str, None]:
        """Asynchronously chunk text for streaming."""
        words = text.split()
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            yield chunk
            await asyncio.sleep(self.delay)
    
    async def batch_stream_responses(self, prompts: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses for multiple prompts concurrently."""
        
        yield {
            'type': 'batch_start',
            'content': f'Processing {len(prompts)} prompts...',
            'timestamp': datetime.now().isoformat()
        }
        
        # Process prompts concurrently
        tasks = []
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(self._process_single_prompt(i, prompt))
            tasks.append(task)
        
        # Stream results as they complete
        for task in asyncio.as_completed(tasks):
            result = await task
            yield result
        
        yield {
            'type': 'batch_complete',
            'content': 'All prompts processed',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _process_single_prompt(self, index: int, prompt: str) -> Dict[str, Any]:
        """Process a single prompt asynchronously."""
        
        start_time = time.time()
        
        try:
            result = self.predictor(prompt=prompt)
            
            return {
                'type': 'batch_result',
                'index': index,
                'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                'response': result.response[:100] + '...' if len(result.response) > 100 else result.response,
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'type': 'batch_error',
                'index': index,
                'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class StreamingChatInterface:
    """Interactive streaming chat interface using DSPy."""
    
    def __init__(self):
        self.conversation_history = []
        self.chat_module = dspy.ChainOfThought("conversation_history, user_message -> assistant_response")
        self.chunk_size = 3
        self.delay = 0.06
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append(f"{role}: {content}")
    
    def stream_chat_response(self, user_message: str) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming chat response."""
        
        # Add user message to history
        self.add_message('user', user_message)
        
        yield {
            'type': 'chat_start',
            'content': 'Assistant is typing...',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Generate response
            history_context = "\n".join(self.conversation_history[-6:])  # Last 3 exchanges
            result = self.chat_module(
                conversation_history=history_context,
                user_message=user_message
            )
            
            response_text = result.assistant_response
            
            yield {
                'type': 'response_start',
                'content': '',
                'timestamp': datetime.now().isoformat()
            }
            
            # Stream response in chunks
            words = response_text.split()
            for i in range(0, len(words), self.chunk_size):
                chunk = ' '.join(words[i:i + self.chunk_size])
                yield {
                    'type': 'response_chunk',
                    'content': chunk,
                    'timestamp': datetime.now().isoformat()
                }
                time.sleep(self.delay)
            
            # Add assistant response to history
            self.add_message('assistant', response_text)
            
            yield {
                'type': 'chat_complete',
                'content': 'Response complete',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            yield {
                'type': 'chat_error',
                'content': f"Error: {e}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary."""
        return {
            'total_messages': len(self.conversation_history),
            'conversation_start': self.conversation_history[0] if self.conversation_history else None,
            'last_activity': datetime.now().isoformat(),
            'message_count_by_role': {
                'user': len([msg for msg in self.conversation_history if msg.startswith('user:')]),
                'assistant': len([msg for msg in self.conversation_history if msg.startswith('assistant:')])
            }
        }

class OptimizedStreamingModule:
    """Performance-optimized streaming implementation."""
    
    def __init__(self):
        self.predictor = dspy.Predict("prompt -> response")
        
        # Performance tracking
        self.response_times = []
        self.chunk_counts = []
        self.total_requests = 0
        
        # Optimization settings
        self.adaptive_chunking = True
        self.min_chunk_size = 2
        self.max_chunk_size = 8
        self.base_delay = 0.05
    
    def optimized_stream(self, prompt: str) -> Generator[Dict[str, Any], None, None]:
        """Generate optimized streaming response."""
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Generate response
            result = self.predictor(prompt=prompt)
            response_text = result.response
            
            # Adaptive chunking based on content
            chunks = self._adaptive_chunk(response_text)
            
            # Stream with adaptive delay
            for i, chunk in enumerate(chunks):
                # Adaptive delay based on chunk position and size
                delay = self._calculate_adaptive_delay(i, len(chunks), len(chunk.split()))
                
                yield {
                    'type': 'chunk',
                    'content': chunk,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'delay': delay,
                    'timestamp': datetime.now().isoformat()
                }
                
                time.sleep(delay)
            
            # Track performance
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.chunk_counts.append(len(chunks))
            
            yield {
                'type': 'optimization_stats',
                'content': f'Response time: {response_time:.2f}s, Chunks: {len(chunks)}',
                'performance': self.get_performance_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            yield {
                'type': 'error',
                'content': f"Error: {e}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _adaptive_chunk(self, text: str) -> List[str]:
        """Adaptively chunk text based on content."""
        if not self.adaptive_chunking:
            # Fixed chunking
            words = text.split()
            return [' '.join(words[i:i + self.max_chunk_size]) 
                   for i in range(0, len(words), self.max_chunk_size)]
        
        # Adaptive chunking based on sentence boundaries
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            current_words = len(current_chunk.split())
            
            if (current_words + sentence_words <= self.max_chunk_size and 
                current_words >= self.min_chunk_size):
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _calculate_adaptive_delay(self, chunk_index: int, total_chunks: int, chunk_size: int) -> float:
        """Calculate adaptive delay based on context."""
        # Base delay
        delay = self.base_delay
        
        # Adjust for chunk size
        if chunk_size < self.min_chunk_size:
            delay *= 0.5  # Shorter delay for small chunks
        elif chunk_size > self.max_chunk_size:
            delay *= 1.5  # Longer delay for large chunks
        
        # Adjust for position in response
        progress = chunk_index / max(1, total_chunks - 1)
        if progress < 0.3:  # Early chunks - faster
            delay *= 0.8
        elif progress > 0.7:  # Later chunks - slower for emphasis
            delay *= 1.2
        
        return max(0.01, delay)  # Minimum delay
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.response_times:
            return {'status': 'no_data'}
        
        return {
            'total_requests': self.total_requests,
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'avg_chunk_count': sum(self.chunk_counts) / len(self.chunk_counts),
            'total_chunks_processed': sum(self.chunk_counts)
        }

def demonstrate_basic_streaming():
    """Demonstrate basic streaming functionality."""
    
    print_step("Basic Streaming", "Testing word-by-word streaming")
    
    generator = StreamingTextGenerator()
    
    test_prompts = [
        "Explain the concept of machine learning",
        "Describe the benefits of renewable energy",
        "What is artificial intelligence?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt}")
        print("Response: ", end="")
        
        for chunk in generator.generate_stream(prompt):
            print(chunk, end=" ", flush=True)
        
        print("\n")

def demonstrate_advanced_streaming():
    """Demonstrate advanced streaming with metadata."""
    
    print_step("Advanced Streaming", "Testing structured streaming with metadata")
    
    advanced_streaming = AdvancedStreamingModule()
    
    # Test Q&A streaming
    print("\n1. Streaming Q&A with Reasoning:")
    print("-" * 40)
    
    qa_question = "How do neural networks learn from data?"
    print(f"Question: {qa_question}\n")
    
    for stream_chunk in advanced_streaming.stream_qa(qa_question):
        if stream_chunk['type'] == 'reasoning_start':
            print(f"🤔 {stream_chunk['content']}")
        elif stream_chunk['type'] == 'reasoning':
            print(f"   {stream_chunk['content']}", end=" ")
        elif stream_chunk['type'] == 'answer_start':
            print(f"\n\n💡 {stream_chunk['content']}")
        elif stream_chunk['type'] == 'answer':
            print(f"   {stream_chunk['content']}", end=" ")
        elif stream_chunk['type'] == 'complete':
            print(f"\n\n✅ {stream_chunk['content']}")
    
    # Test explanation streaming
    print("\n\n2. Streaming Technical Explanation:")
    print("-" * 40)
    
    for stream_chunk in advanced_streaming.stream_explanation("quantum computing", "beginner"):
        if stream_chunk['type'] == 'explanation_start':
            print(f"🎯 {stream_chunk['content']}\n")
        elif stream_chunk['type'] == 'explanation':
            print(f"{stream_chunk['content']}", end=" ", flush=True)
        elif stream_chunk['type'] == 'explanation_complete':
            print(f"\n\n✅ {stream_chunk['content']}")

async def demonstrate_async_streaming():
    """Demonstrate asynchronous streaming."""
    
    print_step("Async Streaming", "Testing asynchronous response generation")
    
    async_streaming = AsyncStreamingModule()
    
    # Test single async stream
    print("\n1. Single Async Stream:")
    print("-" * 30)
    
    async_prompt = "Explain the benefits of asynchronous programming"
    print(f"Prompt: {async_prompt}\n")
    
    async for chunk in async_streaming.async_stream_response(async_prompt):
        if chunk['type'] == 'start':
            print(f"🚀 {chunk['content']}")
        elif chunk['type'] == 'reasoning_start':
            print(f"\n🤔 {chunk['content']}")
        elif chunk['type'] == 'reasoning':
            print(f"   {chunk['content']}", end=" ")
        elif chunk['type'] == 'response_start':
            print(f"\n\n💬 {chunk['content']}")
        elif chunk['type'] == 'response':
            print(f"   {chunk['content']}", end=" ")
        elif chunk['type'] == 'complete':
            print(f"\n\n✅ {chunk['content']}")
    
    # Test batch streaming
    print("\n\n2. Batch Async Streaming:")
    print("-" * 30)
    
    batch_prompts = [
        "What is machine learning?",
        "Explain blockchain technology",
        "Describe cloud computing benefits",
        "What is artificial intelligence?"
    ]
    
    print(f"Processing {len(batch_prompts)} prompts concurrently...\n")
    
    async for result in async_streaming.batch_stream_responses(batch_prompts):
        if result['type'] == 'batch_start':
            print(f"🔄 {result['content']}")
        elif result['type'] == 'batch_result':
            print(f"\n✅ Result {result['index'] + 1}:")
            print(f"   Prompt: {result['prompt']}")
            print(f"   Response: {result['response']}")
            print(f"   Duration: {result['duration']:.2f}s")
        elif result['type'] == 'batch_error':
            print(f"\n❌ Error {result['index'] + 1}:")
            print(f"   Prompt: {result['prompt']}")
            print(f"   Error: {result['error']}")
        elif result['type'] == 'batch_complete':
            print(f"\n🎉 {result['content']}")

def demonstrate_streaming_chat():
    """Demonstrate streaming chat interface."""
    
    print_step("Streaming Chat", "Testing conversational streaming interface")
    
    chat_interface = StreamingChatInterface()
    
    # Simulate a conversation
    chat_messages = [
        "Hello! Can you explain what DSPy is?",
        "That's interesting! How does it differ from LangChain?",
        "Can you give me a practical example of using DSPy?",
        "Thank you! This has been very helpful."
    ]
    
    print("🤖 Starting streaming chat demonstration...\n")
    
    for i, message in enumerate(chat_messages, 1):
        print(f"👤 User {i}: {message}")
        print("🤖 Assistant: ", end="")
        
        for chunk in chat_interface.stream_chat_response(message):
            if chunk['type'] == 'chat_start':
                print(f"({chunk['content']}) ", end="")
            elif chunk['type'] == 'response_chunk':
                print(chunk['content'], end=" ", flush=True)
            elif chunk['type'] == 'chat_complete':
                print("\n")
        
        time.sleep(0.5)  # Pause between messages
    
    # Show conversation summary
    summary = chat_interface.get_conversation_summary()
    print(f"\n📊 Conversation Summary:")
    print_result(f"Total messages: {summary['total_messages']}")
    print_result(f"User messages: {summary['message_count_by_role']['user']}")
    print_result(f"Assistant messages: {summary['message_count_by_role']['assistant']}")

def demonstrate_optimized_streaming():
    """Demonstrate performance-optimized streaming."""
    
    print_step("Optimized Streaming", "Testing performance optimization features")
    
    optimized_streaming = OptimizedStreamingModule()
    
    # Test different optimization settings
    test_prompts = [
        "Explain the basics of artificial intelligence",
        "Describe machine learning algorithms in detail",
        "What are the applications of deep learning?"
    ]
    
    print("\n1. Testing with Adaptive Chunking:")
    print("-" * 40)
    
    optimized_streaming.adaptive_chunking = True
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("Response: ", end="")
        
        chunk_details = []
        
        for chunk_data in optimized_streaming.optimized_stream(prompt):
            if chunk_data['type'] == 'chunk':
                print(chunk_data['content'], end=" ", flush=True)
                chunk_details.append({
                    'index': chunk_data['chunk_index'],
                    'delay': chunk_data['delay']
                })
            elif chunk_data['type'] == 'optimization_stats':
                print(f"\n   📊 {chunk_data['content']}")
        
        print()
    
    # Show final performance stats
    final_stats = optimized_streaming.get_performance_stats()
    print(f"\n📈 Final Performance Statistics:")
    print_result(f"Total requests: {final_stats['total_requests']}")
    print_result(f"Average response time: {final_stats['avg_response_time']:.3f}s")
    print_result(f"Average chunks per response: {final_stats['avg_chunk_count']:.1f}")

def main():
    """Main function demonstrating streaming with DSPy."""
    print("=" * 70)
    print("STREAMING WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for streaming")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=1000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Run demonstrations
    try:
        # Demo 1: Basic Streaming
        demonstrate_basic_streaming()
        
        # Demo 2: Advanced Streaming
        demonstrate_advanced_streaming()
        
        # Demo 3: Async Streaming
        print_step("Running Async Streaming Tests", "Note: May require specific environment setup")
        try:
            asyncio.run(demonstrate_async_streaming())
        except Exception as e:
            print_error(f"Async streaming demo failed: {e}")
            print("Note: Async features may require specific environment setup")
        
        # Demo 4: Streaming Chat
        demonstrate_streaming_chat()
        
        # Demo 5: Optimized Streaming
        demonstrate_optimized_streaming()
        
        print("\n" + "=" * 70)
        print("🎉 STREAMING DEMONSTRATION COMPLETED!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("✅ Basic streaming: Simple word-by-word generation")
        print("✅ Advanced streaming: Structured metadata and reasoning")
        print("✅ Async streaming: Concurrent processing capabilities")
        print("✅ Chat interface: Conversational streaming with memory")
        print("✅ Performance optimization: Adaptive chunking and timing")
        
    except Exception as e:
        print_error(f"Error in streaming demonstration: {e}")

if __name__ == "__main__":
    main()
