#!/usr/bin/env python3
"""
Async Processing with DSPy

This script demonstrates asynchronous processing capabilities in DSPy for building
scalable, non-blocking AI applications with concurrent request processing.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import asyncio
import time
from typing import List, Dict, Any, Optional, Coroutine
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

class AsyncDSPyModule(dspy.Module):
    """Basic async-enabled DSPy module."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("prompt -> response")
        self.chain_of_thought = dspy.ChainOfThought("question -> answer")
        
    def forward(self, **kwargs):
        """Synchronous forward pass."""
        return self.predictor(**kwargs)
    
    async def async_forward(self, **kwargs):
        """Asynchronous forward pass."""
        # Run DSPy prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                lambda: self.predictor(**kwargs)
            )
        return result
    
    async def async_chain_of_thought(self, question: str):
        """Async chain of thought reasoning."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.chain_of_thought(question=question)
            )
        return result

class AsyncBatchProcessor(dspy.Module):
    """Advanced async batch processing module."""
    
    def __init__(self, max_concurrent: int = 5):
        super().__init__()
        self.qa_module = dspy.ChainOfThought("question -> answer")
        self.summarizer = dspy.Predict("text -> summary")
        self.classifier = dspy.Predict("text -> category")
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_item(self, item: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Process a single item asynchronously."""
        
        async with self.semaphore:  # Limit concurrent operations
            start_time = time.time()
            
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    if task_type == "qa":
                        result = await loop.run_in_executor(
                            executor,
                            lambda: self.qa_module(question=item["question"])
                        )
                        output = result.answer
                    elif task_type == "summarize":
                        result = await loop.run_in_executor(
                            executor,
                            lambda: self.summarizer(text=item["text"])
                        )
                        output = result.summary
                    elif task_type == "classify":
                        result = await loop.run_in_executor(
                            executor,
                            lambda: self.classifier(text=item["text"])
                        )
                        output = result.category
                    else:
                        raise ValueError(f"Unknown task type: {task_type}")
                
                return {
                    "id": item.get("id", "unknown"),
                    "input": item,
                    "output": output,
                    "task_type": task_type,
                    "processing_time": time.time() - start_time,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "id": item.get("id", "unknown"),
                    "input": item,
                    "output": None,
                    "task_type": task_type,
                    "processing_time": time.time() - start_time,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    async def batch_process(self, items: List[Dict[str, Any]], task_type: str) -> List[Dict[str, Any]]:
        """Process multiple items asynchronously."""
        
        print(f"Processing {len(items)} items of type '{task_type}' with max concurrency {self.max_concurrent}")
        
        # Create tasks for all items
        tasks = [
            asyncio.create_task(
                self.process_single_item(item, task_type),
                name=f"{task_type}_{item.get('id', i)}"
            )
            for i, item in enumerate(items)
        ]
        
        # Process tasks as they complete
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            
            if completed % 2 == 0 or completed == len(tasks):
                print(f"Completed {completed}/{len(tasks)} tasks")
        
        return results

class RobustAsyncModule(dspy.Module):
    """Async module with robust error handling and retry logic."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__()
        self.predictor = dspy.Predict("prompt -> response")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retries_performed": 0
        }
    
    async def robust_forward(self, prompt: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Forward with error handling, retries, and timeout."""
        
        self.stats["total_requests"] += 1
        
        for attempt in range(self.max_retries + 1):
            try:
                # Run with timeout
                result = await asyncio.wait_for(
                    self._async_predict(prompt),
                    timeout=timeout
                )
                
                self.stats["successful_requests"] += 1
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                    "prompt": prompt[:50] + "...",
                    "timestamp": datetime.now().isoformat()
                }
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {timeout}s on attempt {attempt + 1}"
                if attempt < self.max_retries:
                    print(f"⏰ {error_msg}, retrying...")
                    self.stats["retries_performed"] += 1
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    self.stats["failed_requests"] += 1
                    return {
                        "success": False,
                        "error": error_msg,
                        "attempts": attempt + 1,
                        "prompt": prompt[:50] + "...",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                error_msg = f"Error on attempt {attempt + 1}: {str(e)}"
                if attempt < self.max_retries:
                    print(f"❌ {error_msg}, retrying...")
                    self.stats["retries_performed"] += 1
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    self.stats["failed_requests"] += 1
                    return {
                        "success": False,
                        "error": error_msg,
                        "attempts": attempt + 1,
                        "prompt": prompt[:50] + "...",
                        "timestamp": datetime.now().isoformat()
                    }
    
    async def _async_predict(self, prompt: str):
        """Run prediction in thread pool."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.predictor(prompt=prompt)
            )
        return result
    
    async def batch_robust_forward(self, prompts: List[str], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Process multiple prompts with error handling."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.robust_forward(prompt)
        
        # Create tasks
        tasks = [
            asyncio.create_task(process_with_semaphore(prompt))
            for prompt in prompts
        ]
        
        # Wait for all to complete (don't fail on individual errors)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any task-level exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": f"Task exception: {str(result)}",
                    "prompt": prompts[i][:50] + "...",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and reliability statistics."""
        success_rate = (self.stats["successful_requests"] / max(1, self.stats["total_requests"])) * 100
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "avg_retries_per_request": self.stats["retries_performed"] / max(1, self.stats["total_requests"])
        }

class AsyncPerformanceMonitor:
    """Monitor performance of async DSPy operations."""
    
    def __init__(self):
        self.metrics = {
            "request_times": [],
            "queue_times": [],
            "concurrent_requests": [],
            "throughput_samples": [],
            "error_counts": 0,
            "total_requests": 0
        }
        self.start_time = time.time()
    
    async def monitored_request(self, async_func, *args, **kwargs):
        """Wrap an async function with performance monitoring."""
        
        request_start = time.time()
        
        self.metrics["total_requests"] += 1
        
        try:
            result = await async_func(*args, **kwargs)
            request_time = time.time() - request_start
            
            self.metrics["request_times"].append(request_time)
            
            return result
            
        except Exception as e:
            self.metrics["error_counts"] += 1
            raise e
    
    async def benchmark_concurrency(self, async_func, test_data: List[Any], 
                                  concurrency_levels: List[int]) -> Dict[str, Dict[str, float]]:
        """Benchmark different concurrency levels."""
        
        results = {}
        
        for concurrency in concurrency_levels:
            print(f"\nTesting concurrency level: {concurrency}")
            
            # Reset metrics for this test
            self.metrics["request_times"] = []
            self.metrics["error_counts"] = 0
            self.metrics["total_requests"] = 0
            
            semaphore = asyncio.Semaphore(concurrency)
            
            async def limited_request(data):
                async with semaphore:
                    return await self.monitored_request(async_func, **data)
            
            # Run benchmark
            start_time = time.time()
            
            tasks = [asyncio.create_task(limited_request(data)) for data in test_data]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            if self.metrics["request_times"]:
                avg_request_time = sum(self.metrics["request_times"]) / len(self.metrics["request_times"])
                throughput = len(test_data) / total_time
                success_rate = ((self.metrics["total_requests"] - self.metrics["error_counts"]) / 
                              self.metrics["total_requests"]) * 100
            else:
                avg_request_time = 0
                throughput = 0
                success_rate = 0
            
            results[f"concurrency_{concurrency}"] = {
                "total_time": total_time,
                "avg_request_time": avg_request_time,
                "throughput": throughput,
                "success_rate": success_rate,
                "total_requests": self.metrics["total_requests"],
                "errors": self.metrics["error_counts"]
            }
            
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.2f} requests/second")
            print(f"  Success rate: {success_rate:.1f}%")
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        if not self.metrics["request_times"]:
            return {"status": "no_data"}
        
        request_times = self.metrics["request_times"]
        
        return {
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["error_counts"],
            "success_rate": ((self.metrics["total_requests"] - self.metrics["error_counts"]) / 
                           self.metrics["total_requests"]) * 100,
            "avg_request_time": sum(request_times) / len(request_times),
            "min_request_time": min(request_times),
            "max_request_time": max(request_times),
            "total_runtime": time.time() - self.start_time
        }

async def test_basic_async():
    """Test basic asynchronous operations."""
    
    async_module = AsyncDSPyModule()
    
    test_prompts = [
        "Explain machine learning",
        "What is artificial intelligence?",
        "Describe neural networks"
    ]
    
    # Test 1: Synchronous processing
    print("\n1. Synchronous Processing:")
    print("-" * 30)
    
    sync_start = time.time()
    sync_results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"Processing prompt {i+1}: {prompt}")
        result = async_module.forward(prompt=prompt)
        sync_results.append(result.response[:100] + "...")
        print(f"Result: {sync_results[-1]}\n")
    
    sync_time = time.time() - sync_start
    print(f"Synchronous processing time: {sync_time:.2f} seconds")
    
    # Test 2: Asynchronous processing
    print("\n2. Asynchronous Processing:")
    print("-" * 30)
    
    async_start = time.time()
    
    # Create async tasks
    tasks = []
    for i, prompt in enumerate(test_prompts):
        task = asyncio.create_task(
            async_module.async_forward(prompt=prompt),
            name=f"task_{i+1}"
        )
        tasks.append(task)
        print(f"Started task {i+1}: {prompt}")
    
    # Wait for all tasks to complete
    async_results = await asyncio.gather(*tasks)
    
    async_time = time.time() - async_start
    
    print("\nAsync Results:")
    for i, result in enumerate(async_results):
        print(f"Task {i+1}: {result.response[:100]}...")
    
    print(f"\nAsynchronous processing time: {async_time:.2f} seconds")
    print(f"Speedup: {sync_time/async_time:.2f}x")

async def test_batch_processing():
    """Test batch processing capabilities."""
    
    batch_processor = AsyncBatchProcessor(max_concurrent=3)
    
    # Prepare test data
    qa_items = [
        {"id": "qa_1", "question": "What is machine learning?"},
        {"id": "qa_2", "question": "How does deep learning work?"},
        {"id": "qa_3", "question": "What are neural networks?"},
        {"id": "qa_4", "question": "Explain artificial intelligence"},
    ]
    
    summarization_items = [
        {
            "id": "sum_1", 
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "id": "sum_2", 
            "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
        },
        {
            "id": "sum_3", 
            "text": "Natural language processing combines computational linguistics with statistical, machine learning, and deep learning models."
        }
    ]
    
    # Test single batch processing
    print("\n1. Single Batch Processing (Q&A):")
    print("-" * 40)
    
    qa_start = time.time()
    qa_results = await batch_processor.batch_process(qa_items, "qa")
    qa_time = time.time() - qa_start
    
    print(f"\nQ&A Batch Results (took {qa_time:.2f}s):")
    for result in qa_results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['id']}: {result['output'][:80]}...")
        print(f"   Processing time: {result['processing_time']:.2f}s")
    
    # Test mixed batch processing
    print("\n\n2. Mixed Batch Processing:")
    print("-" * 40)
    
    mixed_start = time.time()
    
    # Process summarization concurrently
    sum_task = asyncio.create_task(
        batch_processor.batch_process(summarization_items, "summarize")
    )
    
    sum_results = await sum_task
    mixed_time = time.time() - mixed_start
    
    print(f"\nSummarization Results (took {mixed_time:.2f}s):")
    for result in sum_results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_icon} {result['id']}: {result['output'][:60]}...")

async def test_error_handling():
    """Test error handling and retry mechanisms."""
    
    robust_module = RobustAsyncModule(max_retries=2, retry_delay=0.5)
    
    # Test prompts
    test_prompts = [
        "Explain the concept of machine learning",
        "What is artificial intelligence?",
        "Describe neural network architectures",
        "How does deep learning work?",
        "What are the applications of AI?"
    ]
    
    print(f"\nTesting robust processing with {len(test_prompts)} prompts...")
    print("-" * 50)
    
    start_time = time.time()
    
    # Process all prompts
    results = await robust_module.batch_robust_forward(
        test_prompts, 
        max_concurrent=2
    )
    
    processing_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    print(f"\n📊 Processing Results:")
    print(f"Total prompts: {len(test_prompts)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Processing time: {processing_time:.2f}s")
    
    # Show individual results
    print(f"\n✅ Successful Results:")
    for result in successful_results:
        attempts = result.get("attempts", 1)
        response_preview = result["result"].response[:80] + "..." if hasattr(result["result"], 'response') else "No response"
        print(f"  • {result['prompt']} (attempts: {attempts})")
        print(f"    Response: {response_preview}")
    
    if failed_results:
        print(f"\n❌ Failed Results:")
        for result in failed_results:
            attempts = result.get("attempts", 1)
            print(f"  • {result['prompt']} (attempts: {attempts})")
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    # Show module statistics
    stats = robust_module.get_stats()
    print(f"\n📈 Module Statistics:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Retries performed: {stats['retries_performed']}")
    print(f"Average retries per request: {stats['avg_retries_per_request']:.2f}")

async def benchmark_async_performance():
    """Benchmark async performance across different concurrency levels."""
    
    async_module = AsyncDSPyModule()
    perf_monitor = AsyncPerformanceMonitor()
    
    # Prepare test data
    test_prompts = [
        "Explain machine learning concepts",
        "What is artificial intelligence?",
        "Describe neural networks",
        "How does deep learning work?",
        "What are transformers in AI?",
        "Explain computer vision",
        "What is natural language processing?",
        "Describe reinforcement learning"
    ]
    
    # Test different concurrency levels
    concurrency_levels = [1, 2, 4]
    
    print(f"Benchmarking with {len(test_prompts)} prompts across {len(concurrency_levels)} concurrency levels...")
    
    # Run benchmark
    benchmark_results = await perf_monitor.benchmark_concurrency(
        async_module.async_forward,
        [{"prompt": prompt} for prompt in test_prompts],
        concurrency_levels
    )
    
    # Display results
    print(f"\n📊 Benchmark Results:")
    print("-" * 60)
    print(f"{'Concurrency':<12} {'Time(s)':<10} {'Throughput':<12} {'Success%':<10}")
    print("-" * 60)
    
    for level_name, metrics in benchmark_results.items():
        concurrency = level_name.split('_')[1]
        print(f"{concurrency:<12} {metrics['total_time']:<10.2f} "
              f"{metrics['throughput']:<12.2f} {metrics['success_rate']:<10.1f}")
    
    # Find optimal concurrency
    best_throughput = max(benchmark_results.values(), key=lambda x: x['throughput'])
    best_level = [k for k, v in benchmark_results.items() if v == best_throughput][0]
    
    print(f"\n🏆 Best Performance:")
    print(f"Optimal concurrency: {best_level.split('_')[1]}")
    print(f"Peak throughput: {best_throughput['throughput']:.2f} requests/second")
    print(f"Total time: {best_throughput['total_time']:.2f}s")

def main():
    """Main function demonstrating async processing with DSPy."""
    print("=" * 70)
    print("ASYNC PROCESSING WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for async operations")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=1000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    async def run_async_demos():
        """Run all async demonstrations."""
        try:
            # Demo 1: Basic Async Operations
            print_step("Basic Async Operations", "Comparing sync vs async performance")
            await test_basic_async()
            
            # Demo 2: Batch Processing
            print_step("Batch Processing", "Demonstrating concurrent batch operations")
            await test_batch_processing()
            
            # Demo 3: Error Handling
            print_step("Error Handling", "Testing robust async processing")
            await test_error_handling()
            
            # Demo 4: Performance Benchmarking
            print_step("Performance Benchmarking", "Testing different concurrency levels")
            await benchmark_async_performance()
            
            print("\n" + "=" * 70)
            print("🎉 ASYNC PROCESSING DEMONSTRATION COMPLETED!")
            print("=" * 70)
            print("\nKey takeaways:")
            print("✅ Basic async operations: Converting sync to async DSPy modules")
            print("✅ Batch processing: Concurrent processing of multiple requests")
            print("✅ Error handling: Robust retry logic and timeout management")
            print("✅ Performance monitoring: Metrics and benchmarking tools")
            print("✅ Concurrency control: Semaphores and rate limiting")
            
        except Exception as e:
            print_error(f"Error in async demonstration: {e}")
    
    # Run async demos
    asyncio.run(run_async_demos())

if __name__ == "__main__":
    main()
