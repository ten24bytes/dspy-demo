#!/usr/bin/env python3
"""
Caching and Performance Optimization with DSPy

This script demonstrates various caching strategies and performance optimization
techniques for DSPy applications.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import time
import json
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path
from utils import setup_default_lm, print_step, print_result, print_error

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    hit_count: int = 0
    ttl: Optional[timedelta] = None

class SimpleCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and datetime.now() - entry.timestamp > entry.ttl:
                self.remove(key)
                return None
            
            # Update access pattern
            entry.hit_count += 1
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.value
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None):
        """Put value in cache."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Add or update entry
        self.cache[key] = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl=ttl
        )
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def remove(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(entry.hit_count for entry in self.cache.values())
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'total_hits': total_hits,
            'keys': list(self.cache.keys())
        }

class FileCache:
    """File-based persistent cache."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print_error(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, default=str, indent=2)
        except Exception as e:
            print_error(f"Error saving cache metadata: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        if key in self.metadata:
            entry_info = self.metadata[key]
            file_path = self._get_file_path(key)
            
            # Check if file exists
            if not file_path.exists():
                del self.metadata[key]
                self._save_metadata()
                return None
            
            # Check TTL
            if entry_info.get('ttl'):
                created_time = datetime.fromisoformat(entry_info['timestamp'])
                ttl = timedelta(seconds=entry_info['ttl'])
                if datetime.now() - created_time > ttl:
                    self.remove(key)
                    return None
            
            # Load and return value
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update hit count
                entry_info['hit_count'] = entry_info.get('hit_count', 0) + 1
                self._save_metadata()
                
                return value
            except Exception as e:
                print_error(f"Error loading cached value: {e}")
                self.remove(key)
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Put value in file cache."""
        file_path = self._get_file_path(key)
        
        try:
            # Save value to file
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            self.metadata[key] = {
                'timestamp': datetime.now().isoformat(),
                'ttl': ttl,
                'hit_count': 0,
                'file_path': str(file_path)
            }
            self._save_metadata()
            
        except Exception as e:
            print_error(f"Error caching value: {e}")
    
    def remove(self, key: str):
        """Remove entry from file cache."""
        if key in self.metadata:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
            del self.metadata[key]
            self._save_metadata()
    
    def clear(self):
        """Clear all cache entries."""
        for key in list(self.metadata.keys()):
            self.remove(key)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(entry.get('hit_count', 0) for entry in self.metadata.values())
        total_size = sum(
            Path(entry['file_path']).stat().st_size 
            for entry in self.metadata.values() 
            if Path(entry['file_path']).exists()
        )
        
        return {
            'entries': len(self.metadata),
            'total_hits': total_hits,
            'total_size_bytes': total_size,
            'cache_dir': str(self.cache_dir)
        }

def cache_decorator(cache_instance, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if isinstance(cache_instance, FileCache):
                cache_instance.put(cache_key, result, ttl)
            else:
                cache_ttl = timedelta(seconds=ttl) if ttl else None
                cache_instance.put(cache_key, result, cache_ttl)
            
            return result
        
        return wrapper
    return decorator

class CachedModule(dspy.Module):
    """DSPy module with built-in caching."""
    
    def __init__(self, signature, cache_instance=None):
        super().__init__()
        self.signature = signature
        self.predictor = dspy.ChainOfThought(signature)
        self.cache = cache_instance or SimpleCache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def forward(self, **kwargs):
        """Forward with caching."""
        # Create cache key from inputs
        cache_key = f"{self.signature.__name__}:{hash(str(sorted(kwargs.items())))}"
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.cache_hits += 1
            return cached_result
        
        # Execute and cache
        self.cache_misses += 1
        result = self.predictor(**kwargs)
        
        # Cache the result
        if isinstance(self.cache, FileCache):
            self.cache.put(cache_key, result, ttl=3600)  # 1 hour TTL
        else:
            self.cache.put(cache_key, result, timedelta(hours=1))
        
        return result
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this module."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class BatchedModule(dspy.Module):
    """DSPy module with request batching for efficiency."""
    
    def __init__(self, signature, batch_size: int = 5, max_wait: float = 0.5):
        super().__init__()
        self.signature = signature
        self.predictor = dspy.ChainOfThought(signature)
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.pending_requests = []
        self.last_batch_time = time.time()
    
    def forward(self, **kwargs):
        """Forward with batching."""
        # Add request to batch
        request = {'kwargs': kwargs, 'timestamp': time.time()}
        self.pending_requests.append(request)
        
        # Check if we should process batch
        should_process = (
            len(self.pending_requests) >= self.batch_size or
            time.time() - self.last_batch_time > self.max_wait
        )
        
        if should_process:
            return self._process_batch()
        else:
            # For demo purposes, process immediately
            # In practice, you'd handle this asynchronously
            return self._process_single(kwargs)
    
    def _process_batch(self):
        """Process all pending requests as a batch."""
        if not self.pending_requests:
            return None
        
        results = []
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.last_batch_time = time.time()
        
        for request in batch:
            result = self.predictor(**request['kwargs'])
            results.append(result)
        
        # Return the last result for demo
        return results[-1] if results else None
    
    def _process_single(self, kwargs):
        """Process a single request."""
        return self.predictor(**kwargs)

# Example signatures for testing
class SummarySignature(dspy.Signature):
    """Summarize the given text."""
    text = dspy.InputField(desc="Text to summarize")
    summary = dspy.OutputField(desc="Brief summary of the text")

class TranslationSignature(dspy.Signature):
    """Translate text to specified language."""
    text = dspy.InputField(desc="Text to translate")
    target_language = dspy.InputField(desc="Target language")
    translation = dspy.OutputField(desc="Translated text")

def demonstrate_simple_cache():
    """Demonstrate simple in-memory caching."""
    
    print_step("Simple In-Memory Cache")
    
    cache = SimpleCache(max_size=3)
    
    # Simulate cache operations
    cache.put("key1", "value1")
    cache.put("key2", "value2", ttl=timedelta(seconds=2))
    cache.put("key3", "value3")
    
    print_result(f"Cache stats after initial puts: {cache.stats()}")
    
    # Test retrieval
    print_result(f"Get key1: {cache.get('key1')}")
    print_result(f"Get key2: {cache.get('key2')}")
    print_result(f"Get non-existent: {cache.get('nonexistent')}")
    
    # Test LRU eviction
    cache.put("key4", "value4")  # Should evict key3 (least recently used)
    print_result(f"After adding key4: {list(cache.cache.keys())}")
    
    # Test TTL expiration
    time.sleep(2.1)
    print_result(f"Get key2 after TTL: {cache.get('key2')}")

def demonstrate_file_cache():
    """Demonstrate file-based persistent caching."""
    
    print_step("File-Based Persistent Cache")
    
    cache = FileCache("demo_cache")
    
    # Clear previous cache
    cache.clear()
    
    # Add some entries
    cache.put("persistent_key1", {"data": "important information"})
    cache.put("persistent_key2", [1, 2, 3, 4, 5], ttl=10)
    cache.put("temp_key", "temporary data", ttl=1)
    
    print_result(f"Initial cache stats: {cache.stats()}")
    
    # Test retrieval
    print_result(f"Get persistent_key1: {cache.get('persistent_key1')}")
    print_result(f"Get persistent_key2: {cache.get('persistent_key2')}")
    
    # Test TTL
    time.sleep(1.1)
    print_result(f"Get temp_key after TTL: {cache.get('temp_key')}")
    
    # Test persistence by creating new cache instance
    new_cache = FileCache("demo_cache")
    print_result(f"New cache instance stats: {new_cache.stats()}")
    print_result(f"Get from new instance: {new_cache.get('persistent_key1')}")

@cache_decorator(SimpleCache(), ttl=60)
def expensive_computation(n: int) -> int:
    """Simulate an expensive computation."""
    print_result(f"Computing expensive_computation({n})...")
    time.sleep(0.1)  # Simulate work
    return n * n * n

def demonstrate_cache_decorator():
    """Demonstrate function caching with decorator."""
    
    print_step("Function Caching with Decorator")
    
    # First call - should compute
    start_time = time.time()
    result1 = expensive_computation(5)
    time1 = time.time() - start_time
    print_result(f"First call result: {result1}, time: {time1:.3f}s")
    
    # Second call - should use cache
    start_time = time.time()
    result2 = expensive_computation(5)
    time2 = time.time() - start_time
    print_result(f"Second call result: {result2}, time: {time2:.3f}s")
    
    # Different parameter - should compute
    start_time = time.time()
    result3 = expensive_computation(3)
    time3 = time.time() - start_time
    print_result(f"Different param result: {result3}, time: {time3:.3f}s")

def demonstrate_cached_module():
    """Demonstrate DSPy module with caching."""
    
    print_step("Cached DSPy Module")
    
    # Create cached module
    cached_summarizer = CachedModule(SummarySignature, SimpleCache(max_size=10))
    
    # Test text
    test_text = ("Artificial Intelligence has revolutionized many industries. "
                "Machine learning algorithms can now process vast amounts of data "
                "and make predictions with remarkable accuracy.")
    
    # First call
    print_result("First summarization call...")
    start_time = time.time()
    result1 = cached_summarizer(text=test_text)
    time1 = time.time() - start_time
    print_result(f"Result: {result1.summary}")
    print_result(f"Time: {time1:.3f}s")
    
    # Second call (should be cached)
    print_result("Second summarization call (cached)...")
    start_time = time.time()
    result2 = cached_summarizer(text=test_text)
    time2 = time.time() - start_time
    print_result(f"Result: {result2.summary}")
    print_result(f"Time: {time2:.3f}s")
    
    # Show cache stats
    print_result(f"Cache stats: {cached_summarizer.cache_stats()}")

def demonstrate_batched_module():
    """Demonstrate batched processing for efficiency."""
    
    print_step("Batched DSPy Module")
    
    # Create batched module
    batched_translator = BatchedModule(TranslationSignature, batch_size=3, max_wait=1.0)
    
    # Test multiple translations
    texts = [
        "Hello, how are you?",
        "The weather is beautiful today.",
        "Technology is advancing rapidly."
    ]
    
    print_result("Processing multiple translation requests...")
    
    for i, text in enumerate(texts):
        start_time = time.time()
        result = batched_translator(text=text, target_language="Spanish")
        elapsed = time.time() - start_time
        
        print_result(f"Translation {i+1}: {result.translation}")
        print_result(f"Processing time: {elapsed:.3f}s")

def demonstrate_performance_comparison():
    """Compare performance with and without caching."""
    
    print_step("Performance Comparison")
    
    # Create cached and non-cached modules
    cached_module = CachedModule(SummarySignature, SimpleCache())
    non_cached_module = dspy.ChainOfThought(SummarySignature)
    
    test_texts = [
        "AI is transforming healthcare through diagnostic tools.",
        "Climate change requires immediate global action.",
        "AI is transforming healthcare through diagnostic tools.",  # Duplicate
        "Remote work has changed how we collaborate.",
        "Climate change requires immediate global action."  # Duplicate
    ]
    
    # Test non-cached performance
    print_result("Testing non-cached module...")
    start_time = time.time()
    non_cached_results = []
    for text in test_texts:
        result = non_cached_module(text=text)
        non_cached_results.append(result)
    non_cached_time = time.time() - start_time
    
    # Test cached performance
    print_result("Testing cached module...")
    start_time = time.time()
    cached_results = []
    for text in test_texts:
        result = cached_module(text=text)
        cached_results.append(result)
    cached_time = time.time() - start_time
    
    # Show results
    print_result(f"Non-cached total time: {non_cached_time:.3f}s")
    print_result(f"Cached total time: {cached_time:.3f}s")
    print_result(f"Time saved: {non_cached_time - cached_time:.3f}s")
    print_result(f"Cache stats: {cached_module.cache_stats()}")

def main():
    """Main function demonstrating caching and optimization."""
    
    print("=" * 60)
    print("DSPy Caching and Performance Demo")
    print("=" * 60)
    
    # Setup language model
    lm = setup_default_lm()
    if not lm:
        return
    
    try:
        # Simple cache demonstration
        demonstrate_simple_cache()
        
        # File cache demonstration
        demonstrate_file_cache()
        
        # Function caching
        demonstrate_cache_decorator()
        
        # Cached DSPy module
        demonstrate_cached_module()
        
        # Batched processing
        demonstrate_batched_module()
        
        # Performance comparison
        demonstrate_performance_comparison()
        
        print_step("Caching and Performance Demo Complete!")
        
    except Exception as e:
        print_error(f"Error in caching demo: {e}")

if __name__ == "__main__":
    main()
