#!/usr/bin/env python3
"""
Saving and Loading DSPy Models

This script demonstrates how to save, load, and manage DSPy models and configurations
for deployment and persistence across sessions.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

@dataclass
class ModelMetadata:
    model_name: str
    version: str
    created_at: str
    dspy_version: str
    model_type: str
    performance_metrics: Dict[str, float]
    training_data_hash: str
    configuration: Dict[str, Any]

def main():
    """Main function demonstrating saving and loading DSPy models."""
    print("=" * 70)
    print("SAVING AND LOADING DSPY MODELS")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for model persistence")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=1000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # DSPy Signatures
    class QuestionAnswering(dspy.Signature):
        """Answer questions with detailed explanations."""
        question = dspy.InputField(desc="The question to be answered")
        context = dspy.InputField(desc="Relevant context information", required=False)
        answer = dspy.OutputField(desc="Detailed answer to the question")
        confidence = dspy.OutputField(desc="Confidence score (1-10)")
    
    class TextSummarization(dspy.Signature):
        """Summarize text content concisely."""
        text = dspy.InputField(desc="Text to be summarized")
        length = dspy.InputField(desc="Desired summary length: short, medium, long")
        summary = dspy.OutputField(desc="Concise summary of the text")
        key_points = dspy.OutputField(desc="Main key points from the text")
    
    class SentimentAnalysis(dspy.Signature):
        """Analyze sentiment and emotional tone of text."""
        text = dspy.InputField(desc="Text to analyze")
        sentiment = dspy.OutputField(desc="Sentiment: positive, negative, neutral")
        confidence = dspy.OutputField(desc="Confidence score (0-100)")
        emotions = dspy.OutputField(desc="Detected emotions and intensity")
    
    # Sample Models
    class QuestionAnsweringModel(dspy.Module):
        """Advanced question answering model."""
        
        def __init__(self, use_cot: bool = True):
            super().__init__()
            self.use_cot = use_cot
            if use_cot:
                self.qa = dspy.ChainOfThought(QuestionAnswering)
            else:
                self.qa = dspy.Predict(QuestionAnswering)
            
            # Model configuration
            self.config = {
                "use_chain_of_thought": use_cot,
                "model_type": "question_answering",
                "version": "1.0.0"
            }
        
        def forward(self, question, context=""):
            result = self.qa(question=question, context=context)
            return dspy.Prediction(
                answer=result.answer,
                confidence=result.confidence
            )
        
        def get_config(self):
            return self.config
    
    class TextSummarizationModel(dspy.Module):
        """Text summarization model with configurable parameters."""
        
        def __init__(self, default_length: str = "medium"):
            super().__init__()
            self.default_length = default_length
            self.summarizer = dspy.ChainOfThought(TextSummarization)
            
            self.config = {
                "default_length": default_length,
                "model_type": "text_summarization",
                "version": "1.1.0"
            }
        
        def forward(self, text, length=None):
            if length is None:
                length = self.default_length
            
            result = self.summarizer(text=text, length=length)
            return dspy.Prediction(
                summary=result.summary,
                key_points=result.key_points
            )
        
        def get_config(self):
            return self.config
    
    class SentimentAnalysisModel(dspy.Module):
        """Sentiment analysis model with emotion detection."""
        
        def __init__(self, include_emotions: bool = True):
            super().__init__()
            self.include_emotions = include_emotions
            self.sentiment_analyzer = dspy.ChainOfThought(SentimentAnalysis)
            
            self.config = {
                "include_emotions": include_emotions,
                "model_type": "sentiment_analysis",
                "version": "2.0.0"
            }
        
        def forward(self, text):
            result = self.sentiment_analyzer(text=text)
            prediction = dspy.Prediction(
                sentiment=result.sentiment,
                confidence=result.confidence
            )
            
            if self.include_emotions:
                prediction.emotions = result.emotions
            
            return prediction
        
        def get_config(self):
            return self.config
    
    # Model Management Class
    class ModelManager:
        """Manages saving, loading, and versioning of DSPy models."""
        
        def __init__(self, models_directory: str = "models"):
            self.models_dir = Path(models_directory)
            self.models_dir.mkdir(exist_ok=True)
            self.metadata_file = self.models_dir / "metadata.json"
            self.metadata = self._load_metadata()
        
        def _load_metadata(self) -> Dict[str, Any]:
            """Load existing metadata or create new."""
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {"models": {}, "last_updated": datetime.now().isoformat()}
        
        def _save_metadata(self):
            """Save metadata to file."""
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        
        def save_model(self, model: dspy.Module, model_name: str, 
                      performance_metrics: Dict[str, float] = None,
                      training_data_hash: str = "unknown") -> str:
            """Save a DSPy model with metadata."""
            try:
                # Create model version directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                version = f"v{timestamp}"
                model_dir = self.models_dir / model_name / version
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model using DSPy's built-in save method
                model_path = model_dir / "model.json"
                try:
                    model.save(str(model_path))
                except AttributeError:
                    # Fallback: save as pickle if DSPy save is not available
                    with open(model_dir / "model.pkl", 'wb') as f:
                        pickle.dump(model, f)
                    model_path = model_dir / "model.pkl"
                
                # Save model configuration
                if hasattr(model, 'get_config'):
                    config = model.get_config()
                else:
                    config = {"model_class": model.__class__.__name__}
                
                with open(model_dir / "config.json", 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Create metadata
                metadata = ModelMetadata(
                    model_name=model_name,
                    version=version,
                    created_at=datetime.now().isoformat(),
                    dspy_version=dspy.__version__ if hasattr(dspy, '__version__') else "unknown",
                    model_type=config.get("model_type", "unknown"),
                    performance_metrics=performance_metrics or {},
                    training_data_hash=training_data_hash,
                    configuration=config
                )
                
                # Save metadata
                with open(model_dir / "metadata.json", 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
                
                # Update global metadata
                if model_name not in self.metadata["models"]:
                    self.metadata["models"][model_name] = {"versions": []}
                
                self.metadata["models"][model_name]["versions"].append({
                    "version": version,
                    "path": str(model_dir),
                    "created_at": metadata.created_at,
                    "performance": performance_metrics or {}
                })
                
                self._save_metadata()
                
                return str(model_path)
                
            except Exception as e:
                raise Exception(f"Failed to save model: {str(e)}")
        
        def load_model(self, model_name: str, version: str = "latest") -> dspy.Module:
            """Load a saved DSPy model."""
            try:
                if model_name not in self.metadata["models"]:
                    raise ValueError(f"Model '{model_name}' not found")
                
                versions = self.metadata["models"][model_name]["versions"]
                if not versions:
                    raise ValueError(f"No versions found for model '{model_name}'")
                
                if version == "latest":
                    # Get the latest version
                    model_info = sorted(versions, key=lambda x: x["created_at"])[-1]
                else:
                    # Find specific version
                    model_info = next((v for v in versions if v["version"] == version), None)
                    if not model_info:
                        raise ValueError(f"Version '{version}' not found for model '{model_name}'")
                
                model_dir = Path(model_info["path"])
                
                # Try to load using DSPy's load method first
                model_json_path = model_dir / "model.json"
                model_pkl_path = model_dir / "model.pkl"
                
                if model_json_path.exists():
                    # Load configuration to determine model class
                    with open(model_dir / "config.json", 'r') as f:
                        config = json.load(f)
                    
                    # For this demo, we'll need to recreate the model based on type
                    model_type = config.get("model_type", "unknown")
                    
                    if model_type == "question_answering":
                        model = QuestionAnsweringModel(config.get("use_chain_of_thought", True))
                    elif model_type == "text_summarization":
                        model = TextSummarizationModel(config.get("default_length", "medium"))
                    elif model_type == "sentiment_analysis":
                        model = SentimentAnalysisModel(config.get("include_emotions", True))
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                    
                    # Load the saved state if available
                    try:
                        model.load(str(model_json_path))
                    except:
                        # If loading fails, return the initialized model
                        pass
                
                elif model_pkl_path.exists():
                    # Load from pickle
                    with open(model_pkl_path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    raise FileNotFoundError("No model file found")
                
                return model
                
            except Exception as e:
                raise Exception(f"Failed to load model: {str(e)}")
        
        def list_models(self) -> Dict[str, Any]:
            """List all saved models and their versions."""
            return self.metadata["models"]
        
        def delete_model(self, model_name: str, version: str = None):
            """Delete a model or specific version."""
            if model_name not in self.metadata["models"]:
                raise ValueError(f"Model '{model_name}' not found")
            
            if version is None:
                # Delete entire model
                import shutil
                model_dir = self.models_dir / model_name
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                del self.metadata["models"][model_name]
            else:
                # Delete specific version
                versions = self.metadata["models"][model_name]["versions"]
                version_info = next((v for v in versions if v["version"] == version), None)
                if version_info:
                    import shutil
                    version_dir = Path(version_info["path"])
                    if version_dir.exists():
                        shutil.rmtree(version_dir)
                    versions.remove(version_info)
            
            self._save_metadata()
        
        def get_model_info(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
            """Get detailed information about a model."""
            if model_name not in self.metadata["models"]:
                raise ValueError(f"Model '{model_name}' not found")
            
            versions = self.metadata["models"][model_name]["versions"]
            if version == "latest":
                version_info = sorted(versions, key=lambda x: x["created_at"])[-1]
            else:
                version_info = next((v for v in versions if v["version"] == version), None)
                if not version_info:
                    raise ValueError(f"Version '{version}' not found")
            
            # Load detailed metadata
            metadata_path = Path(version_info["path"]) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    detailed_metadata = json.load(f)
                return detailed_metadata
            
            return version_info
    
    # Initialize model manager
    model_manager = ModelManager()
    print_result("Model manager initialized successfully!")
    
    # Demo 1: Creating and Saving Models
    print_step("Model Creation and Saving", "Creating different types of models")
    
    # Create sample models
    qa_model = QuestionAnsweringModel(use_cot=True)
    summarization_model = TextSummarizationModel(default_length="medium")
    sentiment_model = SentimentAnalysisModel(include_emotions=True)
    
    # Save models with sample performance metrics
    try:
        qa_path = model_manager.save_model(
            qa_model, 
            "question_answering",
            performance_metrics={"accuracy": 0.85, "response_time": 1.2},
            training_data_hash="qa_data_v1_hash123"
        )
        print_result(f"QA Model saved to: {qa_path}", "QA Model")
        
        sum_path = model_manager.save_model(
            summarization_model,
            "text_summarization", 
            performance_metrics={"rouge_score": 0.78, "bleu_score": 0.65},
            training_data_hash="sum_data_v1_hash456"
        )
        print_result(f"Summarization Model saved to: {sum_path}", "Summarization Model")
        
        sent_path = model_manager.save_model(
            sentiment_model,
            "sentiment_analysis",
            performance_metrics={"f1_score": 0.92, "precision": 0.89, "recall": 0.95},
            training_data_hash="sent_data_v1_hash789"
        )
        print_result(f"Sentiment Model saved to: {sent_path}", "Sentiment Model")
        
    except Exception as e:
        print_error(f"Error saving models: {e}")
    
    # Demo 2: Listing Available Models
    print_step("Model Inventory", "Listing all saved models and versions")
    
    try:
        models = model_manager.list_models()
        
        for model_name, model_info in models.items():
            print(f"\n--- Model: {model_name} ---")
            print_result(f"Total Versions: {len(model_info['versions'])}", "Version Count")
            
            for version_info in model_info["versions"]:
                print(f"  Version: {version_info['version']}")
                print(f"  Created: {version_info['created_at']}")
                print(f"  Performance: {version_info['performance']}")
        
    except Exception as e:
        print_error(f"Error listing models: {e}")
    
    # Demo 3: Loading and Testing Models
    print_step("Model Loading and Testing", "Loading saved models and testing functionality")
    
    test_cases = {
        "question_answering": [
            {"question": "What is machine learning?", "context": "AI and data science context"},
            {"question": "How does photosynthesis work?", "context": "Biology and chemistry"}
        ],
        "text_summarization": [
            {
                "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.",
                "length": "short"
            }
        ],
        "sentiment_analysis": [
            {"text": "I absolutely love this new product! It's amazing and works perfectly."},
            {"text": "This is the worst experience I've ever had. Completely disappointed."},
            {"text": "It's okay, nothing special but does what it's supposed to do."}
        ]
    }
    
    for model_name, tests in test_cases.items():
        try:
            print(f"\n--- Testing {model_name} ---")
            
            # Load the model
            loaded_model = model_manager.load_model(model_name)
            print_result(f"Successfully loaded {model_name}", "Model Loading")
            
            # Test the model
            for i, test_case in enumerate(tests, 1):
                try:
                    if model_name == "question_answering":
                        result = loaded_model(test_case["question"], test_case["context"])
                        print(f"\nTest {i}: {test_case['question']}")
                        print_result(f"Answer: {result.answer}", "Model Response")
                        print_result(f"Confidence: {result.confidence}", "Confidence")
                    
                    elif model_name == "text_summarization":
                        result = loaded_model(test_case["text"], test_case["length"])
                        print(f"\nTest {i}: Summarization")
                        print_result(f"Summary: {result.summary}", "Summary")
                        print_result(f"Key Points: {result.key_points}", "Key Points")
                    
                    elif model_name == "sentiment_analysis":
                        result = loaded_model(test_case["text"])
                        print(f"\nTest {i}: {test_case['text']}")
                        print_result(f"Sentiment: {result.sentiment}", "Sentiment")
                        print_result(f"Confidence: {result.confidence}", "Confidence")
                        if hasattr(result, 'emotions'):
                            print_result(f"Emotions: {result.emotions}", "Emotions")
                
                except Exception as e:
                    print_error(f"Error testing {model_name} case {i}: {e}")
        
        except Exception as e:
            print_error(f"Error loading {model_name}: {e}")
    
    # Demo 4: Model Information and Metadata
    print_step("Model Metadata", "Retrieving detailed model information")
    
    for model_name in ["question_answering", "text_summarization", "sentiment_analysis"]:
        try:
            model_info = model_manager.get_model_info(model_name)
            
            print(f"\n--- {model_name} Information ---")
            print_result(f"Version: {model_info['version']}", "Current Version")
            print_result(f"Created: {model_info['created_at']}", "Creation Date")
            print_result(f"Type: {model_info['model_type']}", "Model Type")
            print_result(f"Performance: {model_info['performance_metrics']}", "Performance Metrics")
            print_result(f"Configuration: {model_info['configuration']}", "Configuration")
            
        except Exception as e:
            print_error(f"Error retrieving info for {model_name}: {e}")
    
    # Demo 5: Model Versioning
    print_step("Model Versioning", "Demonstrating model version management")
    
    try:
        # Create an improved version of the QA model
        improved_qa_model = QuestionAnsweringModel(use_cot=True)
        
        # Save as new version with better performance
        improved_path = model_manager.save_model(
            improved_qa_model,
            "question_answering",
            performance_metrics={"accuracy": 0.92, "response_time": 1.0},
            training_data_hash="qa_data_v2_hash124"
        )
        
        print_result(f"Improved QA model saved: {improved_path}", "New Version")
        
        # List all versions
        models = model_manager.list_models()
        qa_versions = models["question_answering"]["versions"]
        
        print(f"\nQA Model Versions ({len(qa_versions)} total):")
        for version in qa_versions:
            print(f"  {version['version']}: Accuracy {version['performance'].get('accuracy', 'N/A')}")
        
    except Exception as e:
        print_error(f"Error in versioning demo: {e}")
    
    # Demo 6: Model Comparison
    print_step("Model Comparison", "Comparing different model versions")
    
    try:
        # Compare performance across versions
        models = model_manager.list_models()
        
        print("\n--- Performance Comparison ---")
        for model_name, model_info in models.items():
            print(f"\n{model_name}:")
            
            versions = sorted(model_info["versions"], key=lambda x: x["created_at"])
            for version in versions:
                perf = version["performance"]
                if perf:
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in perf.items()])
                    print(f"  {version['version']}: {metrics_str}")
                else:
                    print(f"  {version['version']}: No performance data")
    
    except Exception as e:
        print_error(f"Error in comparison demo: {e}")
    
    # Demo 7: Export and Import
    print_step("Model Export/Import", "Exporting models for deployment")
    
    try:
        # Create export package
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        # Export model with all metadata
        export_package = {
            "model_name": "question_answering",
            "version": "latest",
            "export_date": datetime.now().isoformat(),
            "metadata": model_manager.get_model_info("question_answering"),
            "requirements": ["dspy-ai", "openai"]
        }
        
        # Save export package
        with open(export_dir / "qa_model_export.json", 'w') as f:
            json.dump(export_package, f, indent=2)
        
        print_result("Model exported successfully", "Export Status")
        print_result(f"Export package: {export_dir / 'qa_model_export.json'}", "Export Location")
        
    except Exception as e:
        print_error(f"Error in export demo: {e}")
    
    print("\n" + "="*70)
    print("SAVING AND LOADING COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy model persistence and management!")

if __name__ == "__main__":
    main()
