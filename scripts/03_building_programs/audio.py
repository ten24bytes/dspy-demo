#!/usr/bin/env python3
"""
Audio Processing with DSPy

This script demonstrates how to use DSPy for audio-related tasks including
speech recognition, audio analysis, and voice-based interactions.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
from typing import List, Dict, Any, Optional
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv
import tempfile
import io

def main():
    """Main function demonstrating audio processing with DSPy."""
    print("=" * 70)
    print("AUDIO PROCESSING WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for audio processing")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=1500)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Audio Analysis Signatures
    class AudioTranscription(dspy.Signature):
        """Enhance and analyze audio transcription."""
        raw_transcription = dspy.InputField(desc="Raw speech-to-text transcription")
        audio_quality = dspy.InputField(desc="Audio quality metrics (clarity, noise level)")
        enhanced_transcription = dspy.OutputField(desc="Cleaned and enhanced transcription")
        confidence_score = dspy.OutputField(desc="Confidence score for the transcription (0-100)")
        speaker_analysis = dspy.OutputField(desc="Analysis of speaker characteristics")
    
    class AudioContentAnalysis(dspy.Signature):
        """Analyze the content and context of audio transcription."""
        transcription = dspy.InputField(desc="Audio transcription text")
        audio_metadata = dspy.InputField(desc="Audio file metadata (duration, format, etc.)")
        content_summary = dspy.OutputField(desc="Summary of the audio content")
        sentiment = dspy.OutputField(desc="Emotional tone and sentiment analysis")
        key_topics = dspy.OutputField(desc="Main topics and themes discussed")
        action_items = dspy.OutputField(desc="Any action items or important points mentioned")
    
    class VoiceCommandProcessor(dspy.Signature):
        """Process and interpret voice commands."""
        voice_command = dspy.InputField(desc="Voice command transcription")
        context = dspy.InputField(desc="Current application or system context")
        intent = dspy.OutputField(desc="Identified user intent")
        parameters = dspy.OutputField(desc="Extracted parameters and entities")
        response = dspy.OutputField(desc="Appropriate response or action to take")
    
    # Audio Processing Classes
    class AudioProcessor:
        """Handles audio file processing and analysis."""
        
        def __init__(self):
            self.recognizer = sr.Recognizer()
        
        def load_audio(self, file_path: str) -> Dict[str, Any]:
            """Load and analyze audio file."""
            try:
                # Load audio file
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                
                # Calculate basic metrics
                duration = len(audio_data) / sample_rate
                rms_energy = np.sqrt(np.mean(audio_data**2))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
                
                return {
                    'audio_data': audio_data,
                    'sample_rate': sample_rate,
                    'duration': duration,
                    'rms_energy': float(rms_energy),
                    'spectral_centroid': float(spectral_centroid),
                    'quality_score': min(100, rms_energy * 1000)  # Simple quality metric
                }
            except Exception as e:
                print_error(f"Error loading audio file: {e}")
                return {}
        
        def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
            """Transcribe audio using speech recognition."""
            try:
                # Convert to format compatible with speech_recognition
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_data, sample_rate)
                    
                    with sr.AudioFile(temp_file.name) as source:
                        audio = self.recognizer.record(source)
                        transcription = self.recognizer.recognize_google(audio)
                    
                    os.unlink(temp_file.name)
                    return transcription
                    
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Speech recognition error: {e}"
            except Exception as e:
                return f"Transcription error: {e}"
        
        def generate_sample_audio_data(self) -> Dict[str, Any]:
            """Generate sample audio data for demonstration."""
            # Create a simple sine wave as sample audio
            duration = 3.0  # seconds
            sample_rate = 22050
            frequency = 440  # A note
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.05, audio_data.shape)
            audio_data += noise
            
            return {
                'audio_data': audio_data,
                'sample_rate': sample_rate,
                'duration': duration,
                'rms_energy': float(np.sqrt(np.mean(audio_data**2))),
                'spectral_centroid': frequency,
                'quality_score': 75  # Simulated quality score
            }
    
    # DSPy Audio Module
    class AudioAnalyzer(dspy.Module):
        """Comprehensive audio analysis module using DSPy."""
        
        def __init__(self):
            super().__init__()
            self.audio_processor = AudioProcessor()
            self.transcription_enhancer = dspy.ChainOfThought(AudioTranscription)
            self.content_analyzer = dspy.ChainOfThought(AudioContentAnalysis)
            self.command_processor = dspy.ChainOfThought(VoiceCommandProcessor)
        
        def analyze_audio_file(self, file_path: str = None) -> dspy.Prediction:
            """Analyze an audio file or use sample data."""
            if file_path and os.path.exists(file_path):
                audio_info = self.audio_processor.load_audio(file_path)
            else:
                print("Using sample audio data for demonstration...")
                audio_info = self.audio_processor.generate_sample_audio_data()
            
            # Simulate transcription for demo (in real scenario, would use actual transcription)
            if file_path:
                transcription = self.audio_processor.transcribe_audio(
                    audio_info['audio_data'], 
                    audio_info['sample_rate']
                )
            else:
                # Sample transcription for demo
                transcription = "Hello, this is a sample audio recording for testing purposes. The audio quality seems good and the speaker is clear."
            
            # Format audio quality info
            quality_info = f"""
            Duration: {audio_info['duration']:.2f} seconds
            Sample Rate: {audio_info['sample_rate']} Hz
            RMS Energy: {audio_info['rms_energy']:.4f}
            Quality Score: {audio_info['quality_score']:.1f}/100
            """
            
            # Enhance transcription
            enhanced = self.transcription_enhancer(
                raw_transcription=transcription,
                audio_quality=quality_info
            )
            
            return enhanced
        
        def analyze_content(self, transcription: str, metadata: str) -> dspy.Prediction:
            """Analyze the content of audio transcription."""
            return self.content_analyzer(
                transcription=transcription,
                audio_metadata=metadata
            )
        
        def process_voice_command(self, command: str, context: str = "General assistant") -> dspy.Prediction:
            """Process a voice command."""
            return self.command_processor(
                voice_command=command,
                context=context
            )
    
    # Initialize the audio analyzer
    analyzer = AudioAnalyzer()
    print_result("Audio analyzer initialized successfully!")
    
    # Demo 1: Audio Transcription Enhancement
    print_step("Audio Transcription Demo", "Enhancing and analyzing audio transcription")
    
    try:
        audio_analysis = analyzer.analyze_audio_file()
        
        print_result(f"Enhanced Transcription: {audio_analysis.enhanced_transcription}", "Enhanced Transcription")
        print_result(f"Confidence Score: {audio_analysis.confidence_score}", "Confidence")
        print_result(f"Speaker Analysis: {audio_analysis.speaker_analysis}", "Speaker Analysis")
        
    except Exception as e:
        print_error(f"Error in audio transcription demo: {e}")
    
    # Demo 2: Content Analysis
    print_step("Content Analysis Demo", "Analyzing audio content and extracting insights")
    
    sample_transcriptions = [
        {
            "transcription": "In our quarterly meeting today, we discussed the budget allocation for Q4. The marketing team needs an additional 50,000 dollars for the product launch campaign. We also need to hire two new developers by next month.",
            "type": "Business Meeting"
        },
        {
            "transcription": "Welcome to today's lecture on machine learning. We'll be covering neural networks, specifically how backpropagation works and why it's so important for training deep learning models.",
            "type": "Educational Content"
        },
        {
            "transcription": "I just wanted to call and let you know that the package arrived safely. Thank you so much for sending it so quickly. The kids absolutely love their new toys!",
            "type": "Personal Call"
        }
    ]
    
    for i, sample in enumerate(sample_transcriptions, 1):
        try:
            metadata = f"Audio Type: {sample['type']}, Duration: 2-3 minutes, Quality: High"
            content_analysis = analyzer.analyze_content(sample["transcription"], metadata)
            
            print(f"\n--- Sample {i}: {sample['type']} ---")
            print_result(f"Summary: {content_analysis.content_summary}", "Content Summary")
            print_result(f"Sentiment: {content_analysis.sentiment}", "Sentiment Analysis")
            print_result(f"Key Topics: {content_analysis.key_topics}", "Key Topics")
            print_result(f"Action Items: {content_analysis.action_items}", "Action Items")
            
        except Exception as e:
            print_error(f"Error analyzing sample {i}: {e}")
    
    # Demo 3: Voice Command Processing
    print_step("Voice Command Processing", "Processing and interpreting voice commands")
    
    voice_commands = [
        {
            "command": "Set a reminder for tomorrow at 9 AM to call the dentist",
            "context": "Personal Assistant App"
        },
        {
            "command": "Play some relaxing music in the living room",
            "context": "Smart Home System"
        },
        {
            "command": "What's the weather like today in New York?",
            "context": "Voice Assistant"
        },
        {
            "command": "Open the quarterly sales report and highlight the revenue section",
            "context": "Business Application"
        }
    ]
    
    for i, cmd in enumerate(voice_commands, 1):
        try:
            command_result = analyzer.process_voice_command(cmd["command"], cmd["context"])
            
            print(f"\n--- Command {i} ---")
            print_result(f"Command: {cmd['command']}", "Voice Command")
            print_result(f"Intent: {command_result.intent}", "Detected Intent")
            print_result(f"Parameters: {command_result.parameters}", "Extracted Parameters")
            print_result(f"Response: {command_result.response}", "System Response")
            
        except Exception as e:
            print_error(f"Error processing command {i}: {e}")
    
    # Demo 4: Audio Quality Assessment
    print_step("Audio Quality Assessment", "Analyzing different audio quality scenarios")
    
    quality_scenarios = [
        {
            "description": "High-quality studio recording",
            "transcription": "This is a crystal clear recording made in a professional studio environment.",
            "quality_metrics": "Duration: 10 seconds, Sample Rate: 48000 Hz, RMS Energy: 0.3, Quality Score: 95/100, Background Noise: Minimal"
        },
        {
            "description": "Phone call quality",
            "transcription": "Hello, can you hear me? The connection seems a bit choppy today.",
            "quality_metrics": "Duration: 8 seconds, Sample Rate: 8000 Hz, RMS Energy: 0.15, Quality Score: 60/100, Background Noise: Moderate"
        },
        {
            "description": "Noisy environment",
            "transcription": "I'm calling from the airport so there might be some background noise.",
            "quality_metrics": "Duration: 12 seconds, Sample Rate: 22050 Hz, RMS Energy: 0.4, Quality Score: 40/100, Background Noise: High"
        }
    ]
    
    for i, scenario in enumerate(quality_scenarios, 1):
        try:
            enhanced = analyzer.transcription_enhancer(
                raw_transcription=scenario["transcription"],
                audio_quality=scenario["quality_metrics"]
            )
            
            print(f"\n--- Quality Scenario {i}: {scenario['description']} ---")
            print_result(f"Original: {scenario['transcription']}", "Original Transcription")
            print_result(f"Enhanced: {enhanced.enhanced_transcription}", "Enhanced Version")
            print_result(f"Confidence: {enhanced.confidence_score}", "Confidence Score")
            
        except Exception as e:
            print_error(f"Error in quality scenario {i}: {e}")
    
    print("\n" + "="*70)
    print("AUDIO PROCESSING COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy-powered audio processing capabilities!")

if __name__ == "__main__":
    main()
