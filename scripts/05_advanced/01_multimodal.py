#!/usr/bin/env python3
"""
DSPy Multimodal Support - Working with Images and Audio

This script demonstrates how to use DSPy 3.x's multimodal capabilities:
- dspy.Image for image inputs and outputs
- dspy.Audio for audio inputs and outputs
- Building multimodal AI applications

What You'll Learn:
- How to use dspy.Image in signatures and modules
- How to use dspy.Audio in signatures and modules
- How to build vision-language applications
- How to build speech/audio processing applications
- How to combine multimodal inputs with text
"""

from dotenv import load_dotenv
from utils import setup_default_lm, print_step, print_result, print_error, configure_dspy
import dspy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    """Main function demonstrating DSPy multimodal capabilities."""
    print("=== DSPy Multimodal Support Tutorial ===")
    print("Working with Images and Audio in DSPy 3.x")
    print("=" * 80)

    # Load environment variables
    load_dotenv('.env')

    # Setup Language Model
    # Note: For multimodal support, you need a vision/audio-capable model:
    # - OpenAI: gpt-4o (supports vision)
    # - Anthropic: claude-3-7-sonnet-20250219 (supports vision)
    # - Google: gemini-1.5-pro (supports vision and audio)
    print_step("Setting up Language Model", "Configuring DSPy with a multimodal-capable model")

    try:
        # Using GPT-4o which has excellent vision capabilities
        lm = setup_default_lm(provider="openai", model="gpt-4o", max_tokens=1000)
        configure_dspy(lm=lm)
        print_result("Language model configured successfully!", "Status")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
        return

    # ========== PART 1: Image Understanding ==========
    print_step(
        "Part 1: Image Understanding",
        "Using dspy.Image to analyze and understand images"
    )

    # Example 1: Image Description
    print("\n  Example 1.1: Image Description")

    class ImageDescriber(dspy.Signature):
        """Describe an image in detail."""
        image = dspy.InputField(desc="An image to describe")
        description = dspy.OutputField(desc="A detailed description of the image")

    describer = dspy.Predict(ImageDescriber)

    # Note: In practice, you would load an actual image here
    # For demonstration, we'll show the structure
    print("  Structure:")
    print("    - Create a dspy.Image object from a file path or URL")
    print("    - Pass it to the signature's image field")
    print("    - The model will analyze and describe the image")
    print("\n  Example code:")
    print("    # Load image")
    print("    img = dspy.Image(path='photo.jpg')  # or url='https://...'")
    print("    # Describe it")
    print("    result = describer(image=img)")
    print("    print(result.description)")

    # Example 2: Visual Question Answering
    print("\n  Example 1.2: Visual Question Answering (VQA)")

    class VisualQA(dspy.Signature):
        """Answer questions about an image."""
        image = dspy.InputField(desc="An image to analyze")
        question = dspy.InputField(desc="A question about the image")
        answer = dspy.OutputField(desc="The answer to the question based on the image")

    vqa = dspy.Predict(VisualQA)

    print("  Use case: Answer specific questions about images")
    print("  Example:")
    print("    img = dspy.Image(path='street_scene.jpg')")
    print("    result = vqa(image=img, question='How many cars are visible?')")
    print("    print(f'Answer: {result.answer}')")

    # Example 3: Image Comparison
    print("\n  Example 1.3: Comparing Multiple Images")

    class ImageComparison(dspy.Signature):
        """Compare two images and identify differences."""
        image1 = dspy.InputField(desc="First image")
        image2 = dspy.InputField(desc="Second image")
        differences = dspy.OutputField(desc="Key differences between the images")
        similarity_score = dspy.OutputField(desc="Similarity score from 0-10")

    comparator = dspy.Predict(ImageComparison)

    print("  Use case: Find differences or similarities between images")
    print("  Example:")
    print("    img1 = dspy.Image(path='before.jpg')")
    print("    img2 = dspy.Image(path='after.jpg')")
    print("    result = comparator(image1=img1, image2=img2)")
    print("    print(f'Differences: {result.differences}')")
    print("    print(f'Similarity: {result.similarity_score}/10')")

    # ========== PART 2: Image Generation ==========
    print_step(
        "Part 2: Image Generation",
        "Using dspy.Image as an output field for image generation"
    )

    class ImageGenerator(dspy.Signature):
        """Generate an image based on a text description."""
        description = dspy.InputField(desc="Description of the image to generate")
        image = dspy.OutputField(desc="The generated image")

    generator = dspy.Predict(ImageGenerator)

    print("  Use case: Generate images from text descriptions")
    print("  Note: This requires a model with image generation capabilities")
    print("  Example:")
    print("    result = generator(description='A serene mountain lake at sunset')")
    print("    result.image.save('generated_image.jpg')")

    # ========== PART 3: Audio Processing ==========
    print_step(
        "Part 3: Audio Processing",
        "Using dspy.Audio for speech and audio understanding"
    )

    # Example 1: Audio Transcription
    print("\n  Example 3.1: Audio Transcription")

    class AudioTranscriber(dspy.Signature):
        """Transcribe audio to text."""
        audio = dspy.InputField(desc="Audio file to transcribe")
        transcript = dspy.OutputField(desc="The transcribed text")

    transcriber = dspy.Predict(AudioTranscriber)

    print("  Use case: Convert speech to text")
    print("  Example:")
    print("    audio = dspy.Audio(path='speech.mp3')")
    print("    result = transcriber(audio=audio)")
    print("    print(f'Transcript: {result.transcript}')")

    # Example 2: Audio Analysis
    print("\n  Example 3.2: Audio Content Analysis")

    class AudioAnalyzer(dspy.Signature):
        """Analyze audio content."""
        audio = dspy.InputField(desc="Audio file to analyze")
        summary = dspy.OutputField(desc="Summary of audio content")
        emotions = dspy.OutputField(desc="Detected emotions or tone")
        key_points = dspy.OutputField(desc="Key points or topics discussed")

    analyzer = dspy.Predict(AudioAnalyzer)

    print("  Use case: Analyze audio content for insights")
    print("  Example:")
    print("    audio = dspy.Audio(path='meeting.mp3')")
    print("    result = analyzer(audio=audio)")
    print("    print(f'Summary: {result.summary}')")
    print("    print(f'Emotions: {result.emotions}')")
    print("    print(f'Key points: {result.key_points}')")

    # ========== PART 4: Multimodal Custom Modules ==========
    print_step(
        "Part 4: Custom Multimodal Modules",
        "Building advanced multimodal applications"
    )

    class MultimodalAssistant(dspy.Module):
        """
        A comprehensive assistant that can handle text, images, and audio.
        """
        def __init__(self):
            super().__init__()

            # Image understanding
            class AnalyzeImage(dspy.Signature):
                """Analyze an image and extract information."""
                image = dspy.InputField()
                context = dspy.InputField(desc="Additional context or question")
                analysis = dspy.OutputField(desc="Detailed analysis")

            # Audio processing
            class ProcessAudio(dspy.Signature):
                """Process audio and extract information."""
                audio = dspy.InputField()
                context = dspy.InputField(desc="Additional context")
                information = dspy.OutputField(desc="Extracted information")

            # Combine multimodal inputs
            class CombineMultimodal(dspy.Signature):
                """Combine information from multiple modalities."""
                text_input = dspy.InputField(desc="Text information")
                image_info = dspy.InputField(desc="Information from images")
                audio_info = dspy.InputField(desc="Information from audio")
                response = dspy.OutputField(desc="Comprehensive response")

            self.analyze_image = dspy.Predict(AnalyzeImage)
            self.process_audio = dspy.Predict(ProcessAudio)
            self.combine = dspy.ChainOfThought(CombineMultimodal)

        def forward(self, text, image=None, audio=None):
            """Process multimodal inputs."""
            # Collect information from different modalities
            image_info = "No image provided"
            audio_info = "No audio provided"

            if image:
                img_result = self.analyze_image(image=image, context=text)
                image_info = img_result.analysis

            if audio:
                aud_result = self.process_audio(audio=audio, context=text)
                audio_info = aud_result.information

            # Combine all information
            final_result = self.combine(
                text_input=text,
                image_info=image_info,
                audio_info=audio_info
            )

            return dspy.Prediction(
                response=final_result.response,
                reasoning=final_result.reasoning,
                image_analysis=image_info,
                audio_analysis=audio_info
            )

    assistant = MultimodalAssistant()

    print("\n  This custom module demonstrates:")
    print("  ✓ Processing images with context")
    print("  ✓ Processing audio with context")
    print("  ✓ Combining multimodal information")
    print("  ✓ Generating comprehensive responses")

    print("\n  Example usage:")
    print("    # Process with image and text")
    print("    img = dspy.Image(path='product.jpg')")
    print("    result = assistant(")
    print("        text='What are the key features?',")
    print("        image=img")
    print("    )")
    print("    print(result.response)")
    print()
    print("    # Process with audio and text")
    print("    audio = dspy.Audio(path='review.mp3')")
    print("    result = assistant(")
    print("        text='Summarize the customer feedback',")
    print("        audio=audio")
    print("    )")
    print("    print(result.response)")
    print()
    print("    # Process with all three modalities")
    print("    result = assistant(")
    print("        text='Analyze this product',")
    print("        image=img,")
    print("        audio=audio")
    print("    )")
    print("    print(result.response)")

    # ========== PART 5: Practical Tips ==========
    print_step("Part 5: Practical Tips", "Best practices for multimodal DSPy applications")

    print("  1. Loading Multimodal Data:")
    print("     - From file: dspy.Image(path='image.jpg')")
    print("     - From URL: dspy.Image(url='https://example.com/image.jpg')")
    print("     - From bytes: dspy.Image(data=image_bytes)")
    print()

    print("  2. Supported Formats:")
    print("     - Images: JPEG, PNG, WebP, GIF")
    print("     - Audio: MP3, WAV, OGG, M4A")
    print()

    print("  3. Model Compatibility:")
    print("     - Vision: GPT-4o, GPT-4 Turbo, Claude 3+, Gemini 1.5+")
    print("     - Audio: Gemini 1.5+, Whisper (via OpenAI)")
    print()

    print("  4. Best Practices:")
    print("     - Compress large images before sending")
    print("     - Provide context with multimodal inputs")
    print("     - Handle missing modalities gracefully")
    print("     - Test with different model providers")
    print()

    print("  5. Common Use Cases:")
    print("     - Document analysis (OCR, layout understanding)")
    print("     - Product catalog generation")
    print("     - Accessibility (image descriptions for screen readers)")
    print("     - Content moderation")
    print("     - Medical image analysis")
    print("     - Audio transcription and analysis")
    print("     - Video frame analysis")

    print("\n" + "=" * 80)
    print("Tutorial completed!")
    print("\nKey Takeaways:")
    print("1. DSPy 3.x provides first-class support for multimodal AI")
    print("2. Use dspy.Image for vision tasks (description, VQA, comparison)")
    print("3. Use dspy.Audio for audio tasks (transcription, analysis)")
    print("4. Multimodal fields work seamlessly with DSPy signatures")
    print("5. Combine modalities in custom modules for powerful applications")
    print("6. Always check model compatibility for multimodal features")


if __name__ == "__main__":
    main()
