#!/usr/bin/env python3
"""
Image Generation Prompting with DSPy

This script demonstrates how to use DSPy for improving image generation prompts
through iterative refinement and optimization techniques.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

@dataclass
class ImagePromptResult:
    original_prompt: str
    enhanced_prompt: str
    style_tags: List[str]
    technical_parameters: Dict[str, Any]
    quality_score: float

def main():
    """Main function demonstrating image generation prompting with DSPy."""
    print("=" * 70)
    print("IMAGE GENERATION PROMPTING WITH DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for image prompting")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # DSPy Signatures for Image Generation
    class PromptEnhancement(dspy.Signature):
        """Enhance an image generation prompt for better results."""
        basic_prompt = dspy.InputField(desc="Basic user prompt for image generation")
        style_preference = dspy.InputField(desc="Preferred art style or aesthetic")
        quality_level = dspy.InputField(desc="Desired quality level: basic, good, high, professional")
        
        enhanced_prompt = dspy.OutputField(desc="Detailed, optimized prompt for image generation")
        style_tags = dspy.OutputField(desc="Recommended style and technical tags")
        technical_params = dspy.OutputField(desc="Technical parameters like aspect ratio, resolution suggestions")
    
    class PromptIteration(dspy.Signature):
        """Iteratively improve an image prompt based on feedback."""
        current_prompt = dspy.InputField(desc="Current image generation prompt")
        feedback = dspy.InputField(desc="Feedback on the generated image or desired improvements")
        iteration_number = dspy.InputField(desc="Current iteration number")
        
        improved_prompt = dspy.OutputField(desc="Improved prompt addressing the feedback")
        changes_made = dspy.OutputField(desc="Explanation of changes made to the prompt")
        confidence = dspy.OutputField(desc="Confidence that this iteration will improve results")
    
    class StyleTransfer(dspy.Signature):
        """Adapt a prompt to different artistic styles."""
        base_prompt = dspy.InputField(desc="Base image description prompt")
        target_style = dspy.InputField(desc="Target artistic style (e.g., impressionist, cyberpunk, minimalist)")
        
        adapted_prompt = dspy.OutputField(desc="Prompt adapted for the target style")
        style_elements = dspy.OutputField(desc="Key elements that define this style")
        additional_tags = dspy.OutputField(desc="Additional tags to enhance the style")
    
    class PromptAnalysis(dspy.Signature):
        """Analyze and score an image generation prompt."""
        prompt = dspy.InputField(desc="Image generation prompt to analyze")
        
        clarity_score = dspy.OutputField(desc="Clarity and specificity score (1-10)")
        creativity_score = dspy.OutputField(desc="Creativity and uniqueness score (1-10)")
        technical_score = dspy.OutputField(desc="Technical completeness score (1-10)")
        overall_score = dspy.OutputField(desc="Overall prompt quality score (1-10)")
        improvement_suggestions = dspy.OutputField(desc="Specific suggestions for improvement")
    
    # Image Prompt Optimizer Module
    class ImagePromptOptimizer(dspy.Module):
        """Comprehensive image prompt optimization module."""
        
        def __init__(self):
            super().__init__()
            self.enhancer = dspy.ChainOfThought(PromptEnhancement)
            self.iterator = dspy.ChainOfThought(PromptIteration)
            self.style_adapter = dspy.ChainOfThought(StyleTransfer)
            self.analyzer = dspy.ChainOfThought(PromptAnalysis)
        
        def enhance_prompt(self, basic_prompt: str, style: str = "realistic", quality: str = "high") -> dspy.Prediction:
            """Enhance a basic prompt for better image generation."""
            return self.enhancer(
                basic_prompt=basic_prompt,
                style_preference=style,
                quality_level=quality
            )
        
        def iterate_prompt(self, current_prompt: str, feedback: str, iteration: int = 1) -> dspy.Prediction:
            """Improve a prompt based on feedback."""
            return self.iterator(
                current_prompt=current_prompt,
                feedback=feedback,
                iteration_number=str(iteration)
            )
        
        def adapt_style(self, prompt: str, target_style: str) -> dspy.Prediction:
            """Adapt a prompt to a specific artistic style."""
            return self.style_adapter(
                base_prompt=prompt,
                target_style=target_style
            )
        
        def analyze_prompt(self, prompt: str) -> dspy.Prediction:
            """Analyze and score a prompt."""
            return self.analyzer(prompt=prompt)
        
        def optimize_workflow(self, basic_prompt: str, target_style: str, feedback_iterations: List[str]) -> Dict[str, Any]:
            """Complete optimization workflow."""
            results = {
                'original': basic_prompt,
                'iterations': []
            }
            
            # Initial enhancement
            enhanced = self.enhance_prompt(basic_prompt, target_style, "professional")
            current_prompt = enhanced.enhanced_prompt
            
            results['initial_enhancement'] = {
                'prompt': current_prompt,
                'style_tags': enhanced.style_tags,
                'technical_params': enhanced.technical_params
            }
            
            # Iterative improvements
            for i, feedback in enumerate(feedback_iterations, 1):
                iteration_result = self.iterate_prompt(current_prompt, feedback, i)
                current_prompt = iteration_result.improved_prompt
                
                results['iterations'].append({
                    'iteration': i,
                    'feedback': feedback,
                    'prompt': current_prompt,
                    'changes': iteration_result.changes_made,
                    'confidence': iteration_result.confidence
                })
            
            # Final analysis
            final_analysis = self.analyze_prompt(current_prompt)
            results['final_analysis'] = {
                'clarity': final_analysis.clarity_score,
                'creativity': final_analysis.creativity_score,
                'technical': final_analysis.technical_score,
                'overall': final_analysis.overall_score,
                'suggestions': final_analysis.improvement_suggestions
            }
            
            return results
    
    # Initialize the optimizer
    optimizer = ImagePromptOptimizer()
    print_result("Image prompt optimizer initialized successfully!")
    
    # Demo 1: Basic Prompt Enhancement
    print_step("Basic Prompt Enhancement", "Improving simple prompts for better image generation")
    
    basic_prompts = [
        {
            "prompt": "a cat",
            "style": "photorealistic",
            "quality": "professional"
        },
        {
            "prompt": "sunset over mountains",
            "style": "digital art",
            "quality": "high"
        },
        {
            "prompt": "futuristic city",
            "style": "cyberpunk",
            "quality": "professional"
        }
    ]
    
    for i, prompt_data in enumerate(basic_prompts, 1):
        try:
            enhanced = optimizer.enhance_prompt(
                prompt_data["prompt"],
                prompt_data["style"],
                prompt_data["quality"]
            )
            
            print(f"\n--- Enhancement {i} ---")
            print_result(f"Original: {prompt_data['prompt']}", "Original Prompt")
            print_result(f"Enhanced: {enhanced.enhanced_prompt}", "Enhanced Prompt")
            print_result(f"Style Tags: {enhanced.style_tags}", "Style Tags")
            print_result(f"Technical Parameters: {enhanced.technical_params}", "Technical Parameters")
            
        except Exception as e:
            print_error(f"Error enhancing prompt {i}: {e}")
    
    # Demo 2: Style Adaptation
    print_step("Style Adaptation Demo", "Adapting prompts to different artistic styles")
    
    base_prompt = "A serene landscape with a lake, trees, and mountains in the background"
    artistic_styles = [
        "Van Gogh impressionist style",
        "Japanese minimalist ink painting",
        "Cyberpunk neon aesthetic",
        "Medieval manuscript illumination",
        "Modern abstract geometric"
    ]
    
    for i, style in enumerate(artistic_styles, 1):
        try:
            adapted = optimizer.adapt_style(base_prompt, style)
            
            print(f"\n--- Style Adaptation {i}: {style} ---")
            print_result(f"Base Prompt: {base_prompt}", "Base Prompt")
            print_result(f"Adapted Prompt: {adapted.adapted_prompt}", "Adapted Prompt")
            print_result(f"Style Elements: {adapted.style_elements}", "Style Elements")
            print_result(f"Additional Tags: {adapted.additional_tags}", "Additional Tags")
            
        except Exception as e:
            print_error(f"Error adapting to style {i}: {e}")
    
    # Demo 3: Iterative Prompt Improvement
    print_step("Iterative Improvement Demo", "Improving prompts through feedback iterations")
    
    iteration_example = {
        "initial_prompt": "A robot in a garden",
        "feedback_iterations": [
            "The robot looks too mechanical, make it more friendly and organic",
            "Add more detail to the garden, include flowers and butterflies",
            "Make the lighting more dramatic, like golden hour",
            "Add some interaction between the robot and nature"
        ]
    }
    
    try:
        current_prompt = iteration_example["initial_prompt"]
        print_result(f"Starting Prompt: {current_prompt}", "Initial Prompt")
        
        for i, feedback in enumerate(iteration_example["feedback_iterations"], 1):
            iteration_result = optimizer.iterate_prompt(current_prompt, feedback, i)
            current_prompt = iteration_result.improved_prompt
            
            print(f"\n--- Iteration {i} ---")
            print_result(f"Feedback: {feedback}", "Feedback")
            print_result(f"Improved Prompt: {current_prompt}", "Improved Prompt")
            print_result(f"Changes Made: {iteration_result.changes_made}", "Changes Made")
            print_result(f"Confidence: {iteration_result.confidence}", "Confidence Level")
        
    except Exception as e:
        print_error(f"Error in iterative improvement: {e}")
    
    # Demo 4: Prompt Analysis and Scoring
    print_step("Prompt Analysis Demo", "Analyzing and scoring different prompts")
    
    test_prompts = [
        "cat",  # Very basic
        "A beautiful sunset over the ocean with waves crashing on the shore",  # Decent
        "A hyper-realistic portrait of an elderly wizard with intricate silver beard, wise blue eyes, wearing ornate robes with golden embroidery, standing in an ancient library filled with floating books, magical particles in the air, warm candlelight, shot with 85mm lens, shallow depth of field, professional photography",  # Very detailed
        "Chaotic abstract explosion of colors and shapes representing the concept of time, incorporating clockwork elements, melting clocks in Salvador Dali style, cosmic background with nebulae, digital art, 4K resolution"  # Creative and technical
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            analysis = optimizer.analyze_prompt(prompt)
            
            print(f"\n--- Analysis {i} ---")
            print_result(f"Prompt: {prompt}", "Test Prompt")
            print_result(f"Clarity Score: {analysis.clarity_score}/10", "Clarity")
            print_result(f"Creativity Score: {analysis.creativity_score}/10", "Creativity")
            print_result(f"Technical Score: {analysis.technical_score}/10", "Technical")
            print_result(f"Overall Score: {analysis.overall_score}/10", "Overall")
            print_result(f"Suggestions: {analysis.improvement_suggestions}", "Improvement Suggestions")
            
        except Exception as e:
            print_error(f"Error analyzing prompt {i}: {e}")
    
    # Demo 5: Complete Optimization Workflow
    print_step("Complete Optimization Workflow", "Full end-to-end prompt optimization")
    
    workflow_example = {
        "basic_prompt": "A magical forest",
        "target_style": "fantasy digital art",
        "feedback_iterations": [
            "Add more magical elements like glowing mushrooms and fairy lights",
            "Make the trees more ancient and twisted",
            "Include a mystical creature like a unicorn or dragon"
        ]
    }
    
    try:
        results = optimizer.optimize_workflow(
            workflow_example["basic_prompt"],
            workflow_example["target_style"],
            workflow_example["feedback_iterations"]
        )
        
        print_result(f"Original: {results['original']}", "Original Prompt")
        print_result(f"Enhanced: {results['initial_enhancement']['prompt']}", "Initial Enhancement")
        
        for iteration in results['iterations']:
            print(f"\n--- Iteration {iteration['iteration']} ---")
            print_result(f"Feedback: {iteration['feedback']}", "Feedback")
            print_result(f"Result: {iteration['prompt']}", "Improved Prompt")
            print_result(f"Confidence: {iteration['confidence']}", "Confidence")
        
        print("\n--- Final Analysis ---")
        final = results['final_analysis']
        print_result(f"Overall Score: {final['overall']}/10", "Final Score")
        print_result(f"Clarity: {final['clarity']}/10, Creativity: {final['creativity']}/10, Technical: {final['technical']}/10", "Detailed Scores")
        print_result(f"Suggestions: {final['suggestions']}", "Final Suggestions")
        
    except Exception as e:
        print_error(f"Error in optimization workflow: {e}")
    
    # Demo 6: Specialized Prompt Categories
    print_step("Specialized Categories Demo", "Prompts for different image generation needs")
    
    specialized_categories = [
        {
            "category": "Product Photography",
            "prompt": "smartphone",
            "style": "commercial product photography",
            "requirements": "white background, professional lighting, high detail"
        },
        {
            "category": "Concept Art",
            "prompt": "alien spaceship",
            "style": "sci-fi concept art",
            "requirements": "detailed technical design, multiple angles, annotations"
        },
        {
            "category": "Character Design",
            "prompt": "fantasy warrior",
            "style": "character sheet",
            "requirements": "front and side view, detailed costume, weapon design"
        },
        {
            "category": "Architecture",
            "prompt": "modern house",
            "style": "architectural visualization",
            "requirements": "realistic materials, proper lighting, landscaping"
        }
    ]
    
    for category_data in specialized_categories:
        try:
            # Create enhanced prompt with category-specific requirements
            enhanced_prompt = f"{category_data['prompt']}, {category_data['requirements']}"
            enhanced = optimizer.enhance_prompt(enhanced_prompt, category_data['style'], "professional")
            
            print(f"\n--- {category_data['category']} ---")
            print_result(f"Basic: {category_data['prompt']}", "Basic Prompt")
            print_result(f"Enhanced: {enhanced.enhanced_prompt}", "Enhanced Prompt")
            print_result(f"Style Tags: {enhanced.style_tags}", "Style Tags")
            
        except Exception as e:
            print_error(f"Error with {category_data['category']}: {e}")
    
    print("\n" + "="*70)
    print("IMAGE GENERATION PROMPTING COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy-powered image prompt optimization!")

if __name__ == "__main__":
    main()
