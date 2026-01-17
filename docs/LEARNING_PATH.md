# DSPy Learning Path

Welcome to the DSPy Demo Project! This guide will help you navigate the tutorials in the optimal order for learning DSPy from beginner to advanced user.

## Quick Start (30 minutes)

If you're brand new to DSPy, start here:

1. **Getting Started** (`01_fundamentals/getting_started`)
   - What DSPy is and why it's useful
   - Setting up your first language model
   - Creating basic signatures and modules
   - Making predictions

2. **Signatures Deep Dive** (`01_fundamentals/02_signatures`)
   - Understanding DSPy signatures
   - Input and output fields
   - Writing effective docstrings
   - Best practices for signature design

3. **Your First Program** (Pick one based on interest)
   - `03_building_programs/classification` - Text classification
   - `03_building_programs/entity_extraction` - Entity extraction
   - `03_building_programs/rag_system` - Retrieval-augmented generation

## Beginner Track (2-4 hours)

For those new to DSPy or AI development:

### Week 1: Fundamentals (4 tutorials)

1. `01_fundamentals/getting_started` - Introduction to DSPy
2. `01_fundamentals/02_signatures` - Defining input/output specifications
3. `01_fundamentals/03_modules` - Building reusable components
4. `01_fundamentals/04_predictions` - Working with predictions

**Learning Goals:**
- Understand DSPy's core concepts
- Create your own signatures
- Build simple modules
- Make and evaluate predictions

### Week 2: Core Modules (5 tutorials)

1. `02_core_modules/01_predict` - Basic predictions
2. `02_core_modules/02_chain_of_thought` - Adding reasoning steps
3. `02_core_modules/03_program_of_thought` - Code-based reasoning
4. `02_core_modules/04_react_agents` - Building agents with ReAct
5. `02_core_modules/05_reasoning` - Using reasoning models (NEW in DSPy 3.x)

**Learning Goals:**
- Master different prediction modules
- Understand when to use each type
- Build your first agent
- Work with reasoning-capable models

### Week 3: Building Programs (5 tutorials)

1. `03_building_programs/01_custom_modules` - Creating custom modules
2. `03_building_programs/02_rag_system` - Retrieval-augmented generation
3. `03_building_programs/03_classification` - Text classification
4. `03_building_programs/04_entity_extraction` - Named entity recognition
5. `03_building_programs/05_multi_stage_pipelines` - Complex workflows

**Learning Goals:**
- Build production-ready modules
- Implement RAG systems
- Create classification systems
- Design multi-stage pipelines

## Intermediate Track (4-8 hours)

For those comfortable with basics and ready to optimize:

### Week 4: Optimization (5 tutorials)

1. `04_optimization/01_bootstrap_fewshot` - Quick optimization
2. `04_optimization/02_miprov2` - Bayesian optimization (UPDATED)
3. `04_optimization/03_gepa` - Genetic-Pareto optimization (NEW in DSPy 3.x)
4. `04_optimization/04_simba` - Self-reflective improvement (NEW in DSPy 3.x)
5. `04_optimization/05_finetuning` - Model fine-tuning

**Learning Goals:**
- Understand different optimization strategies
- Choose the right optimizer for your task
- Evaluate and compare optimized models
- Fine-tune for production use

### Week 5: Advanced Features (5 tutorials)

1. `05_advanced/01_multimodal` - Images and audio (NEW in DSPy 3.x)
2. `05_advanced/02_adapters` - ChatAdapter, JSONAdapter, XMLAdapter (NEW)
3. `05_advanced/03_async_batch` - Async and batch processing
4. `05_advanced/04_usage_tracking` - Monitor costs and usage (NEW in DSPy 3.x)
5. `05_advanced/05_multi_hop_rag` - Advanced RAG techniques

**Learning Goals:**
- Work with multimodal inputs (images, audio)
- Use adapters for structured outputs
- Optimize for performance with async
- Track and optimize costs

## Advanced Track (8+ hours)

For those building production systems:

### Week 6: Deployment (4 tutorials)

1. `06_deployment/01_saving_loading` - Model persistence
2. `06_deployment/02_streaming` - Streaming responses
3. `06_deployment/03_caching` - Efficient caching strategies
4. `06_deployment/04_production_tips` - Best practices

**Learning Goals:**
- Deploy DSPy applications to production
- Implement caching for efficiency
- Handle streaming responses
- Follow production best practices

### Week 7: Real-World Applications (3+ tutorials)

1. `07_real_world/ai_text_game` - Interactive game with agents
2. `07_real_world/email_extraction` - Email processing
3. `07_real_world/yahoo_finance_analysis` - Financial analysis
4. More examples available in the directory

**Learning Goals:**
- Apply DSPy to real-world problems
- Build end-to-end applications
- Integrate with external systems
- Handle edge cases and errors

## Specialized Paths

### Path 1: Conversational AI
For building chatbots and conversational systems:

1. `01_fundamentals/getting_started` - Basics
2. `02_core_modules/02_chain_of_thought` - Reasoning
3. `02_core_modules/04_react_agents` - Agents
4. `05_advanced/02_adapters` - ChatAdapter
5. `03_building_programs/customer_service_agent` - Full example

### Path 2: Data Extraction & Analysis
For extracting and analyzing structured data:

1. `01_fundamentals/getting_started` - Basics
2. `03_building_programs/03_classification` - Classification
3. `03_building_programs/04_entity_extraction` - Entity extraction
4. `05_advanced/02_adapters` - JSONAdapter for structured outputs
5. `07_real_world/email_extraction` - Real-world example

### Path 3: RAG Systems
For building knowledge-intensive applications:

1. `01_fundamentals/getting_started` - Basics
2. `03_building_programs/02_rag_system` - Basic RAG
3. `04_optimization/01_bootstrap_fewshot` - Optimize RAG
4. `05_advanced/05_multi_hop_rag` - Advanced RAG
5. `05_advanced/01_multimodal` - Multimodal RAG (images, PDFs)

### Path 4: Research & Optimization
For those focused on improving model performance:

1. `02_core_modules/05_reasoning` - Reasoning models
2. `04_optimization/01_bootstrap_fewshot` - Basic optimization
3. `04_optimization/02_miprov2` - Bayesian optimization
4. `04_optimization/03_gepa` - Genetic-Pareto optimization
5. `04_optimization/04_simba` - Self-reflective improvement
6. `05_advanced/04_usage_tracking` - Monitor and optimize costs

### Path 5: Multimodal AI
For building vision and audio applications:

1. `01_fundamentals/getting_started` - Basics
2. `02_core_modules/01_predict` - Basic predictions
3. `05_advanced/01_multimodal` - Images and audio
4. `03_building_programs/image_generation_prompting` - Image generation
5. `03_building_programs/audio` - Audio processing

## DSPy 3.x New Features Path

If you're already familiar with DSPy 2.x and want to learn what's new in 3.x:

1. **Core New Modules**
   - `02_core_modules/05_reasoning` - dspy.Reasoning for o1/Claude thinking

2. **Multimodal Support**
   - `05_advanced/01_multimodal` - dspy.Image and dspy.Audio

3. **Adapters**
   - `05_advanced/02_adapters` - ChatAdapter, JSONAdapter, XMLAdapter

4. **New Optimizers**
   - `04_optimization/02_miprov2` - Updated MIPROv2 with Bayesian optimization
   - `04_optimization/03_gepa` - Genetic-Pareto optimizer
   - `04_optimization/04_simba` - Self-reflective improvement

5. **Usage Tracking**
   - `05_advanced/04_usage_tracking` - Built-in token and cost tracking

6. **Updated APIs**
   - Check `utils/__init__.py` for new helper functions
   - See CLAUDE.md for migration guide

## Tips for Effective Learning

### 1. Hands-On Practice
- Run every example in both script and notebook formats
- Modify examples to solve your own problems
- Experiment with different parameters

### 2. Build Projects
After each section, try building something:
- **After Fundamentals:** Simple Q&A chatbot
- **After Core Modules:** Agent that uses multiple reasoning strategies
- **After Building Programs:** Classification or RAG system for your domain
- **After Optimization:** Optimize your previous projects
- **After Advanced:** Multimodal application or cost-optimized system

### 3. Join the Community
- GitHub: https://github.com/stanfordnlp/dspy
- Documentation: https://dspy.ai
- Ask questions in discussions or issues

### 4. Reference Material
Keep these handy:
- `CLAUDE.md` - Project-specific guidance
- `README.md` - Project overview
- `utils/` - Helper functions and utilities

## Estimated Time Commitments

- **Quick Start:** 30 minutes
- **Beginner Track:** 2-4 hours (spread over 3 weeks)
- **Intermediate Track:** 4-8 hours (spread over 2 weeks)
- **Advanced Track:** 8+ hours (spread over 2 weeks)
- **Full Path:** 15-25 hours (spread over 7-10 weeks)

## Next Steps

1. **Choose your path** based on your goals and experience
2. **Set up your environment** following README.md
3. **Start with the first tutorial** in your chosen path
4. **Join the community** to ask questions and share projects
5. **Build something real** to solidify your learning

Happy learning! 🚀
