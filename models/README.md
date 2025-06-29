# Sample Models Directory

This directory contains pre-trained and example DSPy models for immediate use in tutorials and demonstrations.

## Contents

### Pre-trained Models

- `basic_qa/` - Simple question-answering models
- `classification/` - Text classification models
- `entity_extraction/` - Named entity recognition models
- `rag_systems/` - Retrieval-augmented generation models

### Model Configurations

- `configs/` - Configuration files for different model setups
- `optimization_results/` - Results from DSPy optimization runs

### Example Deployments

- `production_ready/` - Models ready for production deployment
- `experimental/` - Experimental model configurations

## Usage

Models can be loaded directly in your DSPy programs:

```python
import dspy

# Load a pre-trained model
model = dspy.Module()
model.load('models/basic_qa/optimized_qa_model.json')

# Or use with configuration
config = dspy.Config.load('models/configs/production_config.json')
dspy.configure(config)
```

## Model Versioning

Models are organized by:

- Version numbers (v1.0, v1.1, etc.)
- Training date
- Performance metrics
- Target use case

Each model directory contains:

- `model.json` - The serialized DSPy model
- `metadata.json` - Model information and metrics
- `config.json` - Training and optimization configuration
- `README.md` - Model description and usage instructions
