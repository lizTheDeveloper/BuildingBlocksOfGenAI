# Introduction to the Hugging Face Ecosystem

## What is Hugging Face?

- Open-source platform for machine learning
- Community-driven hub for models, datasets, and applications
- 120,000+ models, 20,000+ datasets, and growing
- Documentation, tutorials, and learning resources
- Commercial offerings (Inference API, AutoTrain)

## Core Components

1. **🤗 Transformers**: Pre-trained models for various ML tasks
2. **🤗 Datasets**: Unified API for accessing and processing datasets
3. **🤗 Tokenizers**: Fast and optimized text tokenization
4. **🤗 Hub**: Platform for sharing models, datasets, and demos
5. **🤗 Spaces**: Interactive ML demos and applications

## Why Hugging Face Matters

- **Democratizes AI**: Makes advanced models accessible to everyone
- **Reduces Duplication**: Reuse instead of rebuilding
- **Accelerates Research**: Build on state-of-the-art models
- **Standardizes Workflows**: Consistent APIs across models
- **Fosters Collaboration**: Community-driven development

## The Transformers Library

```python
from transformers import pipeline

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "I have a problem with my iPhone that needs to be fixed.",
    candidate_labels=["urgent", "not urgent", "phone", "computer", "billing"],
)
print(result)
```

## The Datasets Library

```python
from datasets import load_dataset

# Load a dataset
dataset = load_dataset("glue", "mnli")

# Basic information
print(f"Features: {dataset['train'].features}")
print(f"Dataset size: {len(dataset['train'])}")

# First example
print(dataset['train'][0])
```

## Using Pre-trained Models

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input and get prediction
inputs = tokenizer("I love this course!", return_tensors="pt")
outputs = model(**inputs)

# Get prediction
import torch
prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(prediction)  # [negative_score, positive_score]
```

## The Model Hub

- **Search** by task, framework, size, license
- **Documentation** through model cards
- **Version control** for models
- **Collaboration** tools
- **Metrics** and leaderboards
- **Inference API** for quick testing

## Today's Hugging Face Exercises

1. Explore the Hugging Face ecosystem
2. Load and use pre-trained models
3. Work with the Datasets library
4. Create a simple pipeline
5. Upload a checkpoint to Hugging Face

---

# Let's Get Started!
