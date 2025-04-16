"""
Uploading Models to Hugging Face Hub
----------------------------------
This file demonstrates how to upload trained models and checkpoints
to the Hugging Face Model Hub.
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from huggingface_hub import notebook_login, HfApi

def introduction_to_model_upload():
    """
    Introduction to uploading models to the Hugging Face Hub
    """
    print("=" * 50)
    print("UPLOADING MODELS TO HUGGING FACE HUB")
    print("=" * 50)
    
    print("\nKey Steps in the Upload Process:")
    print("1. Create a Hugging Face account")
    print("2. Set up authentication")
    print("3. Prepare the model and tokenizer")
    print("4. Create a model card")
    print("5. Upload using the push_to_hub method")
    print("6. (Optional) Use the Hugging Face CLI for more options")
    
    print("\nRequired Credentials:")
    print("- Hugging Face account (sign up at huggingface.co)")
    print("- Hugging Face access token (generate in account settings)")
    
    print("\nBenefits of Sharing on the Hub:")
    print("- Community access to your models")
    print("- Versioning and model management")
    print("- Collaboration features")
    print("- Integrated inference API")
    print("- Model usage statistics")

def authenticate_to_hub():
    """
    Demonstrate how to authenticate to the Hugging Face Hub
    
    NOTE: This is typically an interactive process where users need
    to enter their token, so this would be adapted in a notebook.
    """
    print("\n" + "=" * 50)
    print("AUTHENTICATING TO HUGGING FACE HUB")
    print("=" * 50)
    
    print("\nOption 1: Using notebook_login() in Jupyter/Colab:")
    print("```python")
    print("from huggingface_hub import notebook_login")
    print("notebook_login()")
    print("```")
    
    print("\nOption 2: Using the CLI (in terminal):")
    print("```bash")
    print("huggingface-cli login")
    print("```")
    
    print("\nOption 3: Using environment variables:")
    print("```python")
    print("import os")
    print('os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"')
    print("```")
    
    print("\nOption 4: Using a token file:")
    print("```python")
    print("from huggingface_hub import HfFolder")
    print('HfFolder.save_token("your_token_here")')
    print("```")
    
    # Simulate login status
    print("\nCurrent status: Not logged in. (This would show your username if logged in)")

def prepare_simple_model():
    """
    Create a simple model that we could upload to the Hub
    """
    print("\n" + "=" * 50)
    print("PREPARING A MODEL FOR UPLOAD")
    print("=" * 50)
    
    try:
        # Let's create a small fine-tuned BERT for sentiment analysis
        print("\nCreating a small fine-tuned model for demonstration...")
        
        # Define the model architecture (simplified for demonstration)
        base_model_name = "prajjwal1/bert-tiny"  # A very small BERT model for demo purposes
        
        # In a real scenario, this would be an actual fine-tuned model
        # Here we're just loading a pre-trained model for demonstration
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, 
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        print(f"Model created: Fine-tuned {base_model_name} for sentiment analysis")
        print(f"Model size: {model.num_parameters():,} parameters")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Make sure you have the required models and dependencies installed.")
        return None, None

def create_model_card():
    """
    Generate an example model card for documentation
    """
    print("\n" + "=" * 50)
    print("CREATING A MODEL CARD")
    print("=" * 50)
    
    model_name = "example-sentiment-bert"
    username = "your-username"  # This would be the actual HF username
    
    # Generate markdown content for the model card
    model_card_content = f"""---
language: en
license: mit
datasets:
  - sst2
tags:
  - text-classification
  - sentiment-analysis
  - bert
---

# {model_name}

This model is a fine-tuned version of [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) for sentiment analysis.

## Model description

This is a fine-tuned version of the prajjwal1/bert-tiny model for binary sentiment classification (positive/negative).
The model was trained on the SST-2 (Stanford Sentiment Treebank) dataset.

## Training procedure

The model was fine-tuned on the SST-2 dataset with the following hyperparameters:
- Learning rate: 2e-5
- Batch size: 16
- Number of epochs: 3
- Optimizer: AdamW
- Weight decay: 0.01
- Warmup steps: 500

## Evaluation results

The model achieves the following results on the SST-2 validation set:
- Accuracy: 84.5%
- F1 score: 0.85

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{username}/{model_name}")
model = AutoModelForSequenceClassification.from_pretrained("{username}/{model_name}")

# Prepare input
text = "I really enjoyed this movie!"
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=1)

# Map predictions to labels
labels = ["negative", "positive"]
result = labels[predictions[0]]
print(f"Sentiment: {result}")
```

## Limitations and bias

This model was fine-tuned on movie reviews and may not generalize well to other domains.
The training data may contain biases that could be reflected in the model's predictions.

## Training infrastructure

The model was trained on a single NVIDIA V100 GPU for approximately 10 minutes.
"""
    
    print("Example Model Card:")
    print("-" * 30)
    print(model_card_content[:500] + "...\n[truncated for brevity]")
    print("-" * 30)
    
    print("\nKey Elements of a Good Model Card:")
    print("1. Model description and architecture")
    print("2. Training data and procedure")
    print("3. Performance metrics and evaluations")
    print("4. Usage examples with code")
    print("5. Limitations, biases, and ethical considerations")
    print("6. License information")
    print("7. Citation information (if applicable)")
    
    return model_card_content

def upload_model_example(model=None, tokenizer=None, model_card_content=None):
    """
    Demonstrate the process of uploading a model to the Hub
    """
    print("\n" + "=" * 50)
    print("UPLOADING A MODEL TO HUGGING FACE HUB")
    print("=" * 50)
    
    # Example repository name
    repo_name = "example-sentiment-bert"
    
    print("\nMethod 1: Using push_to_hub() with a model/tokenizer:")
    print("```python")
    print("# After training or fine-tuning your model:")
    print("model.push_to_hub(repo_name)")
    print("tokenizer.push_to_hub(repo_name)")
    print("```")
    
    print("\nMethod 2: Using the Trainer API:")
    print("```python")
    print("# Configure training arguments with hub integration")
    print("training_args = TrainingArguments(")
    print("    output_dir='./results',")
    print("    push_to_hub=True,")
    print("    hub_model_id=repo_name")
    print(")")
    
    print("\n# Initialize the Trainer")
    print("trainer = Trainer(")
    print("    model=model,")
    print("    args=training_args,")
    print("    train_dataset=train_dataset,")
    print("    eval_dataset=eval_dataset")
    print(")")
    
    print("\n# Train and push to Hub in one step")
    print("trainer.train()")
    print("trainer.push_to_hub()")
    print("```")
    
    print("\nMethod 3: Using the huggingface_hub library directly:")
    print("```python")
    print("from huggingface_hub import HfApi")
    print("api = HfApi()")
    
    print("\n# Create repository (if it doesn't exist)")
    print("api.create_repo(repo_name)")
    
    print("\n# Save model and tokenizer files locally")
    print("model.save_pretrained('./local_model_dir')")
    print("tokenizer.save_pretrained('./local_model_dir')")
    
    print("\n# Upload files to the Hub")
    print("api.upload_folder(")
    print("    folder_path='./local_model_dir',")
    print("    repo_id=f'username/{repo_name}'")
    print(")")
    print("```")
    
    print("\nMethod 4: Using git commands with the huggingface_hub:")
    print("```python")
    print("from huggingface_hub import Repository")
    
    print("\n# Clone the repository")
    print("repo = Repository('./local_repo', clone_from=f'username/{repo_name}')")
    
    print("\n# Save model and tokenizer")
    print("model.save_pretrained('./local_repo')")
    print("tokenizer.save_pretrained('./local_repo')")
    
    print("\n# Add README.md (model card)")
    print("with open('./local_repo/README.md', 'w') as f:")
    print("    f.write(model_card_content)")
    
    print("\n# Push to the Hub")
    print("repo.push_to_hub()")
    print("```")

def using_model_versions_and_tags():
    """
    Demonstrate version management for models on the Hub
    """
    print("\n" + "=" * 50)
    print("MANAGING MODEL VERSIONS AND TAGS")
    print("=" * 50)
    
    print("\nCreating Model Tags:")
    print("```python")
    print("from huggingface_hub import HfApi")
    print("api = HfApi()")
    
    print("\n# Add a tag to a specific commit")
    print("api.tag_revision(")
    print("    repo_id='username/model-name',")
    print("    revision='commit_hash',  # e.g., '8a7bdf52ba5f8b57119075ceb5444f1ebcd4566b'")
    print("    tag='v1.0.0'")
    print(")")
    print("```")
    
    print("\nUpload with Specific Tags:")
    print("```python")
    print("# Push a new version with a tag")
    print("model.push_to_hub('model-name', tags=['v1.0.0', 'stable'])")
    print("```")
    
    print("\nLoading Specific Versions:")
    print("```python")
    print("# Load by tag")
    print("model = AutoModel.from_pretrained('username/model-name', revision='v1.0.0')")
    
    print("\n# Load by commit hash")
    print("model = AutoModel.from_pretrained('username/model-name', revision='8a7bdf52')")
    print("```")
    
    print("\nHugging Face Hub Web Interface:")
    print("1. Browse model versions in the 'Files and versions' tab")
    print("2. Switch between versions using the dropdown menu")
    print("3. Compare changes between versions")
    print("4. Create and manage tags through the web UI")

def using_hub_spaces():
    """
    Introduce Hugging Face Spaces for model demos
    """
    print("\n" + "=" * 50)
    print("CREATING DEMOS WITH HUGGING FACE SPACES")
    print("=" * 50)
    
    print("\nWhat are Hugging Face Spaces?")
    print("- Hosting platform for ML demo apps")
    print("- Built with Gradio or Streamlit")
    print("- Direct integration with Hub models")
    print("- Free hosting for public demos")
    print("- Support for custom Python dependencies")
    
    print("\nCreating a Space with Gradio:")
    print("```python")
    print("# app.py")
    print("import gradio as gr")
    print("from transformers import pipeline")
    
    print("\n# Load model directly from Hub")
    print("classifier = pipeline('sentiment-analysis', model='username/example-sentiment-bert')")
    
    print("\ndef predict(text):")
    print("    result = classifier(text)[0]")
    print("    return result['label'], result['score']")
    
    print("\n# Create Gradio interface")
    print("demo = gr.Interface(")
    print("    fn=predict,")
    print("    inputs=gr.Textbox(placeholder='Enter text to analyze...'),")
    print("    outputs=[")
    print("        gr.Label(label='Sentiment'),")
    print("        gr.Number(label='Confidence')")
    print("    ],")
    print("    title='Sentiment Analysis Demo',")
    print("    description='Analyze the sentiment of any text using our fine-tuned BERT model.'")
    print(")")
    
    print("\ndemo.launch()")
    print("```")
    
    print("\nDeploying to Spaces:")
    print("1. Create a new Space on the Hugging Face Hub")
    print("2. Choose Gradio or Streamlit as the SDK")
    print("3. Upload your app code (app.py)")
    print("4. Add dependencies in requirements.txt")
    print("5. The Space will automatically build and deploy")
    
    print("\nConnecting Models and Spaces:")
    print("- Link your model to your Space in the model card")
    print("- Reference your model in your Space's README")
    print("- Use model cards to showcase Space demos")

if __name__ == "__main__":
    introduction_to_model_upload()
    authenticate_to_hub()
    model, tokenizer = prepare_simple_model()
    model_card_content = create_model_card()
    upload_model_example(model, tokenizer, model_card_content)
    using_model_versions_and_tags()
    using_hub_spaces()
