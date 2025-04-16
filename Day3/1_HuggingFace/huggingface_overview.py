"""
Hugging Face Ecosystem Overview
------------------------------
This file introduces the core components of the Hugging Face ecosystem
and demonstrates how they fit together.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

def introduction_to_huggingface():
    """
    Print an overview of the Hugging Face ecosystem and its main components.
    """
    print("=" * 50)
    print("INTRODUCTION TO THE HUGGING FACE ECOSYSTEM")
    print("=" * 50)
    
    print("\n1. Core Libraries:")
    print("   - Transformers: Pre-trained models for various ML tasks")
    print("   - Datasets: Unified API for accessing and processing datasets")
    print("   - Tokenizers: Fast and optimized text tokenization")
    print("   - Accelerate: Training optimization across hardware")
    print("   - Evaluate: Standardized evaluation metrics")
    
    print("\n2. Model Hub:")
    print("   - Community-driven repository of pre-trained models")
    print("   - Over 120,000 models available for various tasks")
    print("   - Searchable by task, language, framework, etc.")
    print("   - Model cards with documentation and usage examples")
    
    print("\n3. Datasets Hub:")
    print("   - Thousands of datasets ready for ML tasks")
    print("   - Consistent API for loading and processing")
    print("   - Support for various formats and domains")
    
    print("\n4. Spaces:")
    print("   - Interactive web demos of ML models")
    print("   - Built with Gradio or Streamlit")
    print("   - Showcasing model capabilities and applications")

def check_installed_versions():
    """
    Check and print the installed versions of Hugging Face libraries.
    """
    try:
        import transformers
        import datasets
        
        # Try to import optional libraries
        try:
            import accelerate
            accelerate_version = accelerate.__version__
        except ImportError:
            accelerate_version = "Not installed"
            
        try:
            import evaluate
            evaluate_version = evaluate.__version__
        except ImportError:
            evaluate_version = "Not installed"
        
        print("\nINSTALLED HUGGING FACE LIBRARIES:")
        print(f"transformers: {transformers.__version__}")
        print(f"datasets: {datasets.__version__}")
        print(f"accelerate: {accelerate_version}")
        print(f"evaluate: {evaluate_version}")
        print(f"PyTorch: {torch.__version__}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install the required libraries:")
        print("pip install transformers datasets")
        print("Optional: pip install accelerate evaluate")

def list_available_models(task=None, limit=5):
    """
    List available models from Hugging Face Hub for a specific task.
    
    Args:
        task (str, optional): Filter by task, e.g., 'text-classification', 'translation'
        limit (int): Number of models to display
    """
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        if task:
            print(f"\nPOPULAR MODELS FOR {task.upper()}:")
            models = api.list_models(filter=task, sort="downloads", direction=-1, limit=limit)
        else:
            print(f"\nPOPULAR MODELS ACROSS ALL TASKS:")
            models = api.list_models(sort="downloads", direction=-1, limit=limit)
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model.id} - {model.downloads:,} downloads")
            
    except ImportError:
        print("\nError: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")

def simple_pipeline_demo():
    """
    Demonstrate the use of Hugging Face pipelines for quick inference.
    """
    print("\n" + "=" * 50)
    print("HUGGING FACE PIPELINE DEMO")
    print("=" * 50)
    
    # Example texts for different tasks
    text_classification_example = "I absolutely loved this movie! The acting was superb."
    summarization_example = """
    The Apollo program was the third United States human spaceflight program. 
    It achieved its goal of landing humans on the Moon with Apollo 11 in 1969. 
    NASA launched a total of 11 Apollo expeditions, and six of them landed 
    astronauts successfully on the Moon between 1969 and 1972. Neil Armstrong 
    was the first person to set foot on the lunar surface. The program concluded 
    in 1972 with Apollo 17, the last manned mission to the Moon to date. During 
    the Apollo missions, 12 astronauts walked on the Moon's surface, taking 
    photographs, conducting experiments, and collecting soil and rock samples.
    """
    qa_example = {
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018."
    }
    
    try:
        # Sentiment analysis pipeline
        print("\n1. TEXT CLASSIFICATION (SENTIMENT ANALYSIS)")
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_result = sentiment_analyzer(text_classification_example)
        print(f"Text: {text_classification_example}")
        print(f"Result: {sentiment_result[0]['label']} (score: {sentiment_result[0]['score']:.4f})")
        
        # Text summarization pipeline
        print("\n2. TEXT SUMMARIZATION")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(summarization_example, max_length=75, min_length=30, do_sample=False)
        print(f"Original text length: {len(summarization_example.split())} words")
        print(f"Summary: {summary[0]['summary_text']}")
        print(f"Summary length: {len(summary[0]['summary_text'].split())} words")
        
        # Question answering pipeline
        print("\n3. QUESTION ANSWERING")
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        qa_result = qa_pipeline(qa_example)
        print(f"Question: {qa_example['question']}")
        print(f"Context: {qa_example['context']}")
        print(f"Answer: {qa_result['answer']} (score: {qa_result['score']:.4f})")
        
    except Exception as e:
        print(f"Error during pipeline demo: {e}")
        print("Make sure you have the required models downloaded and enough memory.")

if __name__ == "__main__":
    introduction_to_huggingface()
    check_installed_versions()
    list_available_models()
    simple_pipeline_demo()
