"""
Hugging Face "Hello World" Demo
-----------------------------
This file provides a simple demo of using Hugging Face for various NLP tasks.
It's designed as a quick hands-on introduction to show the power and ease of use
of the Hugging Face ecosystem.
"""

import torch
from transformers import pipeline
from time import time

def hello_world_introduction():
    """
    Introduction to the "Hello World" demo
    """
    print("=" * 60)
    print("HUGGING FACE 'HELLO WORLD' DEMO")
    print("=" * 60)
    
    print("\nThis demo will show you how to use Hugging Face for common NLP tasks:")
    print("1. Text classification")
    print("2. Named Entity Recognition (NER)")
    print("3. Question answering")
    print("4. Text generation")
    print("5. Translation")
    print("6. Summarization")
    
    print("\nEach example uses the pipeline API, which provides a simple and")
    print("consistent interface for working with state-of-the-art NLP models.")

def check_gpu_availability():
    """
    Check if GPU is available and set appropriate device
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU detected, using CPU only.")
        print("Tasks will run slower but all examples will still work.")
    
    return device

def text_classification_example():
    """
    Demonstrate sentiment analysis using the pipeline API
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: TEXT CLASSIFICATION (SENTIMENT ANALYSIS)")
    print("=" * 60)
    
    print("\nLoading sentiment analysis pipeline...")
    start_time = time()
    
    # Create a sentiment analysis pipeline using a small model
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=check_gpu_availability()
    )
    
    print(f"Pipeline loaded in {time() - start_time:.2f} seconds")
    
    # Example sentences to analyze
    texts = [
        "I absolutely loved this movie! The acting was phenomenal.",
        "The service at this restaurant was terrible and the food was overpriced.",
        "The product works as expected, nothing special but does the job.",
        "I can't believe how incredible this experience was!"
    ]
    
    print("\nAnalyzing example texts...")
    start_time = time()
    
    # Process all examples
    results = classifier(texts)
    
    print(f"Analysis completed in {time() - start_time:.2f} seconds")
    
    # Display results
    print("\nResults:")
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\nText {i+1}: \"{text}\"")
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})")
    
    print("\nTry it yourself! Enter a text to analyze (or press Enter to skip):")
    print("user_text = input('> ')")
    print("if user_text:")
    print("    result = classifier(user_text)[0]")
    print("    print(f\"Sentiment: {result['label']} (confidence: {result['score']:.4f})\")")

def named_entity_recognition_example():
    """
    Demonstrate Named Entity Recognition using the pipeline API
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: NAMED ENTITY RECOGNITION")
    print("=" * 60)
    
    print("\nLoading named entity recognition pipeline...")
    start_time = time()
    
    # Create an NER pipeline
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=check_gpu_availability()
    )
    
    print(f"Pipeline loaded in {time() - start_time:.2f} seconds")
    
    # Example sentences for NER
    texts = [
        "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California.",
        "The Eiffel Tower in Paris, France was completed in 1889 for the World's Fair.",
        "OpenAI released GPT-4 in March 2023, improving upon GPT-3."
    ]
    
    print("\nRecognizing named entities in example texts...")
    start_time = time()
    
    # Process the examples
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: \"{text}\"")
        entities = ner(text)
        
        print("Entities:")
        for entity in entities:
            print(f"  • {entity['word']} - {entity['entity_group']} (confidence: {entity['score']:.4f})")
    
    print(f"\nProcessing completed in {time() - start_time:.2f} seconds")
    
    print("\nTry it yourself! Enter a text to analyze (or press Enter to skip):")
    print("user_text = input('> ')")
    print("if user_text:")
    print("    entities = ner(user_text)")
    print("    print('Entities:')")
    print("    for entity in entities:")
    print("        print(f\"  • {entity['word']} - {entity['entity_group']} (confidence: {entity['score']:.4f})\")")

def question_answering_example():
    """
    Demonstrate Question Answering using the pipeline API
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: QUESTION ANSWERING")
    print("=" * 60)
    
    print("\nLoading question answering pipeline...")
    start_time = time()
    
    # Create a question answering pipeline
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=check_gpu_availability()
    )
    
    print(f"Pipeline loaded in {time() - start_time:.2f} seconds")
    
    # Example context and questions
    context = """
    The Transformer architecture was introduced in the paper "Attention is All You Need" 
    by researchers at Google in 2017. Unlike previous sequence-to-sequence models that used 
    recurrent neural networks (RNNs), Transformers rely entirely on attention mechanisms, 
    which allow them to process input sequences in parallel rather than sequentially. This 
    parallel processing enables more efficient training on modern hardware like GPUs and TPUs. 
    The original Transformer model has been adapted into many variants, including BERT 
    (developed by Google), GPT (created by OpenAI), and T5 (also by Google), which have achieved 
    state-of-the-art results on a wide range of natural language processing tasks.
    """
    
    questions = [
        "Who introduced the Transformer architecture?",
        "What's the main difference between Transformers and previous models?",
        "Which hardware benefits from Transformer's parallel processing?",
        "What are some variants of the Transformer model?"
    ]
    
    print("\nAnswering example questions...")
    start_time = time()
    
    # Process each question
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: \"{question}\"")
        result = qa(question=question, context=context)
        print(f"Answer: \"{result['answer']}\"")
        print(f"Confidence: {result['score']:.4f}")
    
    print(f"\nProcessing completed in {time() - start_time:.2f} seconds")
    
    print("\nTry it yourself! Enter a question about the context (or press Enter to skip):")
    print("user_question = input('> ')")
    print("if user_question:")
    print("    result = qa(question=user_question, context=context)")
    print("    print(f\"Answer: \\\"{result['answer']}\\\"\")")
    print("    print(f\"Confidence: {result['score']:.4f}\")")

def text_generation_example():
    """
    Demonstrate Text Generation using the pipeline API
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: TEXT GENERATION")
    print("=" * 60)
    
    print("\nLoading text generation pipeline...")
    start_time = time()
    
    # Create a text generation pipeline using a small model
    generator = pipeline(
        "text-generation",
        model="distilgpt2",  # Much smaller than full GPT-2
        device=check_gpu_availability()
    )
    
    print(f"Pipeline loaded in {time() - start_time:.2f} seconds")
    
    # Example prompts for text generation
    prompts = [
        "The future of artificial intelligence is",
        "In a world where robots have emotions,",
        "The most important scientific discovery was"
    ]
    
    print("\nGenerating text from example prompts...")
    start_time = time()
    
    # Generate text for each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: \"{prompt}\"")
        
        # Generate with some randomness
        result = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        generated_text = result[0]["generated_text"]
        print(f"Generated: \"{generated_text}\"")
    
    print(f"\nGeneration completed in {time() - start_time:.2f} seconds")
    
    print("\nTry it yourself! Enter a prompt for text generation (or press Enter to skip):")
    print("user_prompt = input('> ')")
    print("if user_prompt:")
    print("    result = generator(user_prompt, max_length=50, num_return_sequences=1, temperature=0.7)")
    print("    generated_text = result[0]['generated_text']")
    print("    print(f\"Generated: \\\"{generated_text}\\\"\")")

def translation_example():
    """
    Demonstrate Translation using the pipeline API
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: TRANSLATION")
    print("=" * 60)
    
    print("\nLoading translation pipeline...")
    start_time = time()
    
    # Create a translation pipeline
    translator = pipeline(
        "translation_en_to_fr",
        model="t5-small",  # Using a smaller model for demonstration
        device=check_gpu_availability()
    )
    
    print(f"Pipeline loaded in {time() - start_time:.2f} seconds")
    
    # Example sentences to translate
    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning models can learn from data to make predictions.",
        "Neural networks are inspired by the human brain."
    ]
    
    print("\nTranslating example texts from English to French...")
    start_time = time()
    
    # Translate each text
    for i, text in enumerate(texts):
        print(f"\nEnglish {i+1}: \"{text}\"")
        translation = translator(text, max_length=100)[0]
        print(f"French: \"{translation['translation_text']}\"")
    
    print(f"\nTranslation completed in {time() - start_time:.2f} seconds")
    
    print("\nTry it yourself! Enter a text to translate (or press Enter to skip):")
    print("user_text = input('> ')")
    print("if user_text:")
    print("    translation = translator(user_text, max_length=100)[0]")
    print("    print(f\"French: \\\"{translation['translation_text']}\\\"\")")

def summarization_example():
    """
    Demonstrate Text Summarization using the pipeline API
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: TEXT SUMMARIZATION")
    print("=" * 60)
    
    print("\nLoading summarization pipeline...")
    start_time = time()
    
    # Create a summarization pipeline
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",  # Can use "t5-small" for a smaller model
        device=check_gpu_availability()
    )
    
    print(f"Pipeline loaded in {time() - start_time:.2f} seconds")
    
    # Example text to summarize
    text = """
    Artificial intelligence (AI) has rapidly evolved over the past decade, transforming various sectors 
    including healthcare, finance, transportation, and entertainment. Machine learning algorithms, 
    particularly deep learning models, have demonstrated remarkable capabilities in processing vast 
    amounts of data and identifying patterns that would be impossible for humans to detect manually.
    
    Recent advancements in natural language processing have led to the development of large language 
    models that can understand and generate human-like text, translate between languages, and even 
    write creative content. Computer vision has similarly progressed, enabling systems to recognize 
    objects, faces, and activities in images and videos with high accuracy.
    
    Despite these achievements, AI still faces significant challenges. Ensuring that AI systems are 
    transparent, explainable, and free from harmful biases remains an active area of research. 
    Additionally, questions about the ethical implications of increasingly autonomous AI systems 
    continue to be debated among researchers, policymakers, and the public.
    
    As AI technology continues to advance, it will likely become even more integrated into our daily 
    lives, creating new opportunities and challenges for society to navigate. The future development 
    of AI will require careful consideration of technical, ethical, and societal factors to ensure 
    that these powerful technologies benefit humanity.
    """
    
    print("\nSummarizing example text...")
    start_time = time()
    
    # Generate summary
    summary = summarizer(
        text,
        max_length=100,
        min_length=30,
        do_sample=False
    )[0]["summary_text"]
    
    print(f"\nOriginal text length: {len(text.split())} words")
    print(f"Summary length: {len(summary.split())} words")
    print(f"\nSummary:\n\"{summary}\"")
    
    print(f"\nSummarization completed in {time() - start_time:.2f} seconds")
    
    print("\nNote: You can try this with your own text but be aware that")
    print("summarization works best with longer texts (several paragraphs).")

def main():
    """
    Run the complete "Hello World" demo
    """
    hello_world_introduction()
    device = check_gpu_availability()
    
    # Run all examples
    text_classification_example()
    named_entity_recognition_example()
    question_answering_example()
    text_generation_example()
    translation_example()
    summarization_example()
    
    print("\n" + "=" * 60)
    print("HUGGING FACE 'HELLO WORLD' DEMO COMPLETED")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Explore the Hugging Face Hub: https://huggingface.co/models")
    print("2. Try different models for each task")
    print("3. Fine-tune these models on your own data")
    print("4. Use the Hugging Face Trainer API for more advanced training")
    print("5. Explore multimodal models that combine text, images, audio, etc.")

if __name__ == "__main__":
    main()
