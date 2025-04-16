"""
Hugging Face Model Inference
--------------------------
This file demonstrates how to load and use pre-trained models from 
Hugging Face for various NLP tasks through direct model usage.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModelForCausalLM
)
import numpy as np

def introduction_to_model_inference():
    """
    Introduction to model inference with Hugging Face Transformers
    """
    print("=" * 50)
    print("HUGGING FACE MODEL INFERENCE")
    print("=" * 50)
    
    print("\nKey Benefits of Using Hugging Face Models:")
    print("1. Access to state-of-the-art pre-trained models")
    print("2. Consistent API across model architectures")
    print("3. Easy switching between models for benchmarking")
    print("4. Optimized implementations for inference and fine-tuning")
    print("5. Support for various hardware accelerators")
    
    print("\nCommon Model Types:")
    print("- Sequence Classification: Text classification, sentiment analysis")
    print("- Token Classification: Named entity recognition, part-of-speech tagging")
    print("- Question Answering: Extractive Q&A from context")
    print("- Masked Language Modeling: Fill-in-the-blank predictions")
    print("- Causal Language Modeling: Text generation")
    print("- Seq2Seq Modeling: Summarization, translation")

def text_classification_example():
    """
    Demonstrate text classification with a pre-trained model
    """
    print("\n" + "=" * 50)
    print("TEXT CLASSIFICATION EXAMPLE")
    print("=" * 50)
    
    try:
        # Load pre-trained model and tokenizer for sentiment analysis
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Example texts
        texts = [
            "I absolutely loved this movie! It was fantastic.",
            "This film was terrible and a waste of time.",
            "The movie was okay, neither great nor bad."
        ]
        
        print(f"\nModel: {model_name}")
        print(f"Task: Sentiment Analysis (Positive/Negative)")
        
        # Process each text
        for i, text in enumerate(texts):
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()
            
            # Map class IDs to labels (specific to this model)
            class_names = ["Negative", "Positive"]
            predicted_label = class_names[predicted_class]
            
            # Print results
            print(f"\nText {i+1}: \"{text}\"")
            print(f"Prediction: {predicted_label} (Confidence: {confidence:.4f})")
            print(f"All scores: Negative: {predictions[0][0]:.4f}, Positive: {predictions[0][1]:.4f}")
    
    except Exception as e:
        print(f"Error in text classification: {e}")
        print("Make sure you have the required models and dependencies installed.")

def named_entity_recognition_example():
    """
    Demonstrate named entity recognition with a pre-trained model
    """
    print("\n" + "=" * 50)
    print("NAMED ENTITY RECOGNITION EXAMPLE")
    print("=" * 50)
    
    try:
        # Load pre-trained model and tokenizer for NER
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Example text
        text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
        
        print(f"\nModel: {model_name}")
        print(f"Task: Named Entity Recognition")
        print(f"Text: \"{text}\"")
        
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert token predictions to word predictions
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_predictions = [model.config.id2label[p.item()] for p in predictions[0]]
        
        # Merge WordPiece tokens and align with predicted entities
        word_ids = np.full(len(tokens), -1)
        current_word = -1
        
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                word_ids[i] = current_word
            else:
                current_word += 1
                word_ids[i] = current_word
        
        # Extract entities (simplistic approach for demonstration)
        entities = []
        current_entity = {"text": "", "label": ""}
        
        for i, (token, pred) in enumerate(zip(tokens, token_predictions)):
            if pred.startswith("B-"):  # Beginning of entity
                if current_entity["text"]:
                    entities.append(current_entity)
                current_entity = {"text": token.replace("##", ""), "label": pred[2:]}
            elif pred.startswith("I-") and current_entity["text"]:  # Inside entity
                # Make sure token is decontextualized if it's a WordPiece token
                current_entity["text"] += token.replace("##", "")
            elif not pred.startswith("I-") and current_entity["text"]:  # End of entity
                entities.append(current_entity)
                current_entity = {"text": "", "label": ""}
        
        if current_entity["text"]:  # Don't forget the last entity
            entities.append(current_entity)
        
        # Clean up special tokens
        entities = [e for e in entities if e["text"] not in ("[CLS]", "[SEP]")]
        
        # Print results
        print("\nDetected Entities:")
        for i, entity in enumerate(entities):
            print(f"{i+1}. {entity['text']} - {entity['label']}")
    
    except Exception as e:
        print(f"Error in named entity recognition: {e}")
        print("Make sure you have the required models and dependencies installed.")

def question_answering_example():
    """
    Demonstrate question answering with a pre-trained model
    """
    print("\n" + "=" * 50)
    print("QUESTION ANSWERING EXAMPLE")
    print("=" * 50)
    
    try:
        # Load pre-trained model and tokenizer for QA
        model_name = "distilbert-base-cased-distilled-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Example context and questions
        context = """
        The Transformer architecture was introduced in the paper "Attention is All You Need" 
        by researchers at Google in 2017. Unlike previous sequence-to-sequence models that used 
        recurrent neural networks, Transformers rely entirely on attention mechanisms, which allow 
        them to process input sequences in parallel rather than sequentially. This parallelization 
        enables more efficient training on modern hardware like GPUs and TPUs. The original 
        Transformer model has been adapted into many variants, including BERT, GPT, and T5, 
        which have achieved state-of-the-art results on a wide range of natural language processing tasks.
        """
        
        questions = [
            "When was the Transformer architecture introduced?",
            "Who introduced the Transformer architecture?",
            "What is the main difference between Transformers and previous models?",
            "What are some variants of the Transformer model?"
        ]
        
        print(f"\nModel: {model_name}")
        print(f"Task: Extractive Question Answering")
        print(f"Context: \"{context.strip()}\"\n")
        
        for i, question in enumerate(questions):
            # Tokenize input
            inputs = tokenizer(
                question, 
                context, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_offsets_mapping=True
            )
            offset_mapping = inputs.pop("offset_mapping")
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                # Get the most likely beginning and end of answer
                start_idx = torch.argmax(start_logits)
                end_idx = torch.argmax(end_logits)
                
                # Convert to Python integers
                start_idx = start_idx.item()
                end_idx = end_idx.item()
            
            # Get the answer tokens
            input_ids = inputs["input_ids"][0]
            answer_tokens = input_ids[start_idx:end_idx+1]
            answer = tokenizer.decode(answer_tokens)
            
            # Calculate confidence score (simplified)
            start_score = torch.nn.functional.softmax(start_logits, dim=1)[0][start_idx].item()
            end_score = torch.nn.functional.softmax(end_logits, dim=1)[0][end_idx].item()
            confidence = (start_score + end_score) / 2
            
            # Print results
            print(f"Question {i+1}: \"{question}\"")
            print(f"Answer: \"{answer}\"")
            print(f"Confidence: {confidence:.4f}\n")
    
    except Exception as e:
        print(f"Error in question answering: {e}")
        print("Make sure you have the required models and dependencies installed.")

def masked_language_modeling_example():
    """
    Demonstrate masked language modeling with a pre-trained model
    """
    print("\n" + "=" * 50)
    print("MASKED LANGUAGE MODELING EXAMPLE")
    print("=" * 50)
    
    try:
        # Load pre-trained model and tokenizer for masked LM
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Example masked sentences
        masked_texts = [
            "The capital of France is [MASK].",
            "Machine learning models learn from [MASK] to make predictions.",
            "Transformers use [MASK] mechanisms instead of recurrence.",
            "PyTorch and TensorFlow are popular [MASK] learning frameworks."
        ]
        
        print(f"\nModel: {model_name}")
        print(f"Task: Masked Language Modeling (Fill in the blank)")
        
        for i, text in enumerate(masked_texts):
            # Replace [MASK] with the actual mask token used by the model
            text = text.replace("[MASK]", tokenizer.mask_token)
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Find the position of the mask token
            mask_idx = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits
            
            # Get top predictions at the mask position
            mask_predictions = predictions[0, mask_idx]
            top_tokens = torch.topk(mask_predictions, k=5, dim=1).indices[0]
            
            # Convert token IDs to words
            top_words = [tokenizer.decode([token_id]) for token_id in top_tokens]
            
            # Print results
            print(f"\nMasked text {i+1}: \"{text}\"")
            print("Top 5 predictions:")
            for j, word in enumerate(top_words):
                word = word.strip()
                score = torch.nn.functional.softmax(mask_predictions, dim=1)[0, top_tokens[j]].item()
                print(f"  {j+1}. {word} (score: {score:.4f})")
    
    except Exception as e:
        print(f"Error in masked language modeling: {e}")
        print("Make sure you have the required models and dependencies installed.")

def text_generation_example():
    """
    Demonstrate text generation with a pre-trained model
    """
    print("\n" + "=" * 50)
    print("TEXT GENERATION EXAMPLE")
    print("=" * 50)
    
    try:
        # Load pre-trained model and tokenizer for text generation
        # Using a small model for demonstration purposes
        model_name = "distilgpt2"  # Much smaller than full GPT-2/GPT-3
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Example prompts
        prompts = [
            "The future of artificial intelligence is",
            "In a world where robots have emotions,",
            "The most important scientific discovery of the 21st century was"
        ]
        
        print(f"\nModel: {model_name}")
        print(f"Task: Text Generation (Causal Language Modeling)")
        
        for i, prompt in enumerate(prompts):
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate text
            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Print results
            print(f"\nPrompt {i+1}: \"{prompt}\"")
            print(f"Generated text: \"{generated_text}\"")
    
    except Exception as e:
        print(f"Error in text generation: {e}")
        print("Make sure you have the required models and dependencies installed.")

if __name__ == "__main__":
    introduction_to_model_inference()
    text_classification_example()
    named_entity_recognition_example()
    question_answering_example()
    masked_language_modeling_example()
    text_generation_example()
