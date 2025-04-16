"""
Hugging Face Datasets Library
----------------------------
This file demonstrates how to work with the Hugging Face Datasets library
for loading, processing, and managing datasets for NLP and ML tasks.
"""

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def introduction_to_datasets():
    """
    Introduction to the Hugging Face Datasets library
    """
    print("=" * 50)
    print("HUGGING FACE DATASETS LIBRARY")
    print("=" * 50)
    
    print("\nKey Features:")
    print("1. Unified API for diverse datasets")
    print("2. Memory efficiency through memory mapping")
    print("3. Dataset streaming for large datasets")
    print("4. Consistent preprocessing capabilities")
    print("5. Integration with Hugging Face model training")
    print("6. Versioning and dataset documentation")

def load_dataset_examples():
    """
    Demonstrate different ways to load datasets from the Hugging Face Hub
    """
    print("\n" + "=" * 50)
    print("LOADING DATASETS EXAMPLES")
    print("=" * 50)
    
    # Example 1: Load a popular dataset from the Hub
    print("\n1. Loading a dataset from the Hub:")
    try:
        # Load the IMDB movie reviews dataset
        imdb_dataset = load_dataset("imdb", split="train")
        print(f"IMDB dataset loaded: {imdb_dataset}")
        print(f"Number of examples: {len(imdb_dataset)}")
        print(f"Features: {imdb_dataset.features}")
        print(f"First example: {imdb_dataset[0]}")
    except Exception as e:
        print(f"Error loading IMDB dataset: {e}")
    
    # Example 2: Load only specific splits
    print("\n2. Loading specific splits:")
    try:
        # Load only train split of the GLUE SST2 dataset
        sst2_train = load_dataset("glue", "sst2", split="train")
        print(f"SST2 train split loaded: {sst2_train}")
        print(f"Number of training examples: {len(sst2_train)}")
        
        # Load multiple splits as a dictionary
        glue_dataset = load_dataset("glue", "sst2", split=["train", "validation"])
        print(f"Splits loaded: {list(glue_dataset.keys())}")
        print(f"Train examples: {len(glue_dataset['train'])}")
        print(f"Validation examples: {len(glue_dataset['validation'])}")
    except Exception as e:
        print(f"Error loading GLUE dataset: {e}")
    
    # Example 3: Loading a subset with streaming
    print("\n3. Streaming a large dataset:")
    try:
        # Stream a large dataset like C4
        c4_dataset = load_dataset("c4", "en", split="train", streaming=True)
        print(f"C4 dataset initialized for streaming: {c4_dataset}")
        
        # Take just the first 2 examples to show streaming
        sample = list(c4_dataset.take(2))
        print(f"Sample from C4 (first 2 examples):")
        for i, example in enumerate(sample):
            print(f"Example {i+1} text (truncated): {example['text'][:100]}...")
    except Exception as e:
        print(f"Error streaming C4 dataset: {e}")

def create_custom_dataset():
    """
    Demonstrate how to create custom datasets from different data sources
    """
    print("\n" + "=" * 50)
    print("CREATING CUSTOM DATASETS")
    print("=" * 50)
    
    # Example 1: Create dataset from in-memory data
    print("\n1. Creating a dataset from dictionaries:")
    dict_data = {
        "text": ["I love this movie", "This film was terrible", "Amazing performance!"],
        "label": [1, 0, 1]
    }
    dict_dataset = Dataset.from_dict(dict_data)
    print(f"Dataset from dictionaries: {dict_dataset}")
    print(f"Examples: {dict_dataset}")
    
    # Example 2: Create dataset from pandas DataFrame
    print("\n2. Creating a dataset from pandas DataFrame:")
    df = pd.DataFrame({
        "text": ["The weather is nice", "It's raining again", "Perfect day for a walk"],
        "sentiment": ["positive", "negative", "positive"],
        "temperature": [25, 15, 22]
    })
    df_dataset = Dataset.from_pandas(df)
    print(f"Dataset from DataFrame: {df_dataset}")
    print(f"Features: {df_dataset.features}")
    print(f"First example: {df_dataset[0]}")
    
    # Example 3: Creating a dataset from a CSV file (simulating with StringIO)
    print("\n3. Creating a dataset from CSV:")
    from io import StringIO
    
    # Create a mock CSV file in memory
    csv_data = StringIO("""title,content,category
    Breaking News,Major event happened today,news
    Product Review,This product is fantastic,review
    Sports Update,Team wins championship,sports
    """)
    
    # In a real scenario, you would use:
    # csv_dataset = load_dataset("csv", data_files="path/to/your/file.csv")
    
    # For demo purposes:
    df = pd.read_csv(csv_data)
    csv_dataset = Dataset.from_pandas(df)
    print(f"Dataset from CSV: {csv_dataset}")
    print(f"Examples: {csv_dataset}")

def dataset_transformations():
    """
    Demonstrate common dataset transformation operations
    """
    print("\n" + "=" * 50)
    print("DATASET TRANSFORMATIONS")
    print("=" * 50)
    
    # Create a sample dataset
    data = {
        "text": [
            "I absolutely loved this movie!",
            "This film was terrible, I hated it.",
            "It was just okay, nothing special.",
            "One of the best performances I've seen.",
            "Waste of time and money."
        ],
        "rating": [5, 1, 3, 5, 1]
    }
    dataset = Dataset.from_dict(data)
    print(f"Original dataset: {dataset}")
    
    # Example 1: Map function to add new column
    print("\n1. Using map() to add a new column:")
    
    def add_text_length(example):
        example["text_length"] = len(example["text"])
        return example
    
    dataset_with_length = dataset.map(add_text_length)
    print(f"Dataset with text length: {dataset_with_length}")
    print(f"Features: {dataset_with_length.features}")
    print(f"First example: {dataset_with_length[0]}")
    
    # Example 2: Filter examples by condition
    print("\n2. Using filter() to keep only positive reviews:")
    
    def is_positive(example):
        return example["rating"] >= 4
    
    positive_reviews = dataset.filter(is_positive)
    print(f"Positive reviews dataset: {positive_reviews}")
    print(f"Number of examples: {len(positive_reviews)}")
    print(f"Examples: {positive_reviews}")
    
    # Example 3: Sort by a field
    print("\n3. Using sort() to order by rating:")
    sorted_dataset = dataset.sort("rating")
    print(f"Sorted dataset: {sorted_dataset}")
    print(f"Examples (sorted by rating):")
    for example in sorted_dataset:
        print(f"  Rating: {example['rating']} - {example['text'][:30]}...")
    
    # Example 4: Select specific columns
    print("\n4. Using select() to keep only specific columns:")
    text_only = dataset.select(["text"])
    print(f"Text-only dataset: {text_only}")
    print(f"Features: {text_only.features}")
    print(f"First example: {text_only[0]}")

def tokenize_dataset_example():
    """
    Demonstrate how to tokenize a dataset for model training
    """
    print("\n" + "=" * 50)
    print("TOKENIZING DATASETS FOR MODEL TRAINING")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer
        
        # Create a simple dataset
        texts = [
            "Hugging Face is a company that develops tools for building applications using machine learning.",
            "The Transformers library provides thousands of pretrained models.",
            "Training a model from scratch can be expensive and time-consuming.",
            "Transfer learning allows us to reuse knowledge from pre-trained models."
        ]
        labels = [0, 1, 2, 1]  # Some example labels
        
        dataset = Dataset.from_dict({"text": texts, "label": labels})
        print(f"Dataset for tokenization: {dataset}")
        
        # Load a tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        
        # Apply tokenization to the entire dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        print(f"\nTokenized dataset: {tokenized_dataset}")
        print(f"New features: {tokenized_dataset.features}")
        
        # Show how it transformed the first example
        print(f"\nOriginal text: {dataset[0]['text']}")
        print(f"Input IDs: {tokenized_dataset[0]['input_ids'][:10]}... (truncated)")
        print(f"Attention mask: {tokenized_dataset[0]['attention_mask'][:10]}... (truncated)")
        
        # Format dataset for model training
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset.set_format("torch")
        
        print(f"\nDataset prepared for training: {tokenized_dataset}")
        print(f"First example format: {tokenized_dataset[0]}")
        
    except ImportError:
        print("Error: transformers library not installed")
        print("Install with: pip install transformers")

def dataset_exploration_visualization():
    """
    Demonstrate techniques for exploring and visualizing datasets
    """
    print("\n" + "=" * 50)
    print("DATASET EXPLORATION AND VISUALIZATION")
    print("=" * 50)
    
    # Create a sample dataset (movie reviews)
    data = {
        "review": [
            "This movie was fantastic!",
            "I fell asleep during the film.",
            "The acting was superb but the plot was weak.",
            "What a waste of time and money.",
            "Easily one of the best films of the year.",
            "The visual effects were impressive.",
            "I can't recommend this enough!",
            "Very disappointing experience.",
            "A masterpiece of modern cinema.",
            "Mediocre at best."
        ],
        "genre": [
            "drama", "comedy", "drama", "action", 
            "sci-fi", "sci-fi", "comedy", "horror", 
            "drama", "action"
        ],
        "rating": [
            4.5, 2.0, 3.5, 1.0, 5.0, 
            4.0, 4.8, 1.5, 5.0, 3.0
        ],
        "length_minutes": [
            120, 95, 130, 110, 160, 
            145, 105, 90, 170, 100
        ]
    }
    dataset = Dataset.from_dict(data)
    print(f"Sample dataset: {dataset}")
    
    # Basic statistics
    print("\n1. Basic statistics:")
    ratings = dataset["rating"]
    lengths = dataset["length_minutes"]
    
    print(f"Number of reviews: {len(dataset)}")
    print(f"Average rating: {sum(ratings) / len(ratings):.2f}")
    print(f"Average movie length: {sum(lengths) / len(lengths):.2f} minutes")
    
    # Genre distribution
    print("\n2. Genre distribution:")
    genre_counts = Counter(dataset["genre"])
    for genre, count in genre_counts.items():
        print(f"  {genre}: {count} movies")
    
    # Rating distribution visualization
    print("\n3. Rating distribution (plotting code):")
    print("""
    plt.figure(figsize=(10, 6))
    plt.hist(dataset["rating"], bins=10, edgecolor='black')
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.savefig("rating_distribution.png")
    """)
    
    # Advanced example - rating by genre
    print("\n4. Average rating by genre:")
    genre_ratings = {}
    for genre in set(dataset["genre"]):
        genre_indices = [i for i, g in enumerate(dataset["genre"]) if g == genre]
        genre_avg_rating = sum(dataset["rating"][i] for i in genre_indices) / len(genre_indices)
        genre_ratings[genre] = genre_avg_rating
        print(f"  {genre}: {genre_avg_rating:.2f}")
    
    print("\n5. Length vs. Rating visualization (plotting code):")
    print("""
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset["length_minutes"], dataset["rating"])
    plt.title("Movie Length vs. Rating")
    plt.xlabel("Length (minutes)")
    plt.ylabel("Rating")
    plt.grid(alpha=0.3)
    plt.savefig("length_vs_rating.png")
    """)

if __name__ == "__main__":
    introduction_to_datasets()
    load_dataset_examples()
    create_custom_dataset()
    dataset_transformations()
    tokenize_dataset_example()
    dataset_exploration_visualization()
