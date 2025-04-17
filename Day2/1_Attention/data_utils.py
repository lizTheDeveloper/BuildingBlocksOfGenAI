"""
Data Utilities for Attention Exercise
Building Blocks of Generative AI Course - Day 2

Utility functions for preparing data for the attention mechanism exercise.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def preprocess_text(texts, tokenizer=None, max_sequence_length=None):
    """
    Preprocess text data using a tokenizer and pad sequences.
    
    Args:
        texts: List of text strings to preprocess
        tokenizer: Keras Tokenizer (will be created if None)
        max_sequence_length: Maximum sequence length for padding
        
    Returns:
        padded_sequences: Tokenized and padded sequences
        tokenizer: The tokenizer used
        max_sequence_length: The maximum sequence length used
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Determine max sequence length if not provided
    if max_sequence_length is None:
        max_sequence_length = max(len(seq) for seq in sequences)
    
    # Pad sequences
    padded_sequences = keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_sequence_length, padding='post')
    
    return padded_sequences, tokenizer, max_sequence_length

def load_sample_translation_data():
    """
    Load a small sample of English to French translation data.
    
    Returns:
        dict: Dictionary containing processed data and metadata
    """
    # Sample translation data (English to French)
    # This is a very small dataset for demonstration purposes
    english_texts = [
        "hello",
        "thank you",
        "how are you",
        "goodbye",
        "my name is",
        "what is your name",
        "nice to meet you",
        "where are you from",
        "I am from the United States",
        "do you speak English"
    ]

    french_texts = [
        "bonjour",
        "merci",
        "comment allez-vous",
        "au revoir",
        "je m'appelle",
        "comment vous appelez-vous",
        "enchanté",
        "d'où venez-vous",
        "je viens des États-Unis",
        "parlez-vous anglais"
    ]

    # Preprocess the data
    eng_sequences, eng_tokenizer, eng_max_length = preprocess_text(english_texts)
    fr_sequences, fr_tokenizer, fr_max_length = preprocess_text(french_texts)

    # Add start and end tokens to French sequences
    fr_texts_with_tokens = ['<start> ' + text + ' <end>' for text in french_texts]
    fr_sequences_with_tokens, fr_tokenizer_with_tokens, fr_max_length_with_tokens = preprocess_text(fr_texts_with_tokens)

    # Get vocabulary sizes
    eng_vocab_size = len(eng_tokenizer.word_index) + 1  # +1 for padding token (0)
    fr_vocab_size = len(fr_tokenizer_with_tokens.word_index) + 1
    
    # Return processed data
    return {
        'eng_texts': english_texts,
        'fr_texts': french_texts,
        'eng_sequences': eng_sequences,
        'fr_sequences': fr_sequences_with_tokens,
        'eng_tokenizer': eng_tokenizer,
        'fr_tokenizer': fr_tokenizer_with_tokens,
        'eng_max_length': eng_max_length,
        'fr_max_length': fr_max_length_with_tokens,
        'eng_vocab_size': eng_vocab_size,
        'fr_vocab_size': fr_vocab_size
    }

def create_dataset(eng_sequences, fr_sequences, batch_size=64, shuffle=True):
    """
    Create a TensorFlow dataset from sequence pairs.
    
    Args:
        eng_sequences: Preprocessed English sequences
        fr_sequences: Preprocessed French sequences
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        
    Returns:
        tf.data.Dataset: Dataset of sequence pairs
    """
    dataset = tf.data.Dataset.from_tensor_slices((eng_sequences, fr_sequences))
    
    if shuffle:
        dataset = dataset.shuffle(len(eng_sequences))
    
    dataset = dataset.batch(batch_size)
    
    return dataset
