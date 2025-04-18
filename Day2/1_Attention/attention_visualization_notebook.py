"""
Attention Visualization for Scaled Dot-Product Attention
Building Blocks of Generative AI Course - Day 2

This script demonstrates how to visualize the attention weights
from your implementation of the scaled dot-product attention mechanism.
It assumes your attention implementation is already defined.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def visualize_attention_weights():
    """
    Create a simple example and visualize the attention weights
    produced by your attention implementation.
    """
    # Create a simple example with small vectors
    batch_size = 1                  # Just one example for simplicity
    query_sequence_length = 3       # We have 3 queries (imagine 3 words in an output sentence)
    key_value_sequence_length = 4   # We have 4 keys/values (imagine 4 words in an input sentence)
    feature_dimension = 5           # Each vector has 5 features
    
    # Create sample queries, keys, and values
    query = tf.random.normal((batch_size, query_sequence_length, feature_dimension))
    key = tf.random.normal((batch_size, key_value_sequence_length, feature_dimension))
    value = tf.random.normal((batch_size, key_value_sequence_length, feature_dimension))
    
    # Create an instance of your attention implementation
    # This assumes ScaledDotProductAttention is already defined
    attention = ScaledDotProductAttention()
    
    # Apply your attention mechanism
    output, attention_weights = attention(query, key, value)
    
    # Visualize the attention weights
    plt.figure(figsize=(10, 6))
    plt.imshow(attention_weights[0], cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Keys (Input Words)')
    plt.ylabel('Queries (Output Words)')
    plt.title('Attention Weights: How Much Each Query Focuses on Each Key')
    
    # Add text annotations
    for i in range(query_sequence_length):
        for j in range(key_value_sequence_length):
            plt.text(j, i, f'{attention_weights[0, i, j].numpy():.2f}',
                     ha="center", va="center", 
                     color="white" if attention_weights[0, i, j] < 0.5 else "black")
    
    # Use more intuitive labels
    plt.xticks(range(key_value_sequence_length), [f'Input {i+1}' for i in range(key_value_sequence_length)])
    plt.yticks(range(query_sequence_length), [f'Output {i+1}' for i in range(query_sequence_length)])
    
    plt.tight_layout()
    plt.show()
    
    print("Attention visualization complete!")
    print("If you see a heatmap with attention weights, your implementation is working!")
    print("The weights should sum to 1.0 across each row (for each query).")
    
    # Verify that weights sum to 1 for each query (row)
    for i in range(query_sequence_length):
        weight_sum = tf.reduce_sum(attention_weights[0, i]).numpy()
        print(f"Sum of weights for Query {i+1}: {weight_sum:.6f}")
        if abs(weight_sum - 1.0) > 1e-5:
            print("⚠️ Warning: Weights don't sum to 1.0. Check your softmax implementation.")
    
    return output, attention_weights

def visualize_translation_example():
    """
    Demonstrate attention for a simple English-French translation example.
    """
    # Define a simple English sentence and its French translation
    english_sentence = ["I", "am", "going", "to", "the", "store"]
    french_sentence = ["Je", "vais", "au", "magasin"]
    
    # For simplicity, we'll manually create word embeddings
    # In a real system, these would be learned by the model
    embedding_dimension = 8
    
    # Create random but fixed embeddings for our words
    np.random.seed(123)  # For reproducibility
    english_embeddings = {word: np.random.randn(embedding_dimension) for word in english_sentence}
    french_embeddings = {word: np.random.randn(embedding_dimension) for word in french_sentence}
    
    # Convert sentences to sequences of embeddings
    english_vectors = np.array([english_embeddings[word] for word in english_sentence])
    french_vectors = np.array([french_embeddings[word] for word in french_sentence])
    
    # Reshape for our attention function (add batch dimension)
    english_vectors = english_vectors.reshape(1, len(english_sentence), embedding_dimension)
    french_vectors = french_vectors.reshape(1, len(french_sentence), embedding_dimension)
    
    # Convert to TensorFlow tensors
    english_vectors_tf = tf.constant(english_vectors, dtype=tf.float32)
    french_vectors_tf = tf.constant(french_vectors, dtype=tf.float32)
    
    # Create an instance of your attention implementation
    attention = ScaledDotProductAttention()
    
    print("\n--- English to French Translation ---")
    # Apply attention (French words attending to English words)
    fr_output, fr_weights = attention(
        query=french_vectors_tf,     # Target language (what we're generating)
        key=english_vectors_tf,      # Source language (what we're translating from)
        value=english_vectors_tf     # Using same vectors as values for simplicity
    )
    
    # Visualize the translation attention weights
    plt.figure(figsize=(12, 8))
    plt.imshow(fr_weights[0], cmap='YlOrRd')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('English Words (Source)')
    plt.ylabel('French Words (Target)')
    plt.title('English→French Attention Weights: Word Alignment Visualization')
    
    # Add text annotations for the weights
    for i in range(len(french_sentence)):
        for j in range(len(english_sentence)):
            plt.text(j, i, f'{fr_weights[0, i, j].numpy():.2f}',
                     ha="center", va="center",
                     color="black" if fr_weights[0, i, j] < 0.15 else "white")
    
    # Use actual words as labels
    plt.xticks(range(len(english_sentence)), english_sentence, rotation=45)
    plt.yticks(range(len(french_sentence)), french_sentence)
    
    plt.tight_layout()
    plt.show()
    
    print("\n--- French to English Translation ---")
    # Apply attention in the reverse direction (English words attending to French words)
    en_output, en_weights = attention(
        query=english_vectors_tf,    # Target language (what we're generating)
        key=french_vectors_tf,       # Source language (what we're translating from)
        value=french_vectors_tf      # Using same vectors as values for simplicity
    )
    
    # Visualize the reverse translation attention weights
    plt.figure(figsize=(12, 8))
    plt.imshow(en_weights[0], cmap='PuBu')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('French Words (Source)')
    plt.ylabel('English Words (Target)')
    plt.title('French→English Attention Weights: Word Alignment Visualization')
    
    # Add text annotations for the weights
    for i in range(len(english_sentence)):
        for j in range(len(french_sentence)):
            plt.text(j, i, f'{en_weights[0, i, j].numpy():.2f}',
                     ha="center", va="center",
                     color="black" if en_weights[0, i, j] < 0.15 else "white")
    
    # Use actual words as labels
    plt.xticks(range(len(french_sentence)), french_sentence, rotation=45)
    plt.yticks(range(len(english_sentence)), english_sentence)
    
    plt.tight_layout()
    plt.show()
    
    print("Translation attention visualization complete!")
    print("In an ideal translation system, we would see:")
    print("- 'Je' attending most to 'I'")
    print("- 'vais' attending most to 'am going'")
    print("- 'au' attending most to 'to the'")
    print("- 'magasin' attending most to 'store'")
    print("And similar patterns in the reverse direction.")
    
    return (fr_output, fr_weights), (en_output, en_weights)

def visualize_multi_head_attention_example():
    """
    Demonstrate a multi-head attention example.
    This assumes you've implemented MultiHeadAttention as well.
    """
    try:
        # Create a simple example
        english_sentence = ["I", "am", "going", "to", "the", "store"]
        french_sentence = ["Je", "vais", "au", "magasin"]
        
        # Parameters for multi-head attention
        embedding_dimension = 8     # Must be divisible by num_heads
        num_heads = 2
        
        # Create random embeddings
        np.random.seed(456)
        english_embeddings = {word: np.random.randn(embedding_dimension) for word in english_sentence}
        french_embeddings = {word: np.random.randn(embedding_dimension) for word in french_sentence}
        
        # Convert to sequences
        english_vectors = np.array([english_embeddings[word] for word in english_sentence])
        french_vectors = np.array([french_embeddings[word] for word in french_sentence])
        
        # Reshape for attention (add batch dimension)
        english_vectors = english_vectors.reshape(1, len(english_sentence), embedding_dimension)
        french_vectors = french_vectors.reshape(1, len(french_sentence), embedding_dimension)
        
        # Convert to TensorFlow tensors
        english_vectors_tf = tf.constant(english_vectors, dtype=tf.float32)
        french_vectors_tf = tf.constant(french_vectors, dtype=tf.float32)
        
        # Create an instance of your multi-head attention implementation
        multi_head_attention = MultiHeadAttention(d_model=embedding_dimension, num_heads=num_heads)
        
        # Apply multi-head attention (French attending to English)
        mha_output, mha_weights_dict = multi_head_attention(
            v=english_vectors_tf,
            k=english_vectors_tf,
            q=french_vectors_tf
        )
        
        # Visualize attention weights for each head side by side
        plt.figure(figsize=(15, 6))
        
        # Plot each head
        for h in range(num_heads):
            plt.subplot(1, num_heads, h+1)
            head_weights = mha_weights_dict[f'head_{h+1}'].numpy()
            plt.imshow(head_weights[0], cmap='YlOrRd')
            plt.colorbar(label='Attention Weight')
            plt.xlabel('English Words (Source)')
            plt.ylabel('French Words (Target)')
            plt.title(f'Head {h+1} Attention Weights')
            plt.xticks(range(len(english_sentence)), english_sentence, rotation=45)
            plt.yticks(range(len(french_sentence)), french_sentence)
            
            # Add text annotations
            for i in range(len(french_sentence)):
                for j in range(len(english_sentence)):
                    plt.text(j, i, f'{head_weights[0, i, j]:.2f}',
                             ha="center", va="center",
                             color="black" if head_weights[0, i, j] < 0.15 else "white")
        
        plt.tight_layout()
        plt.show()
        
        print("Multi-head attention visualization complete!")
        print("Notice how different heads might focus on different patterns in the data.")
        print("For example:")
        print("- One head might focus on subject-object relationships")
        print("- Another head might focus on verb tense or prepositions")
        print("This is the power of multi-head attention: capturing different relationships simultaneously.")
        
        return mha_output, mha_weights_dict
    
    except NameError:
        print("MultiHeadAttention class not found. This is expected if you're only implementing ScaledDotProductAttention.")
        return None, None

# Main execution for the visualization
if __name__ == "__main__":
    print("Visualizing attention weights for your ScaledDotProductAttention implementation...")
    
    # Run the simple attention visualization
    output, attention_weights = visualize_attention_weights()
    
    # Run the translation example
    translation_results = visualize_translation_example()
    
    # Try the multi-head example (only works if MultiHeadAttention is implemented)
    try:
        mha_output, mha_weights = visualize_multi_head_attention_example()
    except Exception as e:
        print(f"Note: Multi-head attention visualization requires MultiHeadAttention class.")
        print(f"Error: {e}")
