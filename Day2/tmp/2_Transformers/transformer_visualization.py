"""
Transformer Visualization
Building Blocks of Generative AI Course - Day 2

This module provides visualization tools for exploring and understanding
the various components of the Transformer architecture from the
"Attention Is All You Need" paper (Vaswani et al., 2017).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformer_from_scratch_solution import (
    positional_encoding,
    scaled_dot_product_attention,
    MultiHeadAttention,
    EncoderLayer,
    DecoderLayer
)

# Create a custom colormap for attention visualizations
attention_cmap = LinearSegmentedColormap.from_list(
    'attention_cmap', ['#f7fbff', '#6baed6', '#08519c']
)

def visualize_positional_encoding(max_position=100, d_model=512):
    """
    Visualize the positional encoding used in transformers.
    
    Args:
        max_position: Maximum position to encode
        d_model: Model dimension
    """
    # Generate positional encoding
    pos_encoding = positional_encoding(max_position, d_model).numpy()[0]
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    
    # Add labels and title
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding Visualization')
    plt.colorbar(label='Value')
    
    # Add gridlines to show sin/cos pattern
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show dimension values
    dim_labels = ['sin', 'cos'] * (d_model // 2)
    plt.xticks(np.arange(0, d_model, d_model//10))
    
    # Add annotations
    plt.annotate('Sine function (even indices)', 
                xy=(d_model//4, max_position//2),
                xytext=(d_model//4, max_position//2 - 15),
                arrowprops=dict(arrowstyle="->", color='black'))
    
    plt.annotate('Cosine function (odd indices)', 
                xy=(d_model//4 + 1, max_position//2),
                xytext=(d_model//4 + 10, max_position//2 - 15),
                arrowprops=dict(arrowstyle="->", color='black'))
    
    plt.tight_layout()
    plt.show()
    
    # Also show a smaller subset for clearer visualization
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pos_encoding[:20, :20], cmap='RdBu')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding (Zoomed In)')
    plt.colorbar(label='Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_scaled_dot_product_attention(seq_len_q=4, seq_len_k=6, depth=8):
    """
    Visualize the scaled dot-product attention mechanism.
    
    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        depth: Depth of the query and key vectors
    """
    # Create sample queries, keys, and values
    np.random.seed(42)
    q = tf.constant(np.random.randn(1, seq_len_q, depth), dtype=tf.float32)
    k = tf.constant(np.random.randn(1, seq_len_k, depth), dtype=tf.float32)
    v = tf.constant(np.random.randn(1, seq_len_k, depth), dtype=tf.float32)
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(q, k, v)
    
    # Visualize the attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0], cmap=attention_cmap)
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.title('Attention Weights')
    plt.colorbar(label='Weight')
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(np.arange(seq_len_k))
    plt.yticks(np.arange(seq_len_q))
    
    # Add text annotations for each weight
    for i in range(seq_len_q):
        for j in range(seq_len_k):
            plt.text(j, i, f'{attention_weights[0, i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if attention_weights[0, i, j] > 0.3 else 'black')
    
    plt.tight_layout()
    plt.show()
    
    # Also visualize the computation flow with a diagram
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(q[0], cmap='Blues')
    plt.title('Query Matrix (Q)')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(k[0], cmap='Greens')
    plt.title('Key Matrix (K)')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(np.matmul(q[0], k[0].transpose()), cmap='Oranges')
    plt.title('QK^T (before scaling)')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(attention_weights[0], cmap=attention_cmap)
    plt.title('Attention Weights (after softmax)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Show the weighted values
    plt.figure(figsize=(10, 6))
    weighted_values = np.zeros_like(v[0])
    for i in range(seq_len_q):
        for j in range(seq_len_k):
            weighted_values += attention_weights[0, i, j] * v[0, j]
    
    plt.subplot(1, 2, 1)
    plt.imshow(v[0], cmap='Purples')
    plt.title('Value Matrix (V)')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(output[0], cmap='Reds')
    plt.title('Output (weighted sum of values)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def visualize_multi_head_attention(d_model=512, num_heads=8, seq_len=10):
    """
    Visualize the multi-head attention mechanism.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
    """
    # Create a multi-head attention layer
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Create sample input
    np.random.seed(42)
    x = tf.constant(np.random.randn(1, seq_len, d_model), dtype=tf.float32)
    
    # Compute multi-head attention
    output, attention_weights = mha(x, x, x)
    
    # Visualize the attention weights for each head
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_heads):
        ax = axes[i]
        ax.imshow(attention_weights[0, i], cmap=attention_cmap)
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key positions')
        ax.set_ylabel('Query positions')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Attention Weights for Each Head', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Visualize the input and output
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(x[0], cmap='Blues')
    plt.title('Input')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(output[0], cmap='Reds')
    plt.title('Output')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def visualize_transformer_architecture():
    """
    Visualize the overall transformer architecture using a schematic diagram.
    """
    # Create a figure for the schematic diagram
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Turn off axis
    ax.axis('off')
    
    # Define colors
    encoder_color = '#4292c6'  # Blue
    decoder_color = '#41ab5d'  # Green
    attention_color = '#feb24c'  # Orange
    ffn_color = '#756bb1'  # Purple
    
    # Draw the encoder stack
    ax.text(0.5, 0.95, 'Transformer Architecture', ha='center', fontsize=16, fontweight='bold')
    
    # Encoder
    ax.text(0.25, 0.9, 'Encoder', ha='center', fontsize=14, fontweight='bold')
    encoder_rect = plt.Rectangle((0.1, 0.55), 0.3, 0.3, fill=True, alpha=0.2, color=encoder_color)
    ax.add_patch(encoder_rect)
    
    # Encoder components
    ax.text(0.25, 0.82, 'Multi-Head\nAttention', ha='center', fontsize=12)
    attention_rect = plt.Rectangle((0.15, 0.75), 0.2, 0.1, fill=True, alpha=0.3, color=attention_color)
    ax.add_patch(attention_rect)
    
    ax.text(0.25, 0.7, 'Feed-Forward\nNetwork', ha='center', fontsize=12)
    ffn_rect = plt.Rectangle((0.15, 0.63), 0.2, 0.1, fill=True, alpha=0.3, color=ffn_color)
    ax.add_patch(ffn_rect)
    
    ax.text(0.25, 0.58, '×N', ha='center', fontsize=14)
    
    # Decoder
    ax.text(0.75, 0.9, 'Decoder', ha='center', fontsize=14, fontweight='bold')
    decoder_rect = plt.Rectangle((0.6, 0.45), 0.3, 0.4, fill=True, alpha=0.2, color=decoder_color)
    ax.add_patch(decoder_rect)
    
    # Decoder components
    ax.text(0.75, 0.82, 'Masked\nMulti-Head\nAttention', ha='center', fontsize=12)
    masked_attention_rect = plt.Rectangle((0.65, 0.75), 0.2, 0.1, fill=True, alpha=0.3, color=attention_color)
    ax.add_patch(masked_attention_rect)
    
    ax.text(0.75, 0.7, 'Multi-Head\nAttention', ha='center', fontsize=12)
    cross_attention_rect = plt.Rectangle((0.65, 0.63), 0.2, 0.1, fill=True, alpha=0.3, color=attention_color)
    ax.add_patch(cross_attention_rect)
    
    ax.text(0.75, 0.58, 'Feed-Forward\nNetwork', ha='center', fontsize=12)
    ffn_rect2 = plt.Rectangle((0.65, 0.51), 0.2, 0.1, fill=True, alpha=0.3, color=ffn_color)
    ax.add_patch(ffn_rect2)
    
    ax.text(0.75, 0.48, '×N', ha='center', fontsize=14)
    
    # Inputs and outputs
    ax.text(0.25, 0.4, 'Input\nEmbedding', ha='center', fontsize=12)
    input_rect = plt.Rectangle((0.15, 0.33), 0.2, 0.1, fill=True, alpha=0.3, color='#969696')
    ax.add_patch(input_rect)
    
    ax.text(0.75, 0.4, 'Output\nEmbedding', ha='center', fontsize=12)
    output_rect = plt.Rectangle((0.65, 0.33), 0.2, 0.1, fill=True, alpha=0.3, color='#969696')
    ax.add_patch(output_rect)
    
    ax.text(0.25, 0.28, 'Positional\nEncoding', ha='center', fontsize=12)
    pos_enc_rect1 = plt.Rectangle((0.15, 0.21), 0.2, 0.1, fill=True, alpha=0.3, color='#969696')
    ax.add_patch(pos_enc_rect1)
    
    ax.text(0.75, 0.28, 'Positional\nEncoding', ha='center', fontsize=12)
    pos_enc_rect2 = plt.Rectangle((0.65, 0.21), 0.2, 0.1, fill=True, alpha=0.3, color='#969696')
    ax.add_patch(pos_enc_rect2)
    
    ax.text(0.25, 0.15, 'Input', ha='center', fontsize=12)
    ax.text(0.75, 0.15, 'Output', ha='center', fontsize=12)
    
    # Arrows
    ax.arrow(0.25, 0.17, 0, 0.03, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.25, 0.32, 0, 0.03, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.25, 0.44, 0, 0.1, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    ax.arrow(0.75, 0.17, 0, 0.03, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.75, 0.32, 0, 0.03, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.75, 0.44, 0, 0.06, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Connection between encoder and decoder
    ax.arrow(0.4, 0.67, 0.2, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    plt.tight_layout()
    plt.show()

def visualize_attention_mechanism():
    """
    Visualize the attention mechanism conceptually.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create input sequence representation
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n_words = len(words)
    
    # Draw word boxes
    word_boxes = []
    word_centers = []
    for i, word in enumerate(words):
        x = 0.1 + i * 0.15
        word_centers.append(x)
        rect = plt.Rectangle((x-0.06, 0.8), 0.12, 0.1, fill=True, alpha=0.3, color='#4292c6')
        ax.add_patch(rect)
        ax.text(x, 0.85, word, ha='center', va='center', fontsize=12)
        word_boxes.append(rect)
    
    # Draw attention from "sat" to other words
    query_idx = 2  # "sat"
    x_query = word_centers[query_idx]
    y_query = 0.8
    
    # Draw the query word with different color
    rect = plt.Rectangle((x_query-0.06, y_query), 0.12, 0.1, fill=True, alpha=0.5, color='#ff7f00')
    ax.add_patch(rect)
    ax.text(x_query, y_query+0.05, words[query_idx], ha='center', va='center', fontsize=12)
    
    # Hypothetical attention weights (highest for "cat")
    attention_weights = [0.1, 0.5, 0.0, 0.2, 0.1, 0.1]
    
    # Draw attention arrows
    for i, (x_key, weight) in enumerate(zip(word_centers, attention_weights)):
        if i != query_idx:  # Don't draw self-attention
            # Draw arrow from query to key
            ax.annotate('', xy=(x_key, 0.8), xytext=(x_query, 0.8),
                       arrowprops=dict(arrowstyle='->', lw=1+weight*5, color='#d94801', alpha=0.7),
                       horizontalalignment='center')
            
            # Add weight text
            mid_x = (x_query + x_key) / 2
            mid_y = 0.82 if i < query_idx else 0.78
            ax.text(mid_x, mid_y, f'{weight:.1f}', ha='center', va='center', fontsize=10)
    
    # Add labels and descriptions
    ax.text(0.5, 0.95, 'Attention Mechanism Visualization', ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.7, 'The attention mechanism allows each word to "focus" on other relevant words in the sequence.\n'
                    'Line thickness represents attention weight - "sat" is paying most attention to "cat".',
           ha='center', fontsize=12)
    
    # Add equations
    eq_y = 0.5
    ax.text(0.1, eq_y, 'Attention Calculation:', ha='left', fontsize=14, fontweight='bold')
    ax.text(0.1, eq_y-0.07, '1. For each word, we calculate Query (Q), Key (K), and Value (V) vectors', ha='left', fontsize=12)
    ax.text(0.1, eq_y-0.14, '2. Attention Scores = Q × K^T', ha='left', fontsize=12)
    ax.text(0.1, eq_y-0.21, '3. Attention Weights = softmax(Attention Scores / √d)', ha='left', fontsize=12)
    ax.text(0.1, eq_y-0.28, '4. Output = Attention Weights × V', ha='left', fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Transformer Visualizations...")
    
    print("\n1. Visualizing Positional Encoding")
    visualize_positional_encoding(max_position=50, d_model=128)
    
    print("\n2. Visualizing Scaled Dot-Product Attention")
    visualize_scaled_dot_product_attention()
    
    print("\n3. Visualizing Multi-Head Attention")
    visualize_multi_head_attention(d_model=64, num_heads=4, seq_len=8)
    
    print("\n4. Visualizing Transformer Architecture")
    visualize_transformer_architecture()
    
    print("\n5. Visualizing Attention Mechanism Conceptually")
    visualize_attention_mechanism()
