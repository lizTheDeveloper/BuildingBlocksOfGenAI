"""
Transformer Exercise
Building Blocks of Generative AI Course - Day 2

This exercise guides students through implementing parts of the Transformer architecture
as described in "Attention Is All You Need" (Vaswani et al., 2017).

Students will fill in missing parts of the implementation to better understand
the key components of the Transformer model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def get_angles(pos, i, d_model):
    """
    Calculate the angles for the positional encoding.
    
    Args:
        pos: Position in the sequence
        i: Dimension index
        d_model: Model dimension
        
    Returns:
        angles: Angles for the positional encoding
    """
    # TODO: Implement the formula for positional encoding angles
    # Hint: The formula is angle_rates = 1 / (10000 ** (2i / d_model))
    
    pass  # Replace with your implementation

def positional_encoding(position, d_model):
    """
    Create positional encoding for transformer input.
    
    Args:
        position: Maximum sequence length
        d_model: Model dimension
        
    Returns:
        pos_encoding: Positional encoding of shape (1, position, d_model)
    """
    # TODO: Implement positional encoding
    # 1. Calculate angles using get_angles
    # 2. Apply sin to even indices and cos to odd indices
    # 3. Return the positional encoding with proper shape
    
    pass  # Replace with your implementation

def plot_positional_encoding(position, d_model):
    """
    Plot the positional encoding to visualize the patterns.
    
    Args:
        position: Maximum sequence length
        d_model: Model dimension
    """
    pos_encoding = positional_encoding(position, d_model)
    pos_encoding = tf.squeeze(pos_encoding, axis=0).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    plt.title(f'Positional Encoding (position={position}, d_model={d_model})')
    plt.savefig('positional_encoding.png')
    plt.show()

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate scaled dot-product attention as per the paper "Attention Is All You Need".
    
    Args:
        q: Query tensor of shape (..., seq_len_q, depth)
        k: Key tensor of shape (..., seq_len_k, depth)
        v: Value tensor of shape (..., seq_len_v, depth_v), where seq_len_v = seq_len_k
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
        
    Returns:
        output: Attention output of shape (..., seq_len_q, depth_v)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    # TODO: Implement scaled dot-product attention
    # 1. Calculate dot product of query and key
    # 2. Scale by square root of key dimension
    # 3. Apply mask if provided
    # 4. Apply softmax to get attention weights
    # 5. Compute weighted values
    # 6. Return output and attention weights
    
    pass  # Replace with your implementation

class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention layer as described in the transformer paper.
    """
    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        # TODO: Define the linear projection layers for Q, K, V
        # and the final output projection
        
        pass  # Replace with your implementation
    
    def split_heads(self, x, batch_size):
        """
        Split the input tensor into multiple heads.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # TODO: Implement split_heads
        # 1. Reshape x to (batch_size, seq_len, num_heads, depth)
        # 2. Transpose to (batch_size, num_heads, seq_len, depth)
        
        pass  # Replace with your implementation
    
    def call(self, v, k, q, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights for visualization
        """
        # TODO: Implement multi-head attention forward pass
        # 1. Apply linear projections to Q, K, V
        # 2. Split heads
        # 3. Apply scaled dot-product attention to each head
        # 4. Combine heads
        # 5. Apply final linear projection
        # 6. Return output and attention weights
        
        pass  # Replace with your implementation

def point_wise_feed_forward_network(d_model, dff):
    """
    Create a feed-forward network for transformer.
    
    Args:
        d_model: Model dimension
        dff: Hidden layer size
        
    Returns:
        Sequential model with two dense layers
    """
    # TODO: Implement the feed-forward network
    # 1. First dense layer with ReLU activation
    # 2. Second dense layer without activation
    
    pass  # Replace with your implementation

def create_padding_mask(seq):
    """
    Create padding mask for transformer.
    
    Args:
        seq: Input sequence
        
    Returns:
        mask: Padding mask
    """
    # TODO: Implement padding mask
    # 1. Create a mask for padding tokens (value 0)
    # 2. Add dimensions for broadcasting with attention scores
    
    pass  # Replace with your implementation

def create_look_ahead_mask(size):
    """
    Create look-ahead mask for transformer.
    
    Args:
        size: Size of the sequence
        
    Returns:
        mask: Look-ahead mask
    """
    # TODO: Implement look-ahead mask
    # 1. Create a triangular mask to prevent looking at future tokens
    
    pass  # Replace with your implementation

# Test your implementations
if __name__ == "__main__":
    print("Testing positional encoding...")
    try:
        pos_encoding = positional_encoding(position=50, d_model=512)
        plot_positional_encoding(position=50, d_model=512)
        print("✓ Positional encoding implemented")
    except Exception as e:
        print(f"✗ Error in positional encoding: {e}")
    
    print("\nTesting scaled dot-product attention...")
    try:
        q = tf.random.uniform((4, 5, 10))  # (batch_size, seq_len_q, depth)
        k = tf.random.uniform((4, 6, 10))  # (batch_size, seq_len_k, depth)
        v = tf.random.uniform((4, 6, 10))  # (batch_size, seq_len_v, depth)
        
        output, attention_weights = scaled_dot_product_attention(q, k, v)
        
        print(f"Attention output shape: {output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        print("✓ Scaled dot-product attention implemented")
    except Exception as e:
        print(f"✗ Error in scaled dot-product attention: {e}")
    
    print("\nTesting multi-head attention...")
    try:
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        q = tf.random.uniform((4, 5, 512))  # (batch_size, seq_len_q, d_model)
        k = tf.random.uniform((4, 6, 512))  # (batch_size, seq_len_k, d_model)
        v = tf.random.uniform((4, 6, 512))  # (batch_size, seq_len_v, d_model)
        
        output, attention_weights = mha(v, k, q, mask=None)
        
        print(f"Multi-head attention output shape: {output.shape}")
        print(f"Multi-head attention weights shape: {attention_weights.shape}")
        print("✓ Multi-head attention implemented")
    except Exception as e:
        print(f"✗ Error in multi-head attention: {e}")
    
    print("\nTesting feed-forward network...")
    try:
        ffn = point_wise_feed_forward_network(d_model=512, dff=2048)
        x = tf.random.uniform((4, 5, 512))  # (batch_size, seq_len, d_model)
        
        output = ffn(x)
        
        print(f"Feed-forward network output shape: {output.shape}")
        print("✓ Feed-forward network implemented")
    except Exception as e:
        print(f"✗ Error in feed-forward network: {e}")
    
    print("\nTesting masks...")
    try:
        seq = tf.constant([[1, 2, 0, 0], [1, 2, 3, 0]])
        padding_mask = create_padding_mask(seq)
        
        print(f"Padding mask shape: {padding_mask.shape}")
        print(f"Padding mask:\n{padding_mask.numpy()}")
        
        look_ahead_mask = create_look_ahead_mask(4)
        
        print(f"Look-ahead mask shape: {look_ahead_mask.shape}")
        print(f"Look-ahead mask:\n{look_ahead_mask.numpy()}")
        
        print("✓ Masks implemented")
    except Exception as e:
        print(f"✗ Error in masks: {e}")
