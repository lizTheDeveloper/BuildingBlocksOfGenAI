"""
Attention Layer Implementation
Building Blocks of Generative AI Course - Day 2

This module contains the implementation of the Scaled Dot-Product Attention mechanism
as described in the "Attention Is All You Need" paper (Vaswani et al., 2017).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    # 1. Calculate dot product of query and key (matmul_qk)
    # 2. Scale by sqrt(dk) where dk is the dimension of the key vectors
    # 3. Apply mask if provided
    # 4. Apply softmax to get attention weights
    # 5. Compute weighted sum of values
    # 6. Return output and attention weights
    
    # Placeholder implementation - replace with your code
    batch_size = tf.shape(q)[0]
    seq_len_q = tf.shape(q)[1]
    seq_len_k = tf.shape(k)[1]
    
    # Dummy implementation that should be replaced
    attention_weights = tf.ones((batch_size, seq_len_q, seq_len_k)) / float(seq_len_k)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention layer as described in the "Attention Is All You Need" paper.
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
        
        # TODO: Define linear layers for query, key, value projections, and output
        # These are the WQ, WK, WV, and WO matrices in the paper
        
        pass
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension of x into (num_heads, depth)
        and transpose the result to (batch_size, num_heads, seq_len, depth).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # TODO: Reshape x to (batch_size, seq_len, num_heads, depth)
        # Then transpose to (batch_size, num_heads, seq_len, depth)
        
        # Placeholder implementation - replace with your code
        x_split = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x_split, perm=[0, 2, 1, 3])
    
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
        batch_size = tf.shape(q)[0]
        
        # TODO: Implement the multi-head attention mechanism:
        # 1. Apply the query, key, value linear transformations
        # 2. Split heads
        # 3. Apply scaled dot-product attention
        # 4. Reshape and apply output linear transformation
        
        # Placeholder implementation - replace with your code
        seq_len_q = tf.shape(q)[1]
        output = tf.zeros((batch_size, seq_len_q, self.d_model))
        attention_weights = None
        
        return output, attention_weights
