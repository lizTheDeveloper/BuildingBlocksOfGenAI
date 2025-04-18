"""
Multi-Head Attention Implementation
Building Blocks of Generative AI Course - Day 2

This module contains the implementation of Multi-Head Attention
as described in "Attention Is All You Need" (Vaswani et al., 2017).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ScaledDotProductAttention(layers.Layer):
    """
    Scaled Dot-Product Attention mechanism as described in the Transformer paper.
    
    This is the foundation for multi-head attention.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def call(self, query, key, value, mask=None):
        """
        Apply scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (..., seq_len_q, depth)
            key: Key tensor of shape (..., seq_len_k, depth)
            value: Value tensor of shape (..., seq_len_v, depth_v), where seq_len_v = seq_len_k
            mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output of shape (..., seq_len_q, depth_v)
            attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
        """
        # Calculate dot product of query and key
        # matmul_qk shape: (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # Scale matmul_qk by square root of dimension
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
        
        # Apply mask if provided (for padding or causal attention)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  
        
        # Apply softmax to get attention weights
        # attention_weights shape: (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Compute weighted sum of values
        # output shape: (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, value)
        
        return output, attention_weights

class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention layer as described in the transformer paper.
    
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention layer.
        
        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Check that d_model is divisible by num_heads
        assert d_model % self.num_heads == 0
        
        # Determine dimension per head
        self.depth = d_model // self.num_heads
        
        # Create linear projections for query, key, value
        self.wq = layers.Dense(d_model)  # Query projection
        self.wk = layers.Dense(d_model)  # Key projection
        self.wv = layers.Dense(d_model)  # Value projection
        
        # Output projection
        self.dense = layers.Dense(d_model)
        
        # Create the scaled dot-product attention layer
        self.attention = ScaledDotProductAttention()
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension of x into (num_heads, depth)
        and transpose the result so that the shape is
        (batch_size, num_heads, seq_len, depth)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Split tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # Reshape x to (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        
        # Transpose to (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
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
            attention_weights: Dictionary of attention weights from each head
        """
        batch_size = tf.shape(q)[0]
        
        # Step 1: Apply linear projections to create queries, keys, and values
        # These projections learn different representations for each head
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        # Step 2: Split into multiple heads
        # This allows each head to attend to different parts of the input
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Step 3: Apply scaled dot-product attention for each head
        # Each head independently attends to the input
        scaled_attention, attention_weights = self.attention(q, k, v, mask)
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Step 4: Transpose and reshape to combine all heads' outputs
        # First transpose from: (batch_size, num_heads, seq_len_q, depth)
        # To: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # Reshape to: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))
        
        # Step 5: Apply final output projection
        # This learns how to combine the outputs from all heads
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        # Create a dictionary of attention weights for each head
        # This is useful for visualization and analysis
        attention_weights_dict = {}
        for h in range(self.num_heads):
            attention_weights_dict[f'head_{h+1}'] = attention_weights[:, h, :, :]
        
        return output, attention_weights_dict
