"""
Attention Mechanism Examples for Slides
Building Blocks of Generative AI Course - Day 2
"""

import tensorflow as tf
import numpy as np


# 1. Bahdanau (Additive) Attention
class BahdanauAttention(tf.keras.layers.Layer):
    """
    Implements Bahdanau (Additive) Attention as described in:
    "Neural Machine Translation by Jointly Learning to Align and Translate"
    (Bahdanau et al., 2015)
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # Dense layers for the small feed-forward network
        self.W1 = tf.keras.layers.Dense(units)  # For encoder outputs
        self.W2 = tf.keras.layers.Dense(units)  # For decoder state
        self.V = tf.keras.layers.Dense(1)       # For attention scores
    
    def call(self, query, values):
        """
        Apply attention mechanism to query and values.
        
        Args:
            query: Decoder hidden state (batch_size, hidden_size)
            values: Encoder outputs (batch_size, max_length, hidden_size)
            
        Returns:
            context_vector: Weighted sum of values based on attention weights
            attention_weights: Attention weights for visualization
        """
        # Add time axis to query for broadcasting
        # query shape: (batch_size, hidden_size)
        # query_with_time_axis shape: (batch_size, 1, hidden_size)
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Calculate attention score using a small feed-forward network
        # score shape: (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)))
        
        # Apply softmax to get attention weights
        # attention_weights shape: (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Create context vector as weighted sum of values
        # context_vector shape: (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


# 2. Luong (Multiplicative) Attention
class LuongAttention(tf.keras.layers.Layer):
    """
    Implements Luong Attention as described in:
    "Effective Approaches to Attention-based Neural Machine Translation"
    (Luong et al., 2015)
    """
    def __init__(self, units, attention_type='general'):
        super(LuongAttention, self).__init__()
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.W = tf.keras.layers.Dense(units)
        elif attention_type == 'concat':
            self.W = tf.keras.layers.Dense(units)
            self.U = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        """
        Apply Luong attention mechanism to query and values.
        
        Args:
            query: Decoder hidden state (batch_size, hidden_size)
            values: Encoder outputs (batch_size, max_length, hidden_size)
            
        Returns:
            context_vector: Weighted sum of values based on attention weights
            attention_weights: Attention weights for visualization
        """
        # Calculate attention scores based on attention type
        if self.attention_type == 'dot':
            # Simple dot product
            # query shape: (batch_size, hidden_size)
            # values shape: (batch_size, max_length, hidden_size)
            query_with_time_axis = tf.expand_dims(query, 1)
            score = tf.matmul(query_with_time_axis, values, transpose_b=True)
            
        elif self.attention_type == 'general':
            # Weighted dot product
            # query shape: (batch_size, hidden_size)
            # values shape: (batch_size, max_length, hidden_size)
            query_with_time_axis = tf.expand_dims(query, 1)
            transformed_values = self.W(values)
            score = tf.matmul(query_with_time_axis, transformed_values, transpose_b=True)
            
        elif self.attention_type == 'concat':
            # Concatenation-based attention
            # query shape: (batch_size, hidden_size)
            # values shape: (batch_size, max_length, hidden_size)
            query_with_time_axis = tf.expand_dims(query, 1)
            expanded_query = tf.tile(query_with_time_axis, [1, tf.shape(values)[1], 1])
            
            # Concatenate query and values
            concat = tf.concat([expanded_query, values], axis=-1)
            score = self.V(tf.nn.tanh(self.W(concat)))
        
        # Apply softmax to get attention weights
        # attention_weights shape: (batch_size, max_length)
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # Create context vector as weighted sum of values
        # context_vector shape: (batch_size, hidden_size)
        context_vector = tf.matmul(attention_weights, values)
        
        return context_vector, attention_weights


# 3. Scaled Dot-Product Attention (Transformer Attention)
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate scaled dot-product attention as described in:
    "Attention Is All You Need" (Vaswani et al., 2017)
    
    Args:
        q: Query tensor of shape (..., seq_len_q, depth)
        k: Key tensor of shape (..., seq_len_k, depth)
        v: Value tensor of shape (..., seq_len_v, depth_v), where seq_len_v = seq_len_k
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
        
    Returns:
        output: Attention output of shape (..., seq_len_q, depth_v)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    # Calculate dot product of query and key
    # matmul_qk shape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Scale by square root of key dimension
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # Apply mask (if provided)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # Add large negative values to masked positions
    
    # Apply softmax to get attention weights
    # attention_weights shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # Calculate output as weighted sum of values
    # output shape: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights


# 4. Multi-Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention as described in:
    "Attention Is All You Need" (Vaswani et al., 2017)
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        # Linear projection layers
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        """
        Apply multi-head attention to query, key, and value tensors.
        
        Args:
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Dictionary of attention weights for visualization
        """
        batch_size = tf.shape(q)[0]
        
        # Linear projections
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        # Transpose and reshape to combine heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        # Final linear projection
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights


# 5. Creating masks for transformer attention
def create_padding_mask(seq):
    """
    Create a mask to hide padding tokens in encoder/decoder.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        
    Returns:
        mask: Padding mask of shape (batch_size, 1, 1, seq_len)
    """
    # Create mask where 0 tokens are masked (1 in the mask)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # Add extra dimensions for broadcasting with attention logits
    # output shape: (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    """
    Create a mask to prevent attention to future tokens in decoder.
    
    Args:
        size: Size of the sequence
        
    Returns:
        mask: Look-ahead mask of shape (size, size)
    """
    # Create a lower triangular matrix with ones
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # Output shape: (size, size)
    return mask

def create_combined_mask(seq):
    """
    Create a combined mask for decoder (padding + look-ahead).
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        
    Returns:
        mask: Combined mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Create look-ahead mask
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)
    
    # Create padding mask
    padding_mask = create_padding_mask(seq)[:, :, 0, :]
    
    # Combine masks: max(look_ahead, padding) since both use 1 for masking
    combined_mask = tf.maximum(look_ahead_mask, padding_mask[:, tf.newaxis, :])
    
    return combined_mask


# Example usage code
def attention_mechanism_demo():
    # Create sample inputs
    batch_size = 2
    seq_len = 5
    hidden_size = 8
    
    # Sample data
    query = tf.random.normal([batch_size, hidden_size])
    values = tf.random.normal([batch_size, seq_len, hidden_size])
    
    # Bahdanau attention
    bahdanau_attention = BahdanauAttention(hidden_size)
    bahdanau_context, bahdanau_weights = bahdanau_attention(query, values)
    
    print("Bahdanau (Additive) Attention:")
    print(f"Context vector shape: {bahdanau_context.shape}")
    print(f"Attention weights shape: {bahdanau_weights.shape}")
    
    # Scaled dot-product attention
    q = tf.random.normal([batch_size, seq_len, hidden_size])
    k = tf.random.normal([batch_size, seq_len, hidden_size])
    v = tf.random.normal([batch_size, seq_len, hidden_size])
    
    sdp_output, sdp_weights = scaled_dot_product_attention(q, k, v)
    
    print("\nScaled Dot-Product Attention:")
    print(f"Output shape: {sdp_output.shape}")
    print(f"Attention weights shape: {sdp_weights.shape}")
    
    # Multi-head attention
    d_model = 64
    num_heads = 4
    
    q = tf.random.normal([batch_size, seq_len, d_model])
    k = tf.random.normal([batch_size, seq_len, d_model])
    v = tf.random.normal([batch_size, seq_len, d_model])
    
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output, mha_weights = mha(v, k, q)
    
    print("\nMulti-Head Attention:")
    print(f"Output shape: {mha_output.shape}")
    print(f"Attention weights shape: {mha_weights.shape}")
    
    # Masking examples
    seq = tf.constant([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    padding_mask = create_padding_mask(seq)
    look_ahead_mask = create_look_ahead_mask(5)
    combined_mask = create_combined_mask(seq)
    
    print("\nMasking Examples:")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Look-ahead mask shape: {look_ahead_mask.shape}")
    print(f"Combined mask shape: {combined_mask.shape}")


if __name__ == "__main__":
    attention_mechanism_demo()
