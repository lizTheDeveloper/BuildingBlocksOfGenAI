"""
Transformer Components
Building Blocks of Generative AI Course - Day 2

This module contains the basic building blocks of the Transformer architecture,
including multi-head attention, positional encoding, and feed-forward networks.
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
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
    Create positional encoding for transformer input.
    
    Args:
        position: Maximum sequence length
        d_model: Model dimension
        
    Returns:
        pos_encoding: Positional encoding of shape (1, position, d_model)
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

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
    # Calculate dot product of query and key
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # Add the mask to the scaled tensor (if provided)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    
    # Normalize with softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    # Compute output as weighted sum of values
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights

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
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """
        Split the input tensor into multiple heads.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
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
            attention_weights: Attention weights for visualization
        """
        batch_size = tf.shape(q)[0]
        
        # Linear projections
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        # Split the projected matrices into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        # Reshape the result
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        # Final linear projection
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """
    Create a feed-forward network for transformer.
    
    Args:
        d_model: Model dimension
        dff: Hidden layer size
        
    Returns:
        Sequential model with two dense layers
    """
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(layers.Layer):
    """
    Encoder layer of the transformer architecture.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Initialize the encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward network
            rate: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        """
        Forward pass for the encoder layer.
        
        Args:
            x: Input tensor
            training: Whether in training mode (for dropout)
            mask: Optional mask tensor
            
        Returns:
            x: Output of the encoder layer
        """
        # Multi-head attention with residual connection and layer normalization
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection and normalization
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection and normalization
        
        return out2

class DecoderLayer(layers.Layer):
    """
    Decoder layer of the transformer architecture.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Initialize the decoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward network
            rate: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Self-attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Cross-attention
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the decoder layer.
        
        Args:
            x: Input tensor
            enc_output: Output from the encoder
            training: Whether in training mode (for dropout)
            look_ahead_mask: Mask for the self-attention layer
            padding_mask: Mask for the cross-attention layer
            
        Returns:
            x: Output of the decoder layer
            attention_weights: Dictionary of attention weights for visualization
        """
        # First multi-head attention (self-attention) with lookahead mask
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # Second multi-head attention (cross-attention) with padding mask
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        # Store attention weights for visualization
        attention_weights = {
            'decoder_layer_self_attention': attn_weights_block1,
            'decoder_layer_cross_attention': attn_weights_block2
        }
        
        return out3, attention_weights
