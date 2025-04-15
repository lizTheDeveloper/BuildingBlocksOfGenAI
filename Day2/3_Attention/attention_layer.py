"""
Attention Layer Implementation
Building Blocks of Generative AI Course - Day 2

This module contains the implementation of the Bahdanau (Additive) Attention mechanism
as a TensorFlow Keras layer.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BahdanauAttention(layers.Layer):
    """
    Implements Bahdanau (Additive) Attention as a Keras layer.
    
    This attention mechanism was described in:
    "Neural Machine Translation by Jointly Learning to Align and Translate"
    (Bahdanau et al., 2015)
    """
    def __init__(self, units):
        """
        Initialize the attention layer.
        
        Args:
            units: Number of units in the attention layer
        """
        super(BahdanauAttention, self).__init__()
        
        # TODO: Define the layers for the attention mechanism
        # You will need:
        # 1. A dense layer for processing the encoder outputs (W_a)
        # 2. A dense layer for processing the decoder state (U_a)
        # 3. A dense layer with 1 unit for producing attention scores (v_a)
        
        # Hint: The formula for Bahdanau attention is:
        # score(s_t, h_i) = v_a^T * tanh(W_a * h_i + U_a * s_t)
        # where:
        # - s_t is the decoder state at time t
        # - h_i is the encoder output at position i
        # - W_a, U_a, and v_a are learned parameters
        
        pass
    
    def call(self, query, values):
        """
        Apply attention mechanism to query and values.
        
        Args:
            query: Decoder state (batch_size, hidden_size)
            values: Encoder outputs (batch_size, max_length, hidden_size)
            
        Returns:
            context_vector: Weighted sum of values based on attention weights
            attention_weights: Attention weights for visualization
        """
        # TODO: Implement the attention mechanism
        # 1. Expand dimensions of query to match values for broadcasting
        # 2. Calculate the attention scores using the layers defined in __init__
        # 3. Apply softmax to get attention weights
        # 4. Create the context vector as a weighted sum of values
        # 5. Return the context vector and attention weights
        
        # Placeholder implementation - replace with your code
        batch_size = tf.shape(values)[0]
        max_length = tf.shape(values)[1]
        
        # Dummy implementation that should be replaced
        attention_weights = tf.ones((batch_size, max_length)) / float(max_length)
        attention_weights = tf.expand_dims(attention_weights, -1)
        context_vector = tf.reduce_sum(values * attention_weights, axis=1)
        
        return context_vector, attention_weights
