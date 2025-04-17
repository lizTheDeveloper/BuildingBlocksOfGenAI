"""
Sequence-to-Sequence Model with Attention
Building Blocks of Generative AI Course - Day 2

This module contains the encoder and decoder components for a 
sequence-to-sequence model with attention.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from attention_layer import ScaledDotProductAttention

class Encoder(keras.Model):
    """
    Encoder model for the sequence-to-sequence translation system.
    """
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        """
        Initialize the encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            enc_units: Number of units in the encoder GRU
            batch_size: Batch size for processing
        """
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        
        # TODO: Define the encoder layers
        # You will need:
        # 1. An embedding layer to convert input tokens to vectors
        # 2. A GRU layer to process the sequence
        
        # Hint: The embedding layer should have shape (vocab_size, embedding_dim)
        # The GRU should have enc_units hidden units
        
        pass
    
    def call(self, x, hidden):
        """
        Forward pass for the encoder.
        
        Args:
            x: Input tensor (batch_size, max_length)
            hidden: Initial hidden state
            
        Returns:
            output: Encoder outputs for all timesteps (batch_size, max_length, enc_units)
            state: Final encoder state (batch_size, enc_units)
        """
        # TODO: Implement the encoder forward pass
        # 1. Pass input through the embedding layer
        # 2. Pass embedded input through the GRU
        # 3. Return the outputs and final state
        
        # Placeholder implementation - replace with your code
        batch_size = tf.shape(x)[0]
        output = tf.zeros((batch_size, tf.shape(x)[1], self.enc_units))
        state = tf.zeros((batch_size, self.enc_units))
        
        return output, state
    
    def initialize_hidden_state(self):
        """
        Initialize the encoder hidden state with zeros.
        
        Returns:
            initial_state: Initial hidden state for the encoder
        """
        return tf.zeros((self.batch_size, self.enc_units))

class Decoder(keras.Model):
    """
    Decoder model for the sequence-to-sequence translation system.
    """
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        """
        Initialize the decoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            dec_units: Number of units in the decoder GRU
            batch_size: Batch size for processing
        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        
        # TODO: Define the decoder layers
        # You will need:
        # 1. An embedding layer
        # 2. A GRU layer
        # 3. A dense layer for output projection
        # 4. An attention mechanism (ScaledDotProductAttention)
        
        pass
    
    def call(self, x, hidden, enc_output):
        """
        Forward pass for the decoder for a single timestep.
        
        Args:
            x: Input token (batch_size, 1)
            hidden: Previous decoder hidden state
            enc_output: Encoder outputs
            
        Returns:
            x: Output predictions
            state: New hidden state
            attention_weights: Attention weights for visualization
        """
        # TODO: Implement the decoder forward pass with attention
        # 1. Pass input through the embedding layer
        # 2. Use the attention mechanism to compute context vector and attention weights
        # 3. Concatenate the context vector with the embedded input
        # 4. Pass the concatenated vector through the GRU
        # 5. Project the GRU output to vocabulary size
        # 6. Return output, new state, and attention weights
        
        # Placeholder implementation - replace with your code
        batch_size = tf.shape(x)[0]
        output = tf.zeros((batch_size, 1, self.dec_units))
        state = tf.zeros((batch_size, self.dec_units))
        attention_weights = tf.zeros((batch_size, tf.shape(enc_output)[1], 1))
        
        return output, state, attention_weights
