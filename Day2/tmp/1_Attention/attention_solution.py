"""
Attention Mechanism Exercise - Solution
Building Blocks of Generative AI Course - Day 2

This file contains the solution for implementing the Bahdanau attention mechanism
and the encoder-decoder models.
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
        
        # Define the layers for the attention mechanism
        # W_a for processing encoder outputs
        self.W = layers.Dense(units)
        
        # U_a for processing decoder state
        self.U = layers.Dense(units)
        
        # v_a for producing attention scores
        self.V = layers.Dense(1)
    
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
        # Expand dimensions of query to match values for broadcasting
        # query shape: (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Calculate the attention scores
        # 1. Process encoder outputs with W_a
        # values shape: (batch_size, max_length, hidden_size)
        # w_values shape: (batch_size, max_length, units)
        w_values = self.W(values)
        
        # 2. Process decoder state with U_a
        # query_with_time_axis shape: (batch_size, 1, hidden_size)
        # u_query shape: (batch_size, 1, units)
        u_query = self.U(query_with_time_axis)
        
        # 3. Calculate score: v_a^T * tanh(W_a * h_i + U_a * s_t)
        # score shape: (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(w_values + u_query))
        
        # Apply softmax to get attention weights
        # attention_weights shape: (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Create the context vector as a weighted sum of values
        # values shape: (batch_size, max_length, hidden_size)
        # attention_weights shape: (batch_size, max_length, 1)
        # context_vector shape: (batch_size, hidden_size)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context_vector, attention_weights

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
        
        # Define the encoder layers
        # Embedding layer to convert input tokens to vectors
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        # GRU layer to process the sequence
        self.gru = layers.GRU(enc_units,
                             return_sequences=True,
                             return_state=True,
                             recurrent_initializer='glorot_uniform')
    
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
        # Pass input through the embedding layer
        # x shape: (batch_size, max_length)
        # embedded shape: (batch_size, max_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Pass embedded input through the GRU
        # output shape: (batch_size, max_length, enc_units)
        # state shape: (batch_size, enc_units)
        output, state = self.gru(embedded, initial_state=hidden)
        
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
        
        # Define the decoder layers
        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        # GRU layer
        self.gru = layers.GRU(dec_units,
                             return_sequences=True,
                             return_state=True,
                             recurrent_initializer='glorot_uniform')
        
        # Dense layer for output projection
        self.fc = layers.Dense(vocab_size)
        
        # Attention mechanism
        self.attention = BahdanauAttention(dec_units)
    
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
        # Use the attention mechanism to compute context vector and attention weights
        # context_vector shape: (batch_size, hidden_size)
        # attention_weights shape: (batch_size, max_length, 1)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        
        # Pass input through the embedding layer
        # x shape: (batch_size, 1)
        # embedded shape: (batch_size, 1, embedding_dim)
        embedded = self.embedding(x)
        
        # Concatenate the context vector with the embedded input
        # context_vector shape: (batch_size, hidden_size)
        # expanded_context shape: (batch_size, 1, hidden_size)
        expanded_context = tf.expand_dims(context_vector, 1)
        
        # concat_input shape: (batch_size, 1, embedding_dim + hidden_size)
        concat_input = tf.concat([embedded, expanded_context], axis=-1)
        
        # Pass the concatenated vector through the GRU
        # output shape: (batch_size, 1, dec_units)
        # state shape: (batch_size, dec_units)
        output, state = self.gru(concat_input, initial_state=hidden)
        
        # Reshape output to (batch_size, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # Project the output to vocabulary size
        # x shape: (batch_size, vocab_size)
        x = self.fc(output)
        
        return x, state, attention_weights
