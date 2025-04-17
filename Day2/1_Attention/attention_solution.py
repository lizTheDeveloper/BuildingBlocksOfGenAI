"""
Attention Mechanism Exercise - Solution
Building Blocks of Generative AI Course - Day 2

This file contains the solution for implementing the Scaled Dot-Product Attention 
mechanism from the "Attention is All You Need" paper (Vaswani et al., 2017)
and the encoder-decoder models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ScaledDotProductAttention(layers.Layer):
    """
    Implements Scaled Dot-Product Attention as a Keras layer.
    
    This attention mechanism was described in:
    "Attention Is All You Need" (Vaswani et al., 2017)
    """
    def __init__(self):
        """
        Initialize the scaled dot-product attention layer.
        """
        super(ScaledDotProductAttention, self).__init__()
    
    def call(self, query, key, value, mask=None):
        """
        Apply scaled dot-product attention mechanism.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, depth)
            key: Key tensor (batch_size, seq_len_k, depth)
            value: Value tensor (batch_size, seq_len_v, depth_v), where seq_len_v = seq_len_k
            mask: Optional mask tensor of shape (batch_size, seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, depth_v)
            attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
        """
        # Calculate dot product of query and key
        # matmul_qk shape: (batch_size, seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # Scale matmul_qk by square root of depth (dimension of key vectors)
        # This scaling prevents the softmax from having extremely small gradients
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
        
        # Apply mask if provided (useful for padding or causal attention)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  
        
        # Apply softmax to get attention weights
        # attention_weights shape: (batch_size, seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Compute weighted sum of values
        # output shape: (batch_size, seq_len_q, depth_v)
        output = tf.matmul(attention_weights, value)
        
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
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
        # The core attention mechanism
        self.scaled_attention = ScaledDotProductAttention()
    
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
        scaled_attention, attention_weights = self.scaled_attention(
            q, k, v, mask)
        
        # Reshape the result
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        # Final linear projection
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights

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
    Decoder model for the sequence-to-sequence translation system with attention.
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
        
        # Attention mechanism - using multi-head attention with a single head for simplicity
        # In a full transformer, we would use multiple heads
        self.attention = MultiHeadAttention(d_model=dec_units, num_heads=1)
    
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
        # Reshape hidden state to work with our attention mechanism
        # hidden shape: (batch_size, dec_units)
        # reshaped_hidden shape: (batch_size, 1, dec_units)
        reshaped_hidden = tf.expand_dims(hidden, 1)
        
        # Use the attention mechanism to compute context vector and attention weights
        # context_vector shape: (batch_size, 1, dec_units)
        # attention_weights shape: Depends on implementation, typically (batch_size, 1, src_seq_len)
        context_vector, attention_weights = self.attention(
            v=enc_output,
            k=enc_output,
            q=reshaped_hidden
        )
        
        # Pass input through the embedding layer
        # x shape: (batch_size, 1)
        # embedded shape: (batch_size, 1, embedding_dim)
        embedded = self.embedding(x)
        
        # Concatenate the context vector with the embedded input
        # context_vector shape: (batch_size, 1, dec_units)
        # embedded shape: (batch_size, 1, embedding_dim)
        # concat_input shape: (batch_size, 1, embedding_dim + dec_units)
        concat_input = tf.concat([embedded, context_vector], axis=-1)
        
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


if __name__ == "__main__":
    # Test the ScaledDotProductAttention layer
    query = tf.random.normal([1, 3, 4])
    key = tf.random.normal([1, 3, 4])
    value = tf.random.normal([1, 3, 4])
    attention = ScaledDotProductAttention()
    output, weights = attention(query, key, value)