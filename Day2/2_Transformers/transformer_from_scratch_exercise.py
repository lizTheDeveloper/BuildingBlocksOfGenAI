"""
Transformer From Scratch Exercise
Building Blocks of Generative AI Course - Day 2

This exercise guides students through building a transformer model from scratch,
building on concepts from VAEs, feedforward neural networks, backpropagation,
and softmax covered in previous sessions.

The focus is on implementing the core components as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Part 1: Positional Encoding
# ===========================
# Unlike RNNs, transformers have no inherent notion of position,
# so we need to add positional information to our embeddings.

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
    # TODO: Implement the angle calculation as per the paper
    # The formula is: angle_rates = 1 / (10000 ** (2i / d_model))
    # where i is the dimension index
    
    pass

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
    # 1. Call get_angles to get the angle rates
    # 2. Apply sin to even indices and cos to odd indices
    # 3. Return the positional encoding with shape (1, position, d_model)
    
    pass

# Part 2: Scaled Dot-Product Attention
# ===================================
# The core attention mechanism in transformers

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate scaled dot-product attention as per the paper.
    
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
    # 2. Scale by sqrt(dk) where dk is the dimension of the key vectors
    # 3. Apply mask if provided (mask is used for padding or for causal attention)
    # 4. Apply softmax to get attention weights
    # 5. Compute weighted sum of values
    # 6. Return output and attention weights
    
    pass

# Part 3: Multi-Head Attention
# ===========================
# Multiple attention heads in parallel, allowing the model to attend to
# different parts of the sequence for different representation subspaces.

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
        
        # Check that d_model is divisible by num_heads
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        # TODO: Create linear layers for query, key, value, and output projections
        # 1. Define self.wq, self.wk, self.wv as Dense layers with d_model units
        # 2. Define self.dense as a Dense layer with d_model units for final output
        
        pass
    
    def split_heads(self, x, batch_size):
        """
        Split the input tensor into multiple heads.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # TODO: Implement the head splitting logic
        # 1. Reshape x to (batch_size, seq_len, num_heads, depth)
        # 2. Transpose to (batch_size, num_heads, seq_len, depth)
        
        pass
    
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
        
        # TODO: Implement the multi-head attention forward pass
        # 1. Apply the query, key, value linear projections
        # 2. Split heads using the split_heads method
        # 3. Apply scaled dot-product attention
        # 4. Combine heads and apply final linear projection
        
        pass

# Part 4: Point-wise Feed-Forward Network
# ======================================
# The feed-forward network applied to each position separately and identically

def point_wise_feed_forward_network(d_model, dff):
    """
    Create a feed-forward network for transformer.
    
    Args:
        d_model: Model dimension
        dff: Hidden layer size
        
    Returns:
        Sequential model with two dense layers
    """
    # TODO: Create a feed-forward network with two dense layers
    # 1. First layer with dff units and ReLU activation
    # 2. Second layer with d_model units
    
    pass

# Part 5: Encoder Layer
# ====================
# One layer of the encoder stack in the transformer

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
        
        # TODO: Initialize the components of the encoder layer
        # 1. Multi-head attention layer
        # 2. Feed-forward network
        # 3. Layer normalization layers
        # 4. Dropout layers
        
        pass
    
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
        # TODO: Implement the encoder layer forward pass
        # 1. Multi-head attention with residual connection and layer normalization
        # 2. Feed-forward network with residual connection and layer normalization
        # (Don't forget to apply dropout in training mode!)
        
        pass

# Part 6: Decoder Layer
# ====================
# One layer of the decoder stack in the transformer

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
        
        # TODO: Initialize the components of the decoder layer
        # 1. Two multi-head attention layers (one for self-attention, one for cross-attention)
        # 2. Feed-forward network
        # 3. Layer normalization layers (three of them)
        # 4. Dropout layers
        
        pass
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the decoder layer.
        
        Args:
            x: Input tensor
            enc_output: Output from the encoder
            training: Whether in training mode (for dropout)
            look_ahead_mask: Mask for the first multi-head attention layer (prevents seeing future tokens)
            padding_mask: Mask for the second multi-head attention layer
            
        Returns:
            x: Output of the decoder layer
            attention_weights: Dictionary of attention weights for visualization
        """
        # TODO: Implement the decoder layer forward pass
        # 1. Self-attention with look-ahead mask
        # 2. Cross-attention with encoder output
        # 3. Feed-forward network
        # (Don't forget residual connections, layer normalization, and dropout!)
        
        pass

# Part 7: Encoder
# =============
# The full encoder stack in the transformer

class Encoder(layers.Layer):
    """
    Encoder of the transformer architecture.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        """
        Initialize the encoder.
        
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward networks
            input_vocab_size: Size of the input vocabulary
            maximum_position_encoding: Maximum sequence length
            rate: Dropout rate
        """
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # TODO: Initialize the components of the encoder
        # 1. Embedding layer
        # 2. Positional encoding
        # 3. Dropout layer
        # 4. Stack of encoder layers
        
        pass
    
    def call(self, x, training, mask=None):
        """
        Forward pass for the encoder.
        
        Args:
            x: Input tensor
            training: Whether in training mode (for dropout)
            mask: Optional mask tensor
            
        Returns:
            x: Output of the encoder
        """
        # TODO: Implement the encoder forward pass
        # 1. Apply embedding and scale by sqrt(d_model)
        # 2. Add positional encoding
        # 3. Apply dropout in training mode
        # 4. Process through all encoder layers in sequence
        
        pass

# Part 8: Decoder
# =============
# The full decoder stack in the transformer

class Decoder(layers.Layer):
    """
    Decoder of the transformer architecture.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        """
        Initialize the decoder.
        
        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward networks
            target_vocab_size: Size of the target vocabulary
            maximum_position_encoding: Maximum sequence length
            rate: Dropout rate
        """
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # TODO: Initialize the components of the decoder
        # 1. Embedding layer
        # 2. Positional encoding
        # 3. Dropout layer
        # 4. Stack of decoder layers
        
        pass
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the decoder.
        
        Args:
            x: Input tensor
            enc_output: Output from the encoder
            training: Whether in training mode (for dropout)
            look_ahead_mask: Mask for self-attention to prevent seeing future tokens
            padding_mask: Mask for cross-attention
            
        Returns:
            x: Output of the decoder
            attention_weights: Dictionary of attention weights for visualization
        """
        # TODO: Implement the decoder forward pass
        # 1. Apply embedding and scale by sqrt(d_model)
        # 2. Add positional encoding
        # 3. Apply dropout in training mode
        # 4. Process through all decoder layers in sequence
        # 5. Collect attention weights from all layers
        
        pass

# Part 9: Transformer
# =================
# The complete transformer model

class Transformer(tf.keras.Model):
    """
    Transformer model as described in "Attention Is All You Need".
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        """
        Initialize the transformer.
        
        Args:
            num_layers: Number of encoder and decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward networks
            input_vocab_size: Size of the input vocabulary
            target_vocab_size: Size of the target vocabulary
            pe_input: Maximum input sequence length for positional encoding
            pe_target: Maximum target sequence length for positional encoding
            rate: Dropout rate
        """
        super(Transformer, self).__init__()
        
        # TODO: Initialize the components of the transformer
        # 1. Encoder
        # 2. Decoder
        # 3. Final dense layer for predicting target vocabulary
        
        pass
    
    def call(self, inputs, training):
        """
        Forward pass for the transformer.
        
        Args:
            inputs: Tuple of (inp, tar) where inp is the input sequence and tar is the target sequence
            training: Whether in training mode (for dropout)
            
        Returns:
            final_output: Logits for target vocabulary
            attention_weights: Dictionary of attention weights for visualization
        """
        # TODO: Implement the transformer forward pass
        # 1. Unpack input and target sequences
        # 2. Create appropriate masks
        # 3. Process input through encoder
        # 4. Process through decoder with encoder output
        # 5. Apply final dense layer
        
        pass

# Part 10: Creating Masks
# =====================
# Functions to create padding and look-ahead masks

def create_padding_mask(seq):
    """
    Create a padding mask for the encoder attention.
    
    Args:
        seq: Input sequence
        
    Returns:
        mask: Padding mask
    """
    # TODO: Implement padding mask creation
    # The mask should be 1 for padding tokens (zeros in the sequence) and 0 for non-padding tokens
    
    pass

def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for the decoder self-attention.
    
    Args:
        size: Size of the mask
        
    Returns:
        mask: Look-ahead mask
    """
    # TODO: Implement look-ahead mask creation
    # The mask should allow a position to attend to all positions up to and including itself
    # It should prevent attending to future positions
    
    pass

# Part 11: Visualization Functions
# ==============================
# Functions to visualize positional encoding and attention weights

def plot_positional_encoding(pe, title="Positional Encoding"):
    """
    Plot the positional encoding.
    
    Args:
        pe: Positional encoding
        title: Plot title
    """
    # TODO: Implement a function to visualize the positional encoding
    # Plot the positional encoding values as a heatmap
    
    pass

def plot_attention_weights(attention_weights, sentence, result, layer_name, title="Attention Weights"):
    """
    Plot the attention weights.
    
    Args:
        attention_weights: Attention weights dictionary
        sentence: Input sentence
        result: Output sentence
        layer_name: Name of the layer to visualize
        title: Plot title
    """
    # TODO: Implement a function to visualize attention weights
    # Plot the attention weights as a heatmap
    
    pass

# Try out the positional encoding
if __name__ == "__main__":
    # Example usage:
    # 1. Create sample positional encoding
    # 2. Visualize it
    # This can help students understand how positional encoding works
    
    print("Implement and try the positional encoding!")
