"""
Transformer Model
Building Blocks of Generative AI Course - Day 2

This module implements the complete Transformer architecture as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformer_components import (
    positional_encoding,
    EncoderLayer,
    DecoderLayer
)

class Encoder(layers.Layer):
    """
    Encoder for the transformer architecture.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        """
        Initialize the encoder.
        
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward network
            input_vocab_size: Size of the input vocabulary
            maximum_position_encoding: Maximum sequence length for positional encoding
            rate: Dropout rate
        """
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                               self.d_model)
        
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(rate)
    
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
        seq_len = tf.shape(x)[1]
        
        # Convert input tokens to embeddings
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Add positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        # Pass through all encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)

class Decoder(layers.Layer):
    """
    Decoder for the transformer architecture.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        """
        Initialize the decoder.
        
        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward network
            target_vocab_size: Size of the target vocabulary
            maximum_position_encoding: Maximum sequence length for positional encoding
            rate: Dropout rate
        """
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(rate)
    
    def call(self, x, enc_output, training, 
            look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the decoder.
        
        Args:
            x: Input tensor
            enc_output: Output from the encoder
            training: Whether in training mode (for dropout)
            look_ahead_mask: Mask for the self-attention layer
            padding_mask: Mask for the cross-attention layer
            
        Returns:
            x: Output of the decoder
            attention_weights: Dictionary of attention weights for visualization
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # Convert input tokens to embeddings
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Add positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        # Pass through all decoder layers
        for i in range(self.num_layers):
            x, block_attention = self.dec_layers[i](x, enc_output, training,
                                                  look_ahead_mask, padding_mask)
            
            # Store attention weights for visualization
            attention_weights[f'decoder_layer{i+1}_self_attention'] = block_attention['decoder_layer_self_attention']
            attention_weights[f'decoder_layer{i+1}_cross_attention'] = block_attention['decoder_layer_cross_attention']
        
        return x, attention_weights

class Transformer(keras.Model):
    """
    Complete transformer model as described in "Attention Is All You Need".
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
        """
        Initialize the transformer.
        
        Args:
            num_layers: Number of encoder and decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden layer size in the feed-forward network
            input_vocab_size: Size of the input vocabulary
            target_vocab_size: Size of the target vocabulary
            pe_input: Maximum sequence length for input positional encoding
            pe_target: Maximum sequence length for target positional encoding
            rate: Dropout rate
        """
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                              input_vocab_size, pe_input, rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                              target_vocab_size, pe_target, rate)
        
        self.final_layer = layers.Dense(target_vocab_size)
    
    def call(self, inputs, training):
        """
        Forward pass for the transformer.
        
        Args:
            inputs: Tuple of (input tensor, target tensor)
            training: Whether in training mode (for dropout)
            
        Returns:
            predictions: Output predictions
            attention_weights: Dictionary of attention weights for visualization
        """
        # Unpack inputs
        inp, tar = inputs
        
        # Create masks
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        
        # Encoder
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        # Decoder
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        # Final linear layer
        predictions = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return predictions, attention_weights
    
    def create_masks(self, inp, tar):
        """
        Create masks for transformer.
        
        Args:
            inp: Input tensor
            tar: Target tensor
            
        Returns:
            enc_padding_mask: Padding mask for encoder
            look_ahead_mask: Look-ahead mask for decoder self-attention
            dec_padding_mask: Padding mask for decoder cross-attention
        """
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)
        
        # Decoder padding mask (for encoder-decoder attention)
        dec_padding_mask = self.create_padding_mask(inp)
        
        # Look ahead mask and padding mask for decoder self-attention
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, look_ahead_mask, dec_padding_mask
    
    @staticmethod
    def create_padding_mask(seq):
        """
        Create padding mask for transformer.
        
        Args:
            seq: Input sequence
            
        Returns:
            mask: Padding mask
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        
        # Add extra dimensions to add the padding to the attention logits
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    @staticmethod
    def create_look_ahead_mask(size):
        """
        Create look-ahead mask for transformer.
        
        Args:
            size: Size of the sequence
            
        Returns:
            mask: Look-ahead mask
        """
        # Create triangular mask to ensure decoder can't peek at future tokens
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
