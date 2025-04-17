"""
Transformer From Scratch Solution
Building Blocks of Generative AI Course - Day 2

This file provides the complete solution for the transformer from scratch exercise,
implementing the architecture as described in "Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Part 1: Positional Encoding
# ===========================

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

# Part 2: Scaled Dot-Product Attention
# ===================================

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

# Part 3: Multi-Head Attention
# ===========================

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

# Part 4: Point-wise Feed-Forward Network
# ======================================

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

# Part 5: Encoder Layer
# ====================

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

# Part 6: Decoder Layer
# ====================

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

# Part 7: Encoder
# =============

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
        
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
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
        
        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        return x  # (batch_size, input_seq_len, d_model)

# Part 8: Decoder
# =============

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
        
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate)
    
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
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x, block = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_self_attention'] = block['decoder_layer_self_attention']
            attention_weights[f'decoder_layer{i+1}_cross_attention'] = block['decoder_layer_cross_attention']
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# Part 9: Transformer
# =================

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
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, pe_input, rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                             target_vocab_size, pe_target, rate)
        
        self.final_layer = layers.Dense(target_vocab_size)
    
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
        # Keras models prefer explicit inputs over *args, **kwargs
        inp, tar = inputs
        
        # Create padding mask for encoder
        enc_padding_mask = create_padding_mask(inp)
        
        # Create look ahead mask for decoder self-attention
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        # Create padding mask for encoder-decoder attention
        dec_padding_mask = create_padding_mask(inp)
        
        # Encoder output
        enc_output = self.encoder(inp, training, enc_padding_mask)
        
        # Decoder output
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, combined_mask, dec_padding_mask)
        
        # Final output layer
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights

# Part 10: Creating Masks
# =====================

def create_padding_mask(seq):
    """
    Create a padding mask for the encoder attention.
    
    Args:
        seq: Input sequence
        
    Returns:
        mask: Padding mask
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # Add extra dimensions to add the padding to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for the decoder self-attention.
    
    Args:
        size: Size of the mask
        
    Returns:
        mask: Look-ahead mask
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (size, size)

# Part 11: Visualization Functions
# ==============================

def plot_positional_encoding(pe, title="Positional Encoding"):
    """
    Plot the positional encoding.
    
    Args:
        pe: Positional encoding
        title: Plot title
    """
    pe = pe[0]  # Extract from batch dimension
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pe.numpy(), cmap='RdBu')
    plt.xlabel('Depth')
    plt.ylabel('Position')
    plt.colorbar()
    plt.title(title)
    plt.show()

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
    attention = attention_weights[layer_name][0]  # First batch item
    
    # If there are multiple heads, average over the heads
    if len(attention.shape) == 4:
        attention = tf.reduce_mean(attention, axis=0)
    
    fig = plt.figure(figsize=(16, 8))
    plt.matshow(attention.numpy(), cmap='viridis', fignum=fig.number)
    
    fontdict = {'fontsize': 10}
    
    plt.xticks(range(len(sentence)), sentence, fontdict=fontdict, rotation=90)
    plt.yticks(range(len(result)), result, fontdict=fontdict)
    
    plt.xlabel('Input sequence')
    plt.ylabel('Output sequence')
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

# Try out the positional encoding
if __name__ == "__main__":
    # Example usage:
    pe = positional_encoding(50, 512)
    print(f"Positional encoding shape: {pe.shape}")
    
    # Plot the positional encoding
    plot_positional_encoding(pe)
    
    # Test scaled dot-product attention
    temp_q = tf.random.uniform((1, 3, 60))  # (batch_size, seq_len, d_model)
    temp_k = tf.random.uniform((1, 4, 60))  # (batch_size, seq_len, d_model)
    temp_v = tf.random.uniform((1, 4, 60))  # (batch_size, seq_len, d_model)
    
    output, attention_weights = scaled_dot_product_attention(temp_q, temp_k, temp_v)
    print(f"Scaled dot-product attention output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("Transformer implementation complete!")
