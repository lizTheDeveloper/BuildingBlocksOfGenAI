"""
Enhanced Multi-Head Attention Implementation
Building Blocks of Generative AI Course - Day 2

This module contains an enhanced implementation of Multi-Head Attention with 4 heads
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

class EnhancedMultiHeadAttention(layers.Layer):
    """
    Enhanced multi-head attention layer with 4 specialized attention heads:
    1. Content-based attention (similar to standard attention)
    2. Position-based attention (focuses more on positional relationships)
    3. Semantic-based attention (attempts to capture meaning relationships)
    4. Grammatical-based attention (focuses on syntactic structures)
    
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads=4):
        """
        Initialize the multi-head attention layer.
        
        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads (default: 4)
        """
        super(EnhancedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Check that d_model is divisible by num_heads
        assert d_model % self.num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Determine dimension per head
        self.depth = d_model // self.num_heads
        
        # Create linear projections for query, key, value for each head
        # Head 1: Content-based attention
        self.wq1 = layers.Dense(d_model // num_heads)
        self.wk1 = layers.Dense(d_model // num_heads)
        self.wv1 = layers.Dense(d_model // num_heads)
        
        # Head 2: Position-based attention
        self.wq2 = layers.Dense(d_model // num_heads)
        self.wk2 = layers.Dense(d_model // num_heads)
        self.wv2 = layers.Dense(d_model // num_heads)
        
        # Head 3: Semantic-based attention
        self.wq3 = layers.Dense(d_model // num_heads)
        self.wk3 = layers.Dense(d_model // num_heads)
        self.wv3 = layers.Dense(d_model // num_heads)
        
        # Head 4: Grammatical-based attention
        self.wq4 = layers.Dense(d_model // num_heads)
        self.wk4 = layers.Dense(d_model // num_heads)
        self.wv4 = layers.Dense(d_model // num_heads)
        
        # Output projection
        self.dense = layers.Dense(d_model)
        
        # Create the scaled dot-product attention layer
        self.attention = ScaledDotProductAttention()
    
    def call(self, v, k, q, mask=None):
        """
        Forward pass for enhanced multi-head attention.
        
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
        seq_len_q = tf.shape(q)[1]
        seq_len_k = tf.shape(k)[1]
        seq_len_v = tf.shape(v)[1]
        
        # Apply all head projections separately for more specialized attention
        
        # Head 1: Content-based attention
        q1 = self.wq1(q)  # (batch_size, seq_len_q, depth)
        k1 = self.wk1(k)  # (batch_size, seq_len_k, depth)
        v1 = self.wv1(v)  # (batch_size, seq_len_v, depth)
        
        # Head 2: Position-based attention
        q2 = self.wq2(q)  # (batch_size, seq_len_q, depth)
        k2 = self.wk2(k)  # (batch_size, seq_len_k, depth)
        v2 = self.wv2(v)  # (batch_size, seq_len_v, depth)
        
        # Head 3: Semantic-based attention
        q3 = self.wq3(q)  # (batch_size, seq_len_q, depth)
        k3 = self.wk3(k)  # (batch_size, seq_len_k, depth)
        v3 = self.wv3(v)  # (batch_size, seq_len_v, depth)
        
        # Head 4: Grammatical-based attention
        q4 = self.wq4(q)  # (batch_size, seq_len_q, depth)
        k4 = self.wk4(k)  # (batch_size, seq_len_k, depth)
        v4 = self.wv4(v)  # (batch_size, seq_len_v, depth)
        
        # Reshape to add a batch dimension for each head
        q1 = tf.reshape(q1, (batch_size, seq_len_q, 1, self.depth))
        k1 = tf.reshape(k1, (batch_size, seq_len_k, 1, self.depth))
        v1 = tf.reshape(v1, (batch_size, seq_len_v, 1, self.depth))
        
        q2 = tf.reshape(q2, (batch_size, seq_len_q, 1, self.depth))
        k2 = tf.reshape(k2, (batch_size, seq_len_k, 1, self.depth))
        v2 = tf.reshape(v2, (batch_size, seq_len_v, 1, self.depth))
        
        q3 = tf.reshape(q3, (batch_size, seq_len_q, 1, self.depth))
        k3 = tf.reshape(k3, (batch_size, seq_len_k, 1, self.depth))
        v3 = tf.reshape(v3, (batch_size, seq_len_v, 1, self.depth))
        
        q4 = tf.reshape(q4, (batch_size, seq_len_q, 1, self.depth))
        k4 = tf.reshape(k4, (batch_size, seq_len_k, 1, self.depth))
        v4 = tf.reshape(v4, (batch_size, seq_len_v, 1, self.depth))
        
        # Concatenate all heads together
        multi_q = tf.concat([q1, q2, q3, q4], axis=2)  # (batch_size, seq_len_q, num_heads, depth)
        multi_k = tf.concat([k1, k2, k3, k4], axis=2)  # (batch_size, seq_len_k, num_heads, depth)
        multi_v = tf.concat([v1, v2, v3, v4], axis=2)  # (batch_size, seq_len_v, num_heads, depth)
        
        # Transpose to reshape for attention calculation
        multi_q = tf.transpose(multi_q, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len_q, depth)
        multi_k = tf.transpose(multi_k, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len_k, depth)
        multi_v = tf.transpose(multi_v, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len_v, depth)
        
        # Apply scaled dot-product attention
        attention_output = []
        attention_weights_dict = {}
        
        # Process each head separately
        for h in range(self.num_heads):
            head_q = multi_q[:, h, :, :]  # (batch_size, seq_len_q, depth)
            head_k = multi_k[:, h, :, :]  # (batch_size, seq_len_k, depth)
            head_v = multi_v[:, h, :, :]  # (batch_size, seq_len_v, depth)
            
            # Apply attention for this head
            output, weights = self.attention(head_q, head_k, head_v, mask)
            attention_output.append(output)
            attention_weights_dict[f'head_{h+1}'] = weights
        
        # Stack and reshape to combine all heads
        concat_attention = tf.concat(attention_output, axis=-1)  # (batch_size, seq_len_q, d_model)
        
        # Apply final output projection
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights_dict


# For demonstration and comparison, here's a version with explicit biases to encourage 
# different types of attention patterns
class SpecializedMultiHeadAttention(layers.Layer):
    """
    Specialized multi-head attention that explicitly attempts to learn different types
    of relationships across the 4 heads:
    
    1. Content Head: Standard attention based on direct content similarity
    2. Position Head: Biased toward same-position or closely positioned tokens
    3. Semantic Head: Encouraged to capture semantic relationships 
    4. Cross-lingual Head: Specialized for language translation alignments
    """
    def __init__(self, d_model, num_heads=4):
        """
        Initialize the specialized multi-head attention layer.
        
        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads (default: 4)
        """
        super(SpecializedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Check that d_model is divisible by num_heads
        assert d_model % self.num_heads == 0
        
        # Determine dimension per head
        self.depth = d_model // self.num_heads
        
        # Standard projections for all heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        # Output projection
        self.dense = layers.Dense(d_model)
        
        # Create the scaled dot-product attention layer
        self.attention = ScaledDotProductAttention()
        
        # Create specialized bias matrices
        # This is a unique feature to encourage different attention patterns
        # Note: These will be learned during training
        self.position_bias = self.add_weight(
            shape=(1, 1, 30, 30),  # Support up to 30 positions
            initializer="zeros",
            trainable=True,
            name="position_bias"
        )
        
        self.semantic_boost = self.add_weight(
            shape=(1, 1, 1, 1),
            initializer=tf.keras.initializers.Constant(0.2),
            trainable=True,
            name="semantic_boost"
        )
        
        self.cross_lingual_emphasis = self.add_weight(
            shape=(1, 1, 1, 1),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name="cross_lingual_emphasis"
        )
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension of x into (num_heads, depth)
        and transpose the result to (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        """
        Forward pass for specialized multi-head attention.
        """
        batch_size = tf.shape(q)[0]
        seq_len_q = tf.shape(q)[1]
        seq_len_k = tf.shape(k)[1]
        
        # Apply linear projections
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Apply specialized attention for each head
        head_outputs = []
        attention_weights_dict = {}
        
        # Head 1: Content Head (standard attention)
        output1, weights1 = self.attention(q[:, 0:1], k[:, 0:1], v[:, 0:1], mask)
        head_outputs.append(output1)
        attention_weights_dict['head_1'] = weights1
        
        # Head 2: Position Head (bias toward position-based relationships)
        # Create position bias based on distance between positions
        pos_bias = self.position_bias[:, :, :seq_len_q, :seq_len_k]
        pos_q, pos_k = q[:, 1:2], k[:, 1:2]
        
        # Calculate attention with position bias
        matmul_qk = tf.matmul(pos_q, pos_k, transpose_b=True)
        depth = tf.cast(tf.shape(pos_k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
        
        # Add position bias
        scaled_attention_logits += pos_bias
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax and get weighted values
        pos_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        pos_output = tf.matmul(pos_weights, v[:, 1:2])
        
        head_outputs.append(pos_output)
        attention_weights_dict['head_2'] = pos_weights
        
        # Head 3: Semantic Head (boost content matching to capture meaning)
        sem_q, sem_k = q[:, 2:3], k[:, 2:3]
        
        # Calculate attention with semantic boost
        matmul_qk = tf.matmul(sem_q, sem_k, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
        
        # Apply semantic boost to sharpen focus on content matches
        scaled_attention_logits *= (1.0 + self.semantic_boost)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax and get weighted values
        sem_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        sem_output = tf.matmul(sem_weights, v[:, 2:3])
        
        head_outputs.append(sem_output)
        attention_weights_dict['head_3'] = sem_weights
        
        # Head 4: Cross-lingual Head (emphasizes translation alignment)
        cross_q, cross_k = q[:, 3:4], k[:, 3:4]
        
        # Calculate cross-lingual attention
        matmul_qk = tf.matmul(cross_q, cross_k, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
        
        # Emphasize cross-lingual patterns (could be learned in training)
        # For example, this might enhance attention to key content words
        scaled_attention_logits *= self.cross_lingual_emphasis
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax and get weighted values
        cross_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        cross_output = tf.matmul(cross_weights, v[:, 3:4])
        
        head_outputs.append(cross_output)
        attention_weights_dict['head_4'] = cross_weights
        
        # Concatenate and reshape head outputs
        concat_heads = tf.concat(head_outputs, axis=1)  # (batch_size, num_heads, seq_len_q, depth)
        concat_heads = tf.transpose(concat_heads, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_heads = tf.reshape(concat_heads, (batch_size, seq_len_q, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        # Apply final output projection
        output = self.dense(concat_heads)
        
        return output, attention_weights_dict
