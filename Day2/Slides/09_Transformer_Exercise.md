# Transformer from Scratch Exercise

## Exercise Overview (10 minutes)

- Implement core components of the Transformer architecture
- Focus on understanding the building blocks and their interactions
- Complete missing parts in provided skeleton code
- Test the model on a simple translation task

## Components to Implement

1. Positional encoding
2. Scaled dot-product attention
3. Multi-head attention
4. Feed-forward network
5. Encoder and decoder layers

## Positional Encoding Implementation

```python
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
    # TODO: Implement the formula for positional encoding angles
    # Hint: The formula is angle_rates = 1 / (10000 ** (2i / d_model))
    
    pass  # Replace with your implementation

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
    # 1. Calculate angles using get_angles
    # 2. Apply sin to even indices and cos to odd indices
    # 3. Return the positional encoding with proper shape
    
    pass  # Replace with your implementation
```

## Scaled Dot-Product Attention Implementation

```python
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
    # TODO: Implement scaled dot-product attention
    # 1. Calculate dot product of query and key
    # 2. Scale by square root of key dimension
    # 3. Apply mask if provided
    # 4. Apply softmax to get attention weights
    # 5. Compute weighted values
    # 6. Return output and attention weights
    
    pass  # Replace with your implementation
```

## Multi-Head Attention Implementation

```python
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
        
        # TODO: Define the linear projection layers for Q, K, V
        # and the final output projection
        
        pass  # Replace with your implementation
    
    def split_heads(self, x, batch_size):
        """
        Split the input tensor into multiple heads.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # TODO: Implement split_heads
        # 1. Reshape x to (batch_size, seq_len, num_heads, depth)
        # 2. Transpose to (batch_size, num_heads, seq_len, depth)
        
        pass  # Replace with your implementation
    
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
        # TODO: Implement multi-head attention forward pass
        # 1. Apply linear projections to Q, K, V
        # 2. Split heads
        # 3. Apply scaled dot-product attention to each head
        # 4. Combine heads
        # 5. Apply final linear projection
        # 6. Return output and attention weights
        
        pass  # Replace with your implementation
```

## Feed-Forward Network Implementation

```python
def point_wise_feed_forward_network(d_model, dff):
    """
    Create a feed-forward network for transformer.
    
    Args:
        d_model: Model dimension
        dff: Hidden layer size
        
    Returns:
        Sequential model with two dense layers
    """
    # TODO: Implement the feed-forward network
    # 1. First dense layer with ReLU activation
    # 2. Second dense layer without activation
    
    pass  # Replace with your implementation
```

## Encoder Layer Implementation

```python
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
        
        # TODO: Initialize layers for the encoder
        # 1. Multi-head attention
        # 2. Feed-forward network
        # 3. Layer normalization (2 instances)
        # 4. Dropout layers (2 instances)
        
        pass  # Replace with your implementation
    
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
        # TODO: Implement the encoder forward pass
        # 1. Multi-head self-attention with residual connection and layer normalization
        # 2. Feed-forward network with residual connection and layer normalization
        
        pass  # Replace with your implementation
```

## Decoder Layer Implementation

```python
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
        
        # TODO: Initialize layers for the decoder
        # 1. Multi-head attention (self-attention)
        # 2. Multi-head attention (cross-attention)
        # 3. Feed-forward network
        # 4. Layer normalization (3 instances)
        # 5. Dropout layers (3 instances)
        
        pass  # Replace with your implementation
    
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
        # TODO: Implement the decoder forward pass
        # 1. Self-attention with look-ahead mask
        # 2. Cross-attention with encoder outputs
        # 3. Feed-forward network
        # 4. Collect attention weights for visualization
        
        pass  # Replace with your implementation
```

## Masking Functions Implementation

```python
def create_padding_mask(seq):
    """
    Create padding mask for transformer.
    
    Args:
        seq: Input sequence
        
    Returns:
        mask: Padding mask
    """
    # TODO: Implement padding mask
    # 1. Create a mask for padding tokens (value 0)
    # 2. Add dimensions for broadcasting with attention scores
    
    pass  # Replace with your implementation

def create_look_ahead_mask(size):
    """
    Create look-ahead mask for transformer.
    
    Args:
        size: Size of the sequence
        
    Returns:
        mask: Look-ahead mask
    """
    # TODO: Implement look-ahead mask
    # 1. Create a triangular mask to prevent looking at future tokens
    
    pass  # Replace with your implementation
```

## Testing Framework

- We'll provide a testing framework to verify each component
- Unit tests for individual functions and layers
- Small translation dataset for end-to-end testing
- Visualization tools for attention patterns

## Checkpoint Testing

```python
# Test positional encoding
def test_positional_encoding():
    pos_encoding = positional_encoding(50, 512)
    print(f"Positional encoding shape: {pos_encoding.shape}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.ylabel('Position')
    plt.xlabel('Depth')
    plt.title("Positional Encoding")
    plt.colorbar()
    plt.savefig('positional_encoding.png')
    
    return pos_encoding.shape == (1, 50, 512)

# Test scaled dot-product attention
def test_attention():
    # Create test inputs
    np.random.seed(42)
    temp_q = tf.random.uniform((2, 3, 4))  # (batch_size, seq_len, depth)
    temp_k = tf.random.uniform((2, 4, 4))  # (batch_size, seq_len, depth)
    temp_v = tf.random.uniform((2, 4, 6))  # (batch_size, seq_len, depth_value)
    
    # Run attention
    output, attention_weights = scaled_dot_product_attention(temp_q, temp_k, temp_v)
    
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Verify shapes
    return (output.shape == (2, 3, 6) and 
            attention_weights.shape == (2, 3, 4))
```

## Final Challenge: Creating a Simple Translation Model

- Once all components are implemented, combine them into a full model
- Train on a small translation dataset
- Visualize attention patterns
- Compare with sequence-to-sequence + attention model
