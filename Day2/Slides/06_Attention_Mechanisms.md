# Attention Mechanisms: From RNNs to Transformers

## Introduction to Attention (15 minutes)

- The attention hypothesis: not all inputs are equally important
- Inspiration from human cognitive attention
- Evolution in NLP: from fixed context to dynamic focus
- Attention as a mechanism for alignment in sequence-to-sequence models

## The Problem with Basic RNNs/LSTMs

- Information bottleneck in encoder-decoder architectures
- Difficulty handling long-range dependencies
- Fixed-size context vector limitations
- Vanishing gradients over long sequences

## Bahdanau Attention (2015)

```python
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # Layers for attention mechanism
        self.W1 = tf.keras.layers.Dense(units)  # For encoder outputs
        self.W2 = tf.keras.layers.Dense(units)  # For decoder state
        self.V = tf.keras.layers.Dense(1)       # For attention scores
    
    def call(self, query, values):
        # query: decoder hidden state (batch_size, hidden_size)
        # values: encoder outputs (batch_size, max_length, hidden_size)
        
        # Add time axis to query for broadcasting
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Calculate attention score using a small feed-forward network
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Create context vector as weighted sum of values
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
```

## Luong Attention (Multiplicative Attention)

- Alternative to Bahdanau's additive attention
- Types of score functions:
  - Dot product: score(ht, hs) = ht · hs
  - General: score(ht, hs) = ht · Wa · hs
  - Concat: score(ht, hs) = va · tanh(Wa[ht; hs])
- Computationally more efficient than Bahdanau attention

## Visualization of Attention in NMT

- Attention weights create alignment between source and target
- Visual patterns in attention matrices reveal:
  - Word order differences between languages
  - Complex phrase alignments
  - Multi-word expressions

## Self-Attention: The Key Innovation

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate scaled dot-product attention
    
    Args:
        q: query shape (batch_size, seq_len_q, depth)
        k: key shape (batch_size, seq_len_k, depth)
        v: value shape (batch_size, seq_len_v, depth_v)
        mask: Optional mask shape (batch_size, seq_len_q, seq_len_k)
        
    Returns:
        output, attention_weights
    """
    # Calculate attention scores
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Scale attention scores
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # Apply mask (if provided)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Calculate attention weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # Calculate output as weighted sum of values
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights
```

## Multi-Head Attention

- Running attention in parallel with different projections
- Benefits:
  - Attend to information from different representation subspaces
  - Capture different aspects of similarity (syntax, semantics)
  - Improved representational power

## Types of Attention Masks

- Padding mask: ignores padding tokens (for variable-length sequences)
- Causal/look-ahead mask: prevents attending to future tokens
- Combined mask: applies both padding and causal masking

```python
# Padding mask for encoder
def create_padding_mask(seq):
    # Create mask where 0 tokens are masked out
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # Add dimensions for broadcasting in attention calculations
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# Look-ahead mask for decoder
def create_look_ahead_mask(size):
    # Create a triangular mask to prevent attending to future tokens
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
```

## Attention in Modern Systems

- Core building block of all modern NLP architectures
- Extensions beyond language:
  - Cross-attention: connecting different modalities
  - Sparse attention: selective rather than global
  - Efficient attention: linear complexity alternatives

## From Attention to Transformer

- Attention is the building block, not the full architecture
- Next: How Transformers use attention within a complete architecture
