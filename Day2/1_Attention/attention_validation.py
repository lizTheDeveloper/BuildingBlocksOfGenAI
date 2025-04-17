"""
Attention Implementation Validation
Building Blocks of Generative AI Course - Day 2

This file contains utility functions to help validate and visualize
the scaled dot-product attention implementation from the exercise.
Students can use this to check if their implementation is correct.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Import the student's implementation here
# If they've created attention_layer.py correctly, this will work
try:
    from attention_layer import ScaledDotProductAttention
    student_implementation_available = True
except ImportError:
    print("Warning: Could not import ScaledDotProductAttention from attention_layer.py.")
    print("Make sure you've implemented the attention mechanism correctly.")
    student_implementation_available = False

# Provide a reference implementation for comparison
class ReferenceScaledDotProductAttention(layers.Layer):
    """
    Reference implementation of Scaled Dot-Product Attention for validation.
    """
    def __init__(self):
        super(ReferenceScaledDotProductAttention, self).__init__()
    
    def call(self, query, key, value, mask=None):
        """
        Apply scaled dot-product attention mechanism.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, depth)
            key: Key tensor (batch_size, seq_len_k, depth)
            value: Value tensor (batch_size, seq_len_