"""
Transformer Demo
Building Blocks of Generative AI Course - Day 2

This script demonstrates how to use the transformer model for machine translation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Import the transformer components
from transformer_model import Transformer

# Sample dataset parameters
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 8
DFF = 512
DROPOUT_RATE = 0.1

def load_and_preprocess_dataset():
    """
    Load and preprocess a sample translation dataset.
    
    Returns:
        train_dataset: Preprocessed training dataset
        val_dataset: Preprocessed validation dataset
        tokenizer_en: English tokenizer
        tokenizer_pt: Portuguese tokenizer
    """
    print("Loading dataset...")
    
    # For this demo, we'll use a small dataset from TensorFlow
    examples, metadata = tf.keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    
    # The dataset contains English-Spanish pairs
    path_to_file = examples + '/spa-eng/spa.txt'
    
    # Read the file
    with open(path_to_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    
    # Split each line into English and Spanish sentences
    pairs = [line.split('\t') for line in lines]
    
    # Keep only a subset for this demo
    pairs = pairs[:20000]
    
    # Extract English and Spanish sentences
    en_sentences = [pair[0] for pair in pairs]
    sp_sentences = [pair[1] for pair in pairs]
    
    # Add start and end tokens to target sentences
    sp_sentences = ['<start> ' + sentence + ' <end>' for sentence in sp_sentences]
    
    # Tokenize the sentences
    tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_sp = tf.keras.preprocessing.text.Tokenizer(filters='')
    
    tokenizer_en.fit_on_texts(en_sentences)
    tokenizer_sp.fit_on_texts(sp_sentences)
    
    # Convert text to sequences of integers
    en_sequences = tokenizer_en.texts_to_sequences(en_sentences)
    sp_sequences = tokenizer_sp.texts_to_sequences(sp_sentences)
    
    # Pad sequences
    en_padded = tf.keras.preprocessing.sequence.pad_sequences(
        en_sequences, maxlen=MAX_LENGTH, padding='post')
    sp_padded = tf.keras.preprocessing.sequence.pad_sequences(
        sp_sequences, maxlen=MAX_LENGTH, padding='post')
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((en_padded, sp_padded))
    
    # Split into training and validation sets
    train_size = int(len(pairs) * 0.8)
    val_size = len(pairs) - train_size
    
    train_dataset = dataset.take(train_size).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = dataset.skip(train_size).batch(BATCH_SIZE, drop_remainder=True)
    
    print(f"Train dataset size: {train_size}, Validation dataset size: {val_size}")
    print(f"English vocabulary size: {len(tokenizer_en.word_index) + 1}")
    print(f"Spanish vocabulary size: {len(tokenizer_sp.word_index) + 1}")
    
    return train_dataset, val_dataset, tokenizer_en, tokenizer_sp

def create_masks(inp, tar):
    """
    Create masks for transformer.
    
    Args:
        inp: Input tensor
        tar: Target tensor
        
    Returns:
        enc_padding_mask: Padding mask for encoder
        combined_mask: Combined look-ahead and padding mask for decoder self-attention
        dec_padding_mask: Padding mask for decoder cross-attention
    """
    # Encoder padding mask
    enc_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
    enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]
    
    # Decoder padding mask (for encoder-decoder attention)
    dec_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
    dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]
    
    # Look ahead mask (prevents decoder from looking at future tokens)
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((tf.shape(tar)[1], tf.shape(tar)[1])), -1, 0)
    
    # Decoder target padding mask
    dec_target_padding_mask = tf.cast(tf.math.equal(tar, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    
    # Combine look ahead mask and padding mask
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule with warmup for transformer.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    """
    Calculate loss with masking for padding.
    
    Args:
        real: Real labels
        pred: Predicted logits
        
    Returns:
        loss: Masked loss value
    """
    # Create mask for padding
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    
    # Apply sparse categorical crossentropy
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(real, pred)
    
    # Apply mask
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    # Average over non-padding positions
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def train_transformer():
    """
    Train the transformer model on a translation task.
    
    Returns:
        transformer: Trained transformer model
        train_loss: Training loss history
        train_accuracy: Training accuracy history
    """
    # Load and preprocess dataset
    train_dataset, val_dataset, tokenizer_en, tokenizer_sp = load_and_preprocess_dataset()
    
    # Calculate vocabulary sizes
    input_vocab_size = len(tokenizer_en.word_index) + 1
    target_vocab_size = len(tokenizer_sp.word_index) + 1
    
    # Initialize transformer
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=MAX_LENGTH,
        pe_target=MAX_LENGTH,
        rate=DROPOUT_RATE)
    
    # Initialize custom learning rate scheduler
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # Initialize metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    
    # Define training step
    @tf.function
    def train_step(inp, tar):
        # Split target into input and output
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        # Create masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions, _ = transformer(
                inputs=[inp, tar_inp],
                training=True,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask)
            
            # Calculate loss
            loss = loss_function(tar_real, predictions)
        
        # Calculate gradients
        gradients = tape.gradient(loss, transformer.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        # Update metrics
        train_loss(loss)
        train_accuracy(tar_real, predictions)
    
    # Training loop
    EPOCHS = 10
    
    for epoch in range(EPOCHS):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    
    return transformer, train_loss.result(), train_accuracy.result()

def evaluate(transformer, input_text, tokenizer_en, tokenizer_sp, max_length=MAX_LENGTH):
    """
    Evaluate the trained transformer on a new input.
    
    Args:
        transformer: Trained transformer model
        input_text: English text to translate
        tokenizer_en: English tokenizer
        tokenizer_sp: Spanish tokenizer
        max_length: Maximum sequence length
        
    Returns:
        translated_text: Translated Spanish text
        attention_weights: Attention weights for visualization
    """
    # Convert input text to sequence
    input_text = tokenizer_en.texts_to_sequences([input_text])
    input_text = tf.keras.preprocessing.sequence.pad_sequences(
        input_text, maxlen=max_length, padding='post')
    
    # Initialize decoder input with <start> token
    start_token = tokenizer_sp.word_index['<start>']
    end_token = tokenizer_sp.word_index['<end>']
    
    # Create decoder input with <start> token
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)
    
    # Initialize result
    result = []
    
    # Initialize storage for attention weights
    attention_weights = {}
    
    # Generate output sequence
    for i in range(max_length):
        # Create masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            input_text, output)
        
        # Forward pass
        predictions, attention_weights_step = transformer(
            inputs=[input_text, output],
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask)
        
        # Get the last token prediction
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        
        # Append prediction to output
        result.append(predicted_id.numpy()[0][0])
        output = tf.concat([output, predicted_id], axis=-1)
        
        # Store attention weights
        attention_weights[f'step_{i+1}'] = attention_weights_step
        
        # Break if end token is predicted
        if predicted_id == end_token:
            break
    
    # Convert result to text
    result_text = ' '.join([tokenizer_sp.index_word[id] for id in result
                           if id != start_token and id != end_token])
    
    return result_text, attention_weights

def plot_attention_weights(attention, input_text, result_text, layer=0):
    """
    Plot the attention weights.
    
    Args:
        attention: Attention weights from the transformer
        input_text: Input text (tokenized)
        result_text: Output text (tokenized)
        layer: Which layer's attention to plot (default: 0)
    """
    # Select the attention weights from a specific layer and head
    attn = tf.squeeze(attention[f'decoder_layer{layer+1}_cross_attention'], axis=0)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Split input and result into words
    input_words = input_text.split()
    result_words = result_text.split()
    
    # Add <start> and <end> tokens to result for visualization
    result_words = ['<start>'] + result_words + ['<end>']
    
    # Plot each attention head
    for head in range(NUM_HEADS):
        ax = fig.add_subplot(2, 4, head+1)
        
        # For time steps in the result sentence
        attention_map = attn[head, :len(result_words), :len(input_words)]
        
        # Plotting the attention weights
        im = ax.matshow(attention_map, cmap='viridis')
        
        # Set x and y ticks and labels
        ax.set_xticks(range(len(input_words)))
        ax.set_yticks(range(len(result_words)))
        
        ax.set_xticklabels(input_words, rotation=90)
        ax.set_yticklabels(result_words)
        
        ax.set_title(f'Head {head+1}')
    
    fig.tight_layout()
    plt.savefig('transformer_attention.png')
    plt.show()

def demo_translation(transformer, tokenizer_en, tokenizer_sp):
    """
    Demonstrate the trained transformer on example sentences.
    
    Args:
        transformer: Trained transformer model
        tokenizer_en: English tokenizer
        tokenizer_sp: Spanish tokenizer
    """
    # Example sentences to translate
    examples = [
        "Hello, how are you?",
        "What is your name?",
        "I like to learn languages.",
        "The weather is nice today."
    ]
    
    for example in examples:
        result, attention_weights = evaluate(
            transformer, example, tokenizer_en, tokenizer_sp)
        
        print(f"\nInput: {example}")
        print(f"Predicted translation: {result}")
        
        # Optionally plot attention weights for first example
        if example == examples[0]:
            plot_attention_weights(attention_weights, example, result)

if __name__ == "__main__":
    print("Transformer Demo for Machine Translation")
    print("=" * 40)
    
    # Note: Training a transformer from scratch can take a long time.
    # For a real demo, you might want to load pre-trained weights or use a smaller model.
    print("\nTraining transformer model (this may take a while)...")
    transformer, train_loss, train_accuracy = train_transformer()
    
    print("\nTraining complete!")
    print(f"Final training loss: {train_loss:.4f}")
    print(f"Final training accuracy: {train_accuracy:.4f}")
    
    # Get tokenizers from training function
    _, _, tokenizer_en, tokenizer_sp = load_and_preprocess_dataset()
    
    print("\nDemonstrating translation with the trained model...")
    demo_translation(transformer, tokenizer_en, tokenizer_sp)
