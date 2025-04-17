"""
Attention Mechanism Exercise - Main Script
Building Blocks of Generative AI Course - Day 2

This script brings together the components of the attention-based
sequence-to-sequence model and demonstrates its usage for translation.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Import modules from our project
from data_utils import load_sample_translation_data, create_dataset
from attention_layer import ScaledDotProductAttention
from seq2seq_model import Encoder, Decoder

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load sample data
data = load_sample_translation_data()

# Model parameters
BATCH_SIZE = 4
embedding_dim = 128
units = 256
epochs = 10

# Create training dataset
dataset = create_dataset(
    data['eng_sequences'], 
    data['fr_sequences'],
    batch_size=BATCH_SIZE
)

# Function to visualize attention weights
def plot_attention(attention, input_sentence, predicted_sentence):
    """
    Visualize attention weights.
    
    Args:
        attention: Attention weights (max_length_target, max_length_source)
        input_sentence: Input sentence as a list of words
        predicted_sentence: Predicted sentence as a list of words
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    
    attention = attention[:len(predicted_sentence), :len(input_sentence)]
    
    cax = ax.matshow(attention, cmap='viridis')
    
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    plt.show()

def train_model():
    """
    Initialize and train the sequence-to-sequence model with attention.
    
    Returns:
        encoder: Trained encoder model
        decoder: Trained decoder model
    """
    # Initialize the encoder and decoder
    encoder = Encoder(
        data['eng_vocab_size'], 
        embedding_dim, 
        units, 
        BATCH_SIZE
    )
    
    decoder = Decoder(
        data['fr_vocab_size'], 
        embedding_dim, 
        units, 
        BATCH_SIZE
    )
    
    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    
    def loss_function(real, pred):
        # Mask padded parts of the target sequence
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_mean(loss_)
    
    # Define training step function
    @tf.function
    def train_step(inp, targ):
        loss = 0
        
        with tf.GradientTape() as tape:
            # Initialize encoder hidden state
            enc_hidden = encoder.initialize_hidden_state()
            
            # Encoder call
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            
            # Initialize decoder hidden state with encoder final state
            dec_hidden = enc_hidden
            
            # Teacher forcing - feeding the target as the next input
            # First token is <start>
            dec_input = tf.expand_dims([data['fr_tokenizer'].word_index['<start>']] * BATCH_SIZE, 1)
            
            # Process target sequence token by token
            for t in range(1, targ.shape[1]):
                # Pass encoder output, decoder hidden state and input to decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                
                # Calculate loss
                loss += loss_function(targ[:, t], predictions)
                
                # Using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        
        # Calculate gradients and apply them
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        return loss
    
    # Training loop
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset.take(data['eng_sequences'].shape[0] // BATCH_SIZE)):
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss
            
            if batch % 2 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
        print(f'Epoch {epoch+1} Loss {total_loss:.4f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
    
    return encoder, decoder

def translate(sentence, encoder, decoder):
    """
    Translate an English sentence to French using the trained model.
    
    Args:
        sentence: English sentence as a string
        encoder: Trained encoder model
        decoder: Trained decoder model
        
    Returns:
        result: Translated sentence
        attention_weights: Attention weights for visualization
    """
    # Tokenize the input sentence
    inputs = [data['eng_tokenizer'].word_index[word] for word in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=data['eng_max_length'], padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    # Initialize result and attention weights
    result = ''
    attention_plot = np.zeros((data['fr_max_length'], data['eng_max_length']))
    
    # Initialize hidden state
    hidden = tf.zeros((1, units))
    
    # Get encoder output and hidden state
    enc_output, enc_hidden = encoder(inputs, hidden)
    
    # Set decoder hidden state to encoder final state
    dec_hidden = enc_hidden
    
    # First decoder input is the <start> token
    dec_input = tf.expand_dims([data['fr_tokenizer'].word_index['<start>']], 0)
    
    # Loop until <end> token or maximum length
    for t in range(data['fr_max_length']):
        # Get decoder output, hidden state, and attention weights
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
        
        # Store attention weights for visualization
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        
        # Get the predicted token
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        # Convert token ID to word
        if predicted_id == 0:  # padding token
            continue
            
        word = ''
        for w, i in data['fr_tokenizer'].word_index.items():
            if i == predicted_id:
                word = w
                break
        
        # Append word to result
        if word == '<end>':
            return result, attention_plot
            
        result += word + ' '
        
        # Next input is the predicted token
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result, attention_plot

def test_translation(encoder, decoder):
    """
    Test the translation model on sample sentences.
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
    """
    for i, (eng, fr) in enumerate(zip(data['eng_texts'][:3], data['fr_texts'][:3])):
        print(f"Input: {eng}")
        result, attention_plot = translate(eng, encoder, decoder)
        print(f"Predicted translation: {result}")
        print(f"Actual translation: {fr}")
        print()
        
        # Plot attention weights
        attention_plot = attention_plot[:len(result.split(' ')), :len(eng.split(' '))]
        plot_attention(attention_plot, eng.split(' '), result.split(' '))

if __name__ == "__main__":
    print("This exercise implements a sequence-to-sequence model with attention for translation.")
    print("To complete it, you need to fill in the TODOs in the following files:")
    print("1. attention_layer.py: Implement the Scaled Dot Product attention mechanism")
    print("2. seq2seq_model.py: Implement the encoder and decoder models")
    print("Once completed, run this script to train and test the model.")
    
    # Uncomment these lines once you've completed the TODOs
    # print("\nTraining the model...")
    # encoder, decoder = train_model()
    # 
    # print("\nTesting the model...")
    # test_translation(encoder, decoder)
