# Attention Mechanism Exercise

## Exercise Overview (10 minutes)

- Implement a sequence-to-sequence model with attention
- Focus: machine translation with a small dataset
- Key components to implement:
  - Bahdanau attention mechanism
  - Encoder-decoder architecture
  - Training and inference loops

## Exercise Dataset

```python
# Sample dataset - English to Spanish translation
english_texts = [
    "hello",
    "thank you",
    "how are you",
    "goodbye",
    "my name is",
    "what is your name",
    "nice to meet you",
    "where are you from",
    "I am from the United States",
    "do you speak English"
]

spanish_texts = [
    "hola",
    "gracias",
    "cómo estás",
    "adiós",
    "me llamo",
    "cómo te llamas",
    "encantado de conocerte",
    "de dónde eres",
    "soy de los Estados Unidos",
    "hablas inglés"
]

# Preprocess data
def preprocess_text(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, padding='post')
    return padded_sequences, tokenizer
```

## Bahdanau Attention Implementation

```python
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # TODO: Define the layers for the attention mechanism
        # 1. A dense layer for processing the encoder outputs (W_a)
        # 2. A dense layer for processing the decoder state (U_a)
        # 3. A dense layer with 1 unit for producing attention scores (v_a)
        
    def call(self, query, values):
        # TODO: Implement the attention mechanism
        # 1. Expand dimensions of query to match values for broadcasting
        # 2. Calculate the attention scores using the layers defined in __init__
        # 3. Apply softmax to get attention weights
        # 4. Create the context vector as a weighted sum of values
        # 5. Return the context vector and attention weights
        
        # Placeholder implementation - replace with your code
        batch_size = tf.shape(values)[0]
        max_length = tf.shape(values)[1]
        
        # Dummy implementation
        attention_weights = tf.ones((batch_size, max_length, 1)) / max_length
        context_vector = tf.reduce_sum(values * attention_weights, axis=1)
        
        return context_vector, attention_weights
```

## Encoder Implementation Requirements

- GRU-based encoder
- Process input sequence to create outputs and final state
- Return both outputs (for attention) and state (for decoder init)

```python
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        
        # TODO: Define the encoder layers
        # 1. An embedding layer
        # 2. A GRU layer
        
    def call(self, x, hidden):
        # TODO: Implement the encoder forward pass
        # 1. Pass input through the embedding layer
        # 2. Pass embedded input through the GRU
        # 3. Return the outputs and final state
```

## Decoder Implementation Requirements

- GRU-based decoder with attention
- Process previous token and attention context
- Return predictions for the next token

```python
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        # TODO: Define the decoder layers
        # 1. An embedding layer
        # 2. A GRU layer
        # 3. Dense layers for output
        # 4. Attention mechanism
        
    def call(self, x, hidden, enc_output):
        # TODO: Implement the decoder forward pass with attention
        # 1. Pass input through the embedding layer
        # 2. Use the attention mechanism to compute context vector
        # 3. Concatenate context vector with the embedding output
        # 4. Pass through GRU
        # 5. Generate output prediction
        # 6. Return output, state, and attention weights
```

## Training Loop Implementation

```python
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    
    with tf.GradientTape() as tape:
        # Encoder forward pass
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        
        # Initialize decoder hidden state with encoder final state
        dec_hidden = enc_hidden
        
        # First decoder input is the start token
        dec_input = tf.expand_dims(
            [targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        
        # Teacher forcing - feeding target as next input
        for t in range(1, targ.shape[1]):
            # Pass encoder output, decoder hidden state and input to decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            
            # Calculate loss
            loss += loss_function(targ[:, t], predictions)
            
            # Use teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)
    
    # Calculate gradients and apply them
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return loss / targ.shape[1]
```

## Inference Function and Visualization

- Implement the translation function
- Visualize attention weights as a heatmap
- Decode output tokens to text

```python
def translate(sentence, encoder, decoder, inp_tokenizer, targ_tokenizer, max_length):
    # Tokenize input
    inputs = inp_tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length, padding='post')
    
    # Initialize result and attention matrix
    result = ''
    attention_plot = np.zeros((max_length, max_length))
    
    # Initialize encoder hidden state
    hidden = [tf.zeros((1, units))]
    
    # Get encoder output and final state
    enc_out, enc_hidden = encoder(inputs, hidden)
    
    # Set decoder hidden state to encoder final state
    dec_hidden = enc_hidden
    
    # Start with <start> token
    dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)
    
    # Decode one token at a time
    for t in range(max_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # Store attention weights
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        
        # Get predicted ID
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        # Convert to word
        if predicted_id == targ_tokenizer.word_index['<end>']:
            return result, attention_plot
            
        result += targ_tokenizer.index_word[predicted_id] + ' '
        
        # Feed prediction as next input
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return result, attention_plot
```

## Tasks to Complete

1. Implement the Bahdanau attention mechanism
2. Complete the encoder class
3. Complete the decoder class
4. Test the translation model on sample sentences
5. Visualize attention weights to see alignment

## Bonus Challenges

- Implement different attention mechanisms (Luong/dot-product)
- Add bidirectional encoder
- Implement beam search decoding
- Compare model performance with and without attention
