# Introduction to Large Language Models

## Evolution of Language Models (15 minutes)

- From rule-based systems to statistical models
- N-gram models: capturing local word probabilities
- Word embeddings (Word2Vec, GloVe): capturing semantic relationships
- RNN-based models: handling sequential dependencies
- Attention mechanisms: focusing on relevant context
- Transformer revolution: parallelization and context windows
- Modern LLMs: GPT, PaLM, Llama, Claude

## Understanding Language Model Architecture

- Conceptual building blocks:
  - Tokenization: converting text to numerical tokens
  - Embeddings: representing tokens as vectors
  - Self-attention: capturing relationships between tokens
  - Feed-forward networks: processing token representations
  - Output layer: predicting next token probabilities

## Tokenization and Vocabulary

```python
# Example of tokenization with SentencePiece/BPE
import tiktoken

# GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode text to tokens
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")

# Subword tokenization example
print("Subword breakdown:")
for token in tokens:
    print(f"Token {token}: '{tokenizer.decode([token])}'")
```

## The Autoregressive Language Modeling Task

- Next-token prediction as the core training objective
- Maximizing log-likelihood of next token given context
- During inference:
  - Start with a prompt
  - Sample from output distribution or use greedy decoding
  - Add generated token to context window
  - Repeat until completion

## Decoding Strategies

```python
def sample_from_model(model, prompt, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
    """
    Sample from a language model with various decoding strategies
    """
    input_ids = tokenizer.encode(prompt)
    context = input_ids.copy()
    
    for _ in range(max_length):
        # Get model predictions
        logits = model(context)
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = tf.argsort(next_token_logits, direction='DESCENDING')[0, top_k:]
            next_token_logits = tf.scatter_nd(
                indices_to_remove[:, tf.newaxis], 
                tf.ones_like(indices_to_remove, dtype=tf.float32) * -float('inf'),
                tf.shape(next_token_logits)
            )
        
        # Apply top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits = tf.sort(next_token_logits, direction='DESCENDING')
            sorted_indices = tf.argsort(next_token_logits, direction='DESCENDING')
            cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            indices_to_remove = tf.scatter_nd(
                tf.where(sorted_indices_to_remove), 
                tf.ones_like(tf.where(sorted_indices_to_remove)[:, 0], dtype=tf.float32) * -float('inf'),
                tf.shape(next_token_logits)
            )
            next_token_logits = indices_to_remove
        
        # Sample from the filtered distribution
        probs = tf.nn.softmax(next_token_logits)
        next_token = tf.random.categorical(probs, num_samples=1)
        
        # Add the sampled token to the context
        context.append(next_token.numpy()[0][0])
        
        # Stop if we generate an EOS token
        if next_token == tokenizer.encode('<|endoftext|>')[0]:
            break
            
    return tokenizer.decode(context)
```

## Training vs. Inference in LLMs

- Training: parallel processing of sequences
- Inference: sequential generation one token at a time
- Inference performance optimizations:
  - KV-caching: storing key-value pairs from previous steps
  - Quantization: using lower precision for weights
  - Batching: processing multiple sequences together
  - Beam search: exploring multiple generation paths

## The Generative AI Project Lifecycle

- Traditional ML vs. Generative AI approach
- Data preparation: representative, diverse, high-quality corpora
- Pre-training: learning general language patterns
- Fine-tuning: adapting to specific tasks or domains
- Evaluation: perplexity, task-specific metrics, human evaluation
- Deployment and monitoring: latency, quality, safety

## From Transformers to Multi-Modal Models

- Text as the foundation of modern generative AI
- Extending to other modalities:
  - Image-text models (CLIP, DALL-E)
  - Video-text models
  - Audio-text models
  - Multi-modal embeddings and transformers

## Transition to Prompt Engineering

- LLMs capabilities go beyond just language modeling
- Complex tasks can be performed with carefully crafted prompts
- Next section: exploring prompt engineering techniques
