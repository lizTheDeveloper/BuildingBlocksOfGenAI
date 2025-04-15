"""
Tokenization Examples for LLM Section
Building Blocks of Generative AI Course - Day 2
"""

# Word-level tokenization example
def word_tokenization_example():
    text = "The quick brown fox jumps over the lazy dog."
    
    # Simple word tokenization by splitting on spaces
    tokens = text.split()
    print(f"Word tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Tokenize with punctuation handling
    import re
    tokens_with_punctuation = re.findall(r'\b\w+\b|[^\w\s]', text)
    print(f"Word tokens with punctuation: {tokens_with_punctuation}")
    print(f"Number of tokens: {len(tokens_with_punctuation)}")
    
    # Reconstruct the text
    reconstructed = ' '.join(tokens)
    print(f"Reconstructed text: {reconstructed}")
    
    # Note the loss of punctuation in simple word tokenization


# Character-level tokenization example
def character_tokenization_example():
    text = "Hello, world!"
    
    # Simple character tokenization
    char_tokens = list(text)
    print(f"Character tokens: {char_tokens}")
    print(f"Number of tokens: {len(char_tokens)}")
    
    # Map characters to integers
    char_to_id = {char: i for i, char in enumerate(sorted(set(char_tokens)))}
    id_to_char = {i: char for char, i in char_to_id.items()}
    
    # Convert to token IDs
    token_ids = [char_to_id[char] for char in char_tokens]
    print(f"Token IDs: {token_ids}")
    
    # Reconstruct the text
    reconstructed = ''.join([id_to_char[id] for id in token_ids])
    print(f"Reconstructed text: {reconstructed}")


# Subword tokenization example with BPE-like approach
def subword_tokenization_example():
    # This is a simplified demonstration of subword tokenization concepts
    # Not an actual implementation of BPE or WordPiece
    
    vocabulary = ["the", "quick", "brown", "fox", "jump", "##s", "over", "lazy", "dog"]
    
    def tokenize_with_subwords(text, vocab):
        # Simplified subword tokenization
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in vocab:
                # Word is in vocabulary
                tokens.append(word)
            else:
                # Try to split into subwords
                # This is highly simplified
                if word.endswith('s') and word[:-1] in vocab:
                    tokens.append(word[:-1])
                    tokens.append("##s")
                else:
                    # Out of vocabulary - would be handled differently
                    # in a real implementation
                    tokens.append("[UNK]")
        
        return tokens
    
    text = "The fox jumps over the dog."
    tokens = tokenize_with_subwords(text, vocabulary)
    print(f"Subword tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")


# Modern tokenizer example with Hugging Face tokenizers
def huggingface_tokenizer_example():
    try:
        from transformers import AutoTokenizer
        
        # Load a pre-trained tokenizer (GPT-2)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Sample text
        text = "The quick brown fox jumps over the lazy dog."
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"GPT-2 tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Convert to token IDs
        token_ids = tokenizer.encode(text)
        print(f"Token IDs: {token_ids}")
        
        # Decode back to text
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded text: {decoded_text}")
        
        # Example with out-of-vocabulary words
        text_with_rare_word = "The quick brown fox jumps over the antidisestablishmentarianism dog."
        tokens_rare = tokenizer.tokenize(text_with_rare_word)
        print(f"Tokens with rare word: {tokens_rare}")
        print(f"Number of tokens: {len(tokens_rare)}")
        
    except ImportError:
        print("Hugging Face transformers library not installed. Run: pip install transformers")


# Main execution for demonstration
if __name__ == "__main__":
    print("=== Word Tokenization ===")
    word_tokenization_example()
    
    print("\n=== Character Tokenization ===")
    character_tokenization_example()
    
    print("\n=== Subword Tokenization (Simplified) ===")
    subword_tokenization_example()
    
    print("\n=== Hugging Face Tokenizer (GPT-2) ===")
    huggingface_tokenizer_example()
