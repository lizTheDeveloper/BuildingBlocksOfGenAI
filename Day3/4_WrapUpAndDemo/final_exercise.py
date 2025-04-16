"""
Final Exercise: Build Your Own Mini-LLM Application
-----------------------------------------------
This exercise allows students to apply concepts learned throughout the course
by building a simple but functional LLM application in a limited time frame.
"""

def exercise_instructions():
    """
    Print instructions for the final exercise
    """
    print("=" * 70)
    print("FINAL EXERCISE: BUILD YOUR OWN MINI-LLM APPLICATION")
    print("=" * 70)
    
    print("\nIn this final exercise, you'll build a simple but functional LLM application")
    print("that demonstrates your understanding of the key concepts from the course.")
    print("You'll work in small groups (2-3 people) for 45 minutes, and then each")
    print("group will present their solution for 2-3 minutes.")
    
    print("\n" + "-" * 70)
    print("EXERCISE OPTIONS (CHOOSE ONE)")
    print("-" * 70)
    
    print("\nOption 1: 📝 Specialized Text Generator")
    print("----------------------------------------")
    print("Create an application that generates specialized text in a specific style,")
    print("format, or domain. Examples include:")
    print("- Marketing copy generator")
    print("- Poetry or lyrics in a specific style")
    print("- Technical documentation generator")
    print("- Email response generator for specific contexts")
    
    print("\nOption 2: 🔄 Text Transformation Tool")
    print("-------------------------------------")
    print("Build a tool that transforms text from one style or format to another.")
    print("Examples include:")
    print("- Text simplifier (complex → simple language)")
    print("- Tone transformer (casual → formal or vice versa)")
    print("- Technical jargon explainer")
    print("- Content summarizer with adjustable detail levels")
    
    print("\nOption 3: 💬 Specialized Chatbot")
    print("--------------------------------")
    print("Develop a focused chatbot for a specific domain or purpose.")
    print("Examples include:")
    print("- Educational tutor in a specific subject")
    print("- Character-based chatbot (historical figure, fictional character)")
    print("- Process guide (e.g., cooking assistant, DIY helper)")
    print("- Customer support bot for a specific product type")
    
    print("\n" + "-" * 70)
    print("IMPLEMENTATION REQUIREMENTS")
    print("-" * 70)
    
    print("\nYour solution should include:")
    
    print("\n1. Model Selection and Setup")
    print("   - Choose an appropriate pre-trained model")
    print("   - Configure the model for your specific task")
    print("   - Document your choice and reasoning")
    
    print("\n2. Prompt Engineering OR Fine-tuning")
    print("   - Either craft effective prompts for zero/few-shot learning")
    print("   - OR implement a simple fine-tuning approach")
    print("   - Document your approach and design decisions")
    
    print("\n3. User Interface")
    print("   - Create a simple interface for interacting with your application")
    print("   - This can be a command-line interface, simple web UI, or notebook")
    print("   - Focus on usability within the time constraints")
    
    print("\n4. Evaluation Method")
    print("   - Define how you'll evaluate the quality of outputs")
    print("   - Implement at least one automated metric or systematic evaluation approach")
    print("   - Provide example outputs with your evaluation")
    
    print("\n" + "-" * 70)
    print("SUGGESTED WORKFLOW")
    print("-" * 70)
    
    print("\n1. Planning Phase (10 minutes)")
    print("   - Choose your option and specific application focus")
    print("   - Define the core functionality and user experience")
    print("   - Select your model and approach (prompt engineering or fine-tuning)")
    print("   - Divide tasks among team members")
    
    print("\n2. Development Phase (25 minutes)")
    print("   - Set up the model and implement core functionality")
    print("   - Create the user interface")
    print("   - Develop evaluation method")
    print("   - Test with sample inputs")
    
    print("\n3. Refinement Phase (10 minutes)")
    print("   - Improve prompts or model configuration based on testing")
    print("   - Prepare for demonstration")
    print("   - Document key aspects of your implementation")
    
    print("\n" + "-" * 70)
    print("RESOURCES")
    print("-" * 70)
    
    print("\nModels:")
    print("- Hugging Face models like T5, BERT, GPT-2")
    print("- Smaller versions of models for quick experimentation")
    print("- Consider using pipelines for simplicity")
    
    print("\nLibraries:")
    print("- Transformers (Hugging Face)")
    print("- PEFT for parameter-efficient fine-tuning")
    print("- Gradio or Streamlit for simple web UIs")
    print("- NLTK or spaCy for text processing")
    
    print("\nEvaluation:")
    print("- ROUGE, BLEU for generation tasks")
    print("- Custom metrics specific to your application")
    print("- Human evaluation rubrics")
    
    print("\n" + "-" * 70)
    print("STARTER CODE")
    print("-" * 70)

def starter_code_option1():
    """
    Provide starter code for Option 1: Specialized Text Generator
    """
    print("\nStarter Code for Option 1: Specialized Text Generator\n")
    
    code = """
# Specialized Text Generator using Transformers
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Model selection
model_name = "distilgpt2"  # A smaller model for quick experimentation
# Other options: "gpt2", "EleutherAI/gpt-neo-125M", "bigscience/bloom-560m"

def load_model():
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_specialized_text(prompt, style, length=150, temperature=0.7):
    """
    Generate text in a specific style based on the prompt.
    
    Args:
        prompt: The initial prompt for generation
        style: The style of text to generate (e.g., "marketing", "technical")
        length: Maximum length of generated text
        temperature: Control randomness (higher = more random)
        
    Returns:
        Generated text
    """
    # You can customize this function for different styles
    style_prompts = {
        "marketing": "Write compelling marketing copy for: ",
        "technical": "Create detailed technical documentation for: ",
        "creative": "Write a creative story about: ",
        "formal": "Write a formal business communication regarding: "
    }
    
    # Default to creative if style not found
    style_prefix = style_prompts.get(style, style_prompts["creative"])
    full_prompt = f"{style_prefix}{prompt}"
    
    # Use the pipeline for simplicity
    generator = pipeline('text-generation', model=model_name)
    
    result = generator(
        full_prompt,
        max_length=len(full_prompt.split()) + length,
        temperature=temperature,
        top_p=0.9,
        no_repeat_ngram_size=2
    )
    
    # Extract the generated text
    generated_text = result[0]['generated_text']
    
    # Remove the prompt from the output if desired
    # generated_text = generated_text[len(full_prompt):].strip()
    
    return generated_text

# For a simple UI with Gradio
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Specialized Text Generator")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Enter your prompt")
                style = gr.Dropdown(
                    choices=["marketing", "technical", "creative", "formal"],
                    label="Select text style",
                    value="creative"
                )
                length = gr.Slider(minimum=50, maximum=500, value=200, label="Output Length")
                temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
                generate_btn = gr.Button("Generate Text")
            
            with gr.Column():
                output = gr.Textbox(label="Generated Text", lines=10)
        
        generate_btn.click(
            fn=generate_specialized_text,
            inputs=[prompt, style, length, temperature],
            outputs=output
        )
    
    return demo

# Simple evaluation function - you should expand this
def evaluate_generations(generated_texts, criteria=["coherence", "style_match", "creativity"]):
    """
    Evaluate the quality of generated texts based on specified criteria.
    
    Args:
        generated_texts: List of generated text samples
        criteria: List of criteria to evaluate
        
    Returns:
        Evaluation results
    """
    # This is a placeholder for a more sophisticated evaluation
    # You could implement automated metrics or a human evaluation process
    results = {}
    
    # Example: Simple length-based coherence metric
    if "coherence" in criteria:
        avg_sentence_length = sum(len(text.split(".")) for text in generated_texts) / len(generated_texts)
        results["coherence"] = min(1.0, avg_sentence_length / 10)  # Normalize
    
    # Example: Style match could use keyword matching or more advanced techniques
    if "style_match" in criteria:
        # Placeholder for style matching metric
        results["style_match"] = 0.75  # Replace with actual implementation
    
    # Return placeholder results
    return results

# Main function
def main():
    model, tokenizer = load_model()
    
    # For command-line testing
    prompt = "a new smartphone that has innovative features"
    styles = ["marketing", "technical", "creative", "formal"]
    
    print("Example generations:")
    for style in styles:
        print(f"\nStyle: {style}")
        text = generate_specialized_text(prompt, style)
        print(text)
    
    # Evaluation
    print("\nEvaluation:")
    sample_texts = [generate_specialized_text(prompt, style) for style in styles]
    eval_results = evaluate_generations(sample_texts)
    for criterion, score in eval_results.items():
        print(f"{criterion}: {score:.2f}")
    
    # Uncomment to launch the UI
    # demo = create_ui()
    # demo.launch()

if __name__ == "__main__":
    main()
"""
    
    print(code)

def starter_code_option2():
    """
    Provide starter code for Option 2: Text Transformation Tool
    """
    print("\nStarter Code for Option 2: Text Transformation Tool\n")
    
    code = """
# Text Transformation Tool using Transformers
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

# Model selection - T5 is great for text-to-text transformation tasks
model_name = "t5-small"  # Options: "t5-small", "t5-base", "facebook/bart-base"

def load_model():
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def transform_text(text, transformation_type, max_length=150):
    """
    Transform text according to the specified transformation type.
    
    Args:
        text: The text to transform
        transformation_type: The type of transformation to apply
        max_length: Maximum length of transformed text
        
    Returns:
        Transformed text
    """
    # Map transformation types to T5 task prefixes
    transformation_prefixes = {
        "summarize": "summarize: ",
        "simplify": "simplify: ",  # This is not a standard T5 task; for demonstration
        "translate_to_french": "translate English to French: ",
        "grammar_correction": "grammar: ",
        "formal": "convert to formal language: "  # Custom task
    }
    
    # Default to summarization if transformation type not found
    prefix = transformation_prefixes.get(transformation_type, "summarize: ")
    
    # Create the full input text
    input_text = f"{prefix}{text}"
    
    # Use the model to generate the transformed text
    model, tokenizer = load_model()
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the output
    transformed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return transformed_text

# For a simple UI with Gradio
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Text Transformation Tool")
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Enter text to transform", lines=5)
                transformation_type = gr.Dropdown(
                    choices=["summarize", "simplify", "translate_to_french", 
                             "grammar_correction", "formal"],
                    label="Select transformation type",
                    value="summarize"
                )
                max_length = gr.Slider(minimum=50, maximum=500, value=150, 
                                      label="Max Output Length")
                transform_btn = gr.Button("Transform Text")
            
            with gr.Column():
                output = gr.Textbox(label="Transformed Text", lines=5)
        
        transform_btn.click(
            fn=transform_text,
            inputs=[input_text, transformation_type, max_length],
            outputs=output
        )
    
    return demo

# Simple evaluation function - you should expand this
def evaluate_transformations(original_texts, transformed_texts, types):
    """
    Evaluate the quality of text transformations.
    
    Args:
        original_texts: List of original texts
        transformed_texts: List of transformed texts
        types: List of transformation types applied
        
    Returns:
        Evaluation results
    """
    # This is a placeholder for a more sophisticated evaluation
    results = {}
    
    # Example: Length reduction for summarization
    summarization_indices = [i for i, t in enumerate(types) if t == "summarize"]
    if summarization_indices:
        original_lengths = [len(original_texts[i].split()) for i in summarization_indices]
        transformed_lengths = [len(transformed_texts[i].split()) for i in summarization_indices]
        avg_reduction = 1 - sum(transformed_lengths) / sum(original_lengths)
        results["summarization_length_reduction"] = avg_reduction
    
    # Example: Simplification metric (could use readability scores)
    simplification_indices = [i for i, t in enumerate(types) if t == "simplify"]
    if simplification_indices:
        # Placeholder for simplification metric
        results["simplification_score"] = 0.8  # Replace with actual implementation
    
    # Return placeholder results
    return results

# Main function
def main():
    model, tokenizer = load_model()
    
    # For command-line testing
    example_text = """
    The development of artificial intelligence has accelerated dramatically in recent years, 
    with significant advancements in natural language processing, computer vision, and 
    reinforcement learning. These technologies have potential applications across various 
    industries including healthcare, finance, transportation, and entertainment. However, 
    concerns about privacy, bias, and ethical implications remain important considerations 
    for responsible development and deployment.
    """
    
    transformation_types = ["summarize", "simplify", "translate_to_french", 
                           "grammar_correction", "formal"]
    
    print("Example transformations:")
    transformed_texts = []
    for t_type in transformation_types:
        print(f"\nTransformation: {t_type}")
        transformed = transform_text(example_text, t_type)
        transformed_texts.append(transformed)
        print(transformed)
    
    # Evaluation
    print("\nEvaluation:")
    original_texts = [example_text] * len(transformation_types)
    eval_results = evaluate_transformations(original_texts, transformed_texts, 
                                          transformation_types)
    for criterion, score in eval_results.items():
        print(f"{criterion}: {score:.2f}")
    
    # Uncomment to launch the UI
    # demo = create_ui()
    # demo.launch()

if __name__ == "__main__":
    main()
"""
    
    print(code)

def starter_code_option3():
    """
    Provide starter code for Option 3: Specialized Chatbot
    """
    print("\nStarter Code for Option 3: Specialized Chatbot\n")
    
    code = """
# Specialized Chatbot using Transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import numpy as np

# Model selection
model_name = "gpt2"  # Options: "gpt2", "facebook/blenderbot-400M-distill", "EleutherAI/gpt-neo-125M"

def load_model():
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add special tokens for conversation format if needed
    special_tokens = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>"
    }
    
    # Add tokens that might be missing
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            tokenizer.add_special_tokens({token_type: token})
    
    # Resize model embeddings if new tokens were added
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

class SpecializedChatbot:
    def __init__(self, domain="education", character=None):
        """
        Initialize the specialized chatbot.
        
        Args:
            domain: The domain/topic of expertise (e.g., "education", "tech_support")
            character: Optional character personality to emulate
        """
        self.model, self.tokenizer = load_model()
        self.domain = domain
        self.character = character
        self.conversation_history = []
        self.max_history_length = 5  # Remember last 5 turns
        
        # Prepare the domain-specific prompt
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self):
        """Create a system prompt based on domain and character"""
        domain_prompts = {
            "education": "You are a helpful educational tutor who specializes in explaining concepts clearly.",
            "tech_support": "You are a technical support specialist who helps users solve problems with technology.",
            "cooking": "You are a cooking assistant who provides recipes and cooking advice.",
            "fitness": "You are a fitness coach who provides workout advice and motivation."
        }
        
        character_prompts = {
            "shakespeare": "You speak in Shakespearean English and use many metaphors.",
            "detective": "You speak like a film noir detective, always suspicious and dramatic.",
            "pirate": "You speak like a pirate, using terms like 'Arr', 'matey', and 'treasure'.",
            "robot": "You speak in a robotic manner, precise and logical."
        }
        
        # Combine domain and character if both specified
        prompt = domain_prompts.get(self.domain, domain_prompts["education"])
        
        if self.character:
            prompt += " " + character_prompts.get(self.character, "")
        
        return prompt
    
    def _format_conversation(self):
        """Format the conversation history for the model input"""
        formatted = self.system_prompt + "\\n\\n"
        
        for turn in self.conversation_history:
            if turn["role"] == "user":
                formatted += f"User: {turn['content']}\\n"
            else:
                formatted += f"Assistant: {turn['content']}\\n"
        
        formatted += "Assistant:"
        return formatted
    
    def respond(self, user_input, max_length=100, temperature=0.7):
        """
        Generate a response to the user input.
        
        Args:
            user_input: The user's message
            max_length: Maximum response length
            temperature: Control randomness
            
        Returns:
            The chatbot's response
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Trim history if needed
        if len(self.conversation_history) > 2 * self.max_history_length:
            self.conversation_history = self.conversation_history[-2 * self.max_history_length:]
        
        # Format the full conversation
        conversation_text = self._format_conversation()
        
        # Generate response
        input_ids = self.tokenizer.encode(conversation_text, return_tensors="pt")
        
        output = self.model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_length,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
        response = self.tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Clean up response (remove additional turns if generated)
        response = response.split("User:")[0].strip()
        response = response.split("Assistant:")[0].strip()
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        return "Conversation has been reset."

# For a simple UI with Gradio
def create_ui():
    # Initialize chatbot
    domains = ["education", "tech_support", "cooking", "fitness"]
    characters = [None, "shakespeare", "detective", "pirate", "robot"]
    chatbot = SpecializedChatbot(domains[0])
    
    def respond_to_message(message, domain, character, chat_history):
        # Reinitialize if domain or character changes
        nonlocal chatbot
        if domain != chatbot.domain or character != chatbot.character:
            chatbot = SpecializedChatbot(domain, character)
            # Transfer the latest conversation turn if it exists
            if chat_history:
                chatbot.conversation_history = [{"role": "user", "content": chat_history[-1][0]}]
        
        response = chatbot.respond(message)
        chat_history.append((message, response))
        return "", chat_history
    
    def reset_chat():
        nonlocal chatbot
        chatbot.reset_conversation()
        return []
    
    with gr.Blocks() as demo:
        gr.Markdown("# Specialized Chatbot")
        
        with gr.Row():
            with gr.Column():
                domain_dropdown = gr.Dropdown(
                    choices=domains,
                    label="Domain",
                    value=domains[0]
                )
                character_dropdown = gr.Dropdown(
                    choices=[str(c) if c else "None" for c in characters],
                    label="Character",
                    value="None"
                )
                reset_button = gr.Button("Reset Conversation")
            
            with gr.Column():
                chatbot_ui = gr.Chatbot(label="Conversation")
                message = gr.Textbox(label="Your message", placeholder="Type your message here...")
                submit_button = gr.Button("Send")
        
        # Handle events
        submit_button.click(
            fn=respond_to_message,
            inputs=[message, domain_dropdown, character_dropdown, chatbot_ui],
            outputs=[message, chatbot_ui]
        )
        
        message.submit(
            fn=respond_to_message,
            inputs=[message, domain_dropdown, character_dropdown, chatbot_ui],
            outputs=[message, chatbot_ui]
        )
        
        reset_button.click(
            fn=reset_chat,
            outputs=[chatbot_ui]
        )
    
    return demo

# Simple evaluation function - you should expand this
def evaluate_chatbot(chatbot, test_questions, criteria=["relevance", "coherence", "domain_knowledge"]):
    """
    Evaluate the chatbot on test questions.
    
    Args:
        chatbot: The chatbot instance
        test_questions: List of test questions
        criteria: Evaluation criteria
        
    Returns:
        Evaluation results
    """
    # Reset conversation for clean evaluation
    chatbot.reset_conversation()
    
    # Store responses for each question
    responses = []
    for question in test_questions:
        response = chatbot.respond(question)
        responses.append(response)
        print(f"Q: {question}")
        print(f"A: {response}")
        print()
    
    # This is a placeholder for a more sophisticated evaluation
    results = {}
    
    # Example: Response length as a proxy for depth
    if "domain_knowledge" in criteria:
        avg_length = sum(len(response.split()) for response in responses) / len(responses)
        results["domain_knowledge"] = min(1.0, avg_length / 50)  # Normalize
    
    # Example: Domain keyword matching
    if "relevance" in criteria:
        domain_keywords = {
            "education": ["learn", "understand", "concept", "student", "knowledge", "explain"],
            "tech_support": ["issue", "problem", "device", "system", "update", "troubleshoot"],
            "cooking": ["recipe", "ingredient", "cook", "heat", "flavor", "dish"],
            "fitness": ["exercise", "workout", "muscle", "cardio", "training", "strength"]
        }
        
        keywords = domain_keywords.get(chatbot.domain, domain_keywords["education"])
        keyword_matches = 0
        
        for response in responses:
            for keyword in keywords:
                if keyword in response.lower():
                    keyword_matches += 1
        
        avg_matches = keyword_matches / (len(responses) * len(keywords))
        results["relevance"] = min(1.0, avg_matches * 5)  # Scale up a bit
    
    # Return placeholder results
    return results

# Main function
def main():
    # Initialize chatbot
    chatbot = SpecializedChatbot(domain="education")
    
    # For command-line testing
    test_questions = [
        "Can you explain how photosynthesis works?",
        "What's the difference between a virus and bacteria?",
        "How do I solve quadratic equations?",
        "Why is the sky blue?"
    ]
    
    print(f"Domain: {chatbot.domain}")
    print(f"System prompt: {chatbot.system_prompt}")
    print("\nTesting chatbot responses:")
    
    for question in test_questions:
        response = chatbot.respond(question)
        print(f"\nUser: {question}")
        print(f"Chatbot: {response}")
    
    # Evaluation
    print("\nEvaluation:")
    eval_results = evaluate_chatbot(chatbot, test_questions)
    for criterion, score in eval_results.items():
        print(f"{criterion}: {score:.2f}")
    
    # Uncomment to launch the UI
    # demo = create_ui()
    # demo.launch()

if __name__ == "__main__":
    main()
"""
    
    print(code)

def submission_guidelines():
    """
    Provide guidelines for submitting the exercise
    """
    print("\n" + "-" * 70)
    print("SUBMISSION AND PRESENTATION GUIDELINES")
    print("-" * 70)
    
    print("\nCode Submission:")
    print("- Upload your code files to the shared folder")
    print("- Include a brief README.md with:")
    print("  * Team members' names")
    print("  * Application description")
    print("  * Model and approach used")
    print("  * Installation/setup instructions")
    print("  * Example usage")
    
    print("\nPresentation (2-3 minutes):")
    print("1. Introduction (30 seconds)")
    print("   - Team members")
    print("   - Application purpose")
    print("   - Target use case")
    
    print("\n2. Technical Approach (30 seconds)")
    print("   - Model selection and why")
    print("   - Key implementation details")
    print("   - Challenges and solutions")
    
    print("\n3. Demo (1 minute)")
    print("   - Live demonstration of core functionality")
    print("   - Show input/output examples")
    
    print("\n4. Evaluation & Reflection (30 seconds)")
    print("   - How well does it perform?")
    print("   - What would you improve with more time?")
    print("   - Key learnings from the exercise")
    
    print("\nRemember, the focus is on demonstrating your understanding")
    print("of the core concepts rather than building a perfect application.")
    print("Be prepared to answer questions about your implementation choices.")

def main():
    """
    Run the complete exercise instructions
    """
    exercise_instructions()
    starter_code_option1()
    starter_code_option2()
    starter_code_option3()
    submission_guidelines()
    
    print("\n" + "=" * 70)
    print("GOOD LUCK WITH YOUR FINAL EXERCISE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
