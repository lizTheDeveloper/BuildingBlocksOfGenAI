"""
Prompt Engineering Examples for LLM Section
Building Blocks of Generative AI Course - Day 2
"""

# This file contains example prompts that demonstrate different prompt engineering techniques.
# These are meant to be used in slides and for reference.


# Basic prompt structure examples
BASIC_PROMPTS = {
    "simple_instruction": "Summarize the following article about climate change.",
    
    "with_context": """
    Context: You are writing a summary for high school students who are learning about
    environmental science for the first time.
    
    Task: Summarize the following article about climate change.
    """,
    
    "with_format_specification": """
    Summarize the following article about climate change.
    
    Format your response as:
    - Main Point: [1-2 sentences]
    - Key Facts: [3-5 bullet points]
    - Conclusion: [1 sentence]
    """,
    
    "with_persona": """
    You are an environmental scientist with 20 years of experience explaining
    complex topics to the general public.
    
    Summarize the following article about climate change.
    """
}


# Zero-shot vs. few-shot prompting examples
SHOT_EXAMPLES = {
    "zero_shot_classification": """
    Classify this movie review as positive or negative:
    
    "I couldn't stand the slow pacing and confusing plot. The characters were flat and uninteresting."
    """,
    
    "few_shot_classification": """
    Classify these movie reviews as positive or negative:
    
    Review: "The cinematography was breathtaking, and the soundtrack perfectly complemented the mood."
    Classification: Positive
    
    Review: "While the acting was decent, the dialogue felt forced and unnatural."
    Classification: Negative
    
    Review: "I couldn't stand the slow pacing and confusing plot. The characters were flat and uninteresting."
    Classification:
    """,
    
    "zero_shot_generation": """
    Write a short poem about artificial intelligence.
    """,
    
    "few_shot_generation": """
    Here are some examples of short, four-line poems about technology:
    
    Poem about smartphones:
    Glowing screens in hand,
    Windows to a digital land.
    Connecting hearts from afar,
    Yet sometimes we forget who we are.
    
    Poem about robots:
    Metal minds of our design,
    Working where humans once stood in line.
    Programmed helpers day and night,
    Evolution of mechanical might.
    
    Now, write a similar four-line poem about artificial intelligence:
    """
}


# Chain of thought prompting examples
CHAIN_OF_THOUGHT_EXAMPLES = {
    "standard_reasoning": """
    Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
    """,
    
    "chain_of_thought": """
    Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
    
    Let's think through this step by step:
    1. Initially, Roger has 5 tennis balls.
    2. He buys 2 cans of tennis balls.
    3. Each can contains 3 tennis balls.
    4. So the 2 cans contain 2 × 3 = 6 tennis balls.
    5. In total, Roger now has 5 + 6 = 11 tennis balls.
    
    The answer is 11 tennis balls.
    """,
    
    "few_shot_chain_of_thought": """
    Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
    
    Step 1: Olivia initially has $23.
    Step 2: Each bagel costs $3, and she buys 5 bagels.
    Step 3: The total cost of the bagels is 5 × $3 = $15.
    Step 4: Olivia's remaining money is $23 - $15 = $8.
    Answer: $8
    
    Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
    
    Step 1:
    """
}


# Structured output prompting
STRUCTURED_OUTPUT_EXAMPLES = {
    "json_output": """
    Extract the following information from this text and return it as JSON:
    - Person name
    - Company
    - Job title
    - Contact details (phone and email)
    
    Text: "John Smith is a Senior Software Engineer at ABC Technologies. 
    You can reach him at john.smith@abctech.com or call (555) 123-4567."
    
    Format the output as:
    {
      "name": "",
      "company": "",
      "title": "",
      "contact": {
        "email": "",
        "phone": ""
      }
    }
    """,
    
    "markdown_table": """
    Create a comparison table of the following programming languages: Python, JavaScript, and Java.
    Compare them based on:
    - Typing system
    - Primary use cases
    - Performance characteristics
    
    Format your response as a markdown table.
    """,
    
    "bullet_list": """
    List the top 5 benefits of exercise for mental health.
    
    Format your response as a bulleted list with a brief explanation (1-2 sentences) for each benefit.
    """
}


# System vs. user prompt examples (for systems that support role separation)
ROLE_BASED_PROMPTS = {
    "system_user_example": """
    System: You are a professional chef specializing in Italian cuisine. You provide concise recipes
    that are authentic and achievable for home cooks with basic equipment.
    
    User: I have tomatoes, basil, mozzarella, and olive oil. What can I make?
    """,
    
    "system_user_assistant_example": """
    System: You are a professional chef specializing in Italian cuisine. You provide concise recipes
    that are authentic and achievable for home cooks with basic equipment.
    
    User: I have tomatoes, basil, mozzarella, and olive oil. What can I make?
    
    Assistant: With those ingredients, you can make a classic Caprese salad, a cornerstone of Italian antipasti.
    
    Recipe: Slice the tomatoes and mozzarella into 1/4-inch rounds. Arrange them alternately on a plate, 
    overlapping slightly. Tuck fresh basil leaves between the slices. Drizzle with high-quality olive oil, 
    and season with salt and freshly ground black pepper to taste.
    
    This simple dish highlights the quality of each ingredient. For the best result, use room-temperature 
    tomatoes and mozzarella.
    
    User: Could I add balsamic vinegar to that?
    """
}


# Template prompts with variables
def create_summarization_prompt(text, max_length=150, audience="general"):
    """
    Create a customized summarization prompt
    """
    audience_descriptions = {
        "general": "for a general audience",
        "technical": "for a technical audience, using domain-specific terminology",
        "children": "for children, using simple language and concepts",
        "executive": "for busy executives, focusing on key business implications"
    }
    
    audience_desc = audience_descriptions.get(audience, audience_descriptions["general"])
    
    prompt = f"""
    Summarize the following text in no more than {max_length} words {audience_desc}.
    Make sure the summary covers the main points and maintains the original tone.
    
    Text to summarize:
    {text}
    
    Summary:
    """
    
    return prompt


# Example implementation of self-refining prompting
def self_refining_prompt(initial_output):
    """
    Create a prompt that asks for refinement of an initial output
    """
    prompt = f"""
    Here is a drafted response:
    
    "{initial_output}"
    
    Please improve this response by:
    1. Checking for factual accuracy
    2. Improving clarity and conciseness
    3. Ensuring a logical flow of ideas
    4. Adding relevant examples or details if missing
    
    Provide your improved version:
    """
    
    return prompt


# Main function to print examples (for demonstration purposes)
def display_example_prompts():
    """Display the example prompts for reference"""
    
    print("=== Basic Prompt Structures ===")
    for name, prompt in BASIC_PROMPTS.items():
        print(f"\n{name}:")
        print(prompt)
    
    print("\n\n=== Zero-Shot vs. Few-Shot Prompting ===")
    for name, prompt in SHOT_EXAMPLES.items():
        print(f"\n{name}:")
        print(prompt)
    
    print("\n\n=== Chain of Thought Prompting ===")
    for name, prompt in CHAIN_OF_THOUGHT_EXAMPLES.items():
        print(f"\n{name}:")
        print(prompt)
    
    print("\n\n=== Structured Output Prompting ===")
    for name, prompt in STRUCTURED_OUTPUT_EXAMPLES.items():
        print(f"\n{name}:")
        print(prompt)
    
    print("\n\n=== Role-Based Prompting ===")
    for name, prompt in ROLE_BASED_PROMPTS.items():
        print(f"\n{name}:")
        print(prompt)
    
    print("\n\n=== Template-Based Prompting ===")
    sample_text = "This is a sample text that needs to be summarized for different audiences."
    for audience in ["general", "technical", "children", "executive"]:
        print(f"\nSummarization for {audience} audience:")
        print(create_summarization_prompt(sample_text, max_length=100, audience=audience))
    
    print("\n\n=== Self-Refining Prompting ===")
    initial_output = "Machine learning is when computers learn from data."
    print(self_refining_prompt(initial_output))


if __name__ == "__main__":
    display_example_prompts()
