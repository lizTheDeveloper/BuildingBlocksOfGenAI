"""
Grammar Correction Prompt Engineering Exercise
Building Blocks of Generative AI Course - Day 2

This exercise explores how different prompt strategies affect grammar correction quality.
Students will experiment with various prompting techniques for text correction tasks.
"""

# Import the utility functions
from prompt_utils import call_openai_api, compare_responses, display_response

# Sample texts with grammatical errors
sample_texts = [
    "I has went to the store yesterday and buyed three apple.",
    "The company have announced that they is moving to a new location next month.",
    "She dont like when people talks too loud in the library."
]

# TODO: Implement different prompt strategies for grammar correction
# Here are some example prompting strategies to explore:

# 1. Basic prompt
def basic_prompt(text):
    return f"Correct the grammar in this text: {text}"

# 2. TODO: Create a prompt that asks for explanations of corrections

# 3. TODO: Create a prompt that specifies the formality level of the correction

# 4. TODO: Create a prompt that requests highlighting of changes

# 5. TODO: Create a prompt that specifies a target style or regional English variant

# This function will help you test different prompt strategies
def test_prompt_strategy(prompt_function, description):
    """
    Test a prompt strategy on all sample texts
    
    Args:
        prompt_function: Function that takes a text and returns a prompt
        description: Description of the prompt strategy
    """
    print(f"=== Testing: {description} ===")
    
    for i, text in enumerate(sample_texts):
        prompt = prompt_function(text)
        print(f"\nSample {i+1}: {text}")
        print(f"Prompt: {prompt}")
        
        # Uncomment to make actual API calls
        # response = call_openai_api(prompt)
        # print(f"Response: {response}")
        
        print("-" * 50)

# Example usage (commented out to avoid accidental API calls)
"""
# Test the basic prompt strategy
test_prompt_strategy(basic_prompt, "Basic Grammar Correction")

# TODO: Test your other prompt strategies

# TODO: Compare responses from different strategies
# Pick one sample text to compare across all strategies
text_to_compare = sample_texts[0]

responses = []
labels = []

# Collect responses from each strategy
for strategy, description in [
    (basic_prompt, "Basic"),
    # Add your other strategies here
]:
    prompt = strategy(text_to_compare)
    response = call_openai_api(prompt)
    responses.append(response)
    labels.append(description)

# Compare the responses
compare_responses(responses, labels)
"""

if __name__ == "__main__":
    print("This is an exercise template for prompt engineering with grammar correction.")
    print("To run the exercise, set up your API key and uncomment the API calls.")
    print("Then, implement different prompting strategies and compare the results.")
