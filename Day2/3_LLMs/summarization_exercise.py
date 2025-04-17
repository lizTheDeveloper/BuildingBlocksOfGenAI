"""
Text Summarization Prompt Engineering Exercise
Building Blocks of Generative AI Course - Day 2

This exercise explores how different prompt strategies affect text summarization quality.
Students will experiment with various prompting techniques to improve summarization.
"""

# Import the utility functions from prompt_utils.py
from prompt_utils import call_openai_api, compare_responses, display_response

# Sample text to summarize (a passage about machine learning)
sample_text = """
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. IBM has a rich history with machine learning. One of its own, Arthur Samuel, is credited for coining the term "machine learning" with his research around the game of checkers. Robert Nealey, the self-proclaimed checkers master, played the game on an IBM 7094 computer in 1962, and lost to the computer. Compared to what can be done today, this feat seems trivial, but it's considered a major milestone in the field of artificial intelligence. Over the next couple of decades, the technological developments around storage and processing power will enable some innovative products that we know and love today, such as Netflix's recommendation engine or self-driving cars. Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics. As big data continues to expand and grow, the market demand for data scientists will increase, requiring them to assist in the identification of the most relevant business questions and the data to answer them.
"""

# TODO: Implement different prompt strategies for summarization
# Here are some example prompting strategies to explore:

# 1. Basic prompt - just asking for a summary
basic_prompt = f"Summarize the following text:\n\n{sample_text}"

# 2. TODO: Add a more specific prompt that sets length constraints

# 3. TODO: Add a prompt that specifies the target audience (e.g., 5th graders)

# 4. TODO: Add a prompt that requests a specific format (e.g., bullet points)

# 5. TODO: Add a prompt that uses a chain of thought approach 
# (e.g., "First identify key points, then synthesize them into a summary")

# 6. TODO: Add a prompt that takes on a specific persona or role
# (e.g., "You are a professor explaining this to undergraduate students")

# Call the API with different prompts

# This is commented out to avoid accidental API calls
# If you have set up your API key, you can uncomment and run these
"""
response_basic = call_openai_api(basic_prompt)
display_response(response_basic, "Basic Prompt Response")

# TODO: Call the API with your other prompts and display the results

# TODO: Compare your different responses to see which prompt strategy works best
# responses = [response_basic, response_length, response_audience, ...]
# labels = ["Basic", "Length-Constrained", "Child-Friendly", ...]
# compare_responses(responses, labels)
"""

if __name__ == "__main__":
    print("This is an exercise template for prompt engineering with summarization.")
    print("To run the exercise, set up your API key and uncomment the API calls.")
    print("Then, implement different prompting strategies and compare the results.")
