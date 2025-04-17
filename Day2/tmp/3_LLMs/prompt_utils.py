"""
Prompt Engineering Utilities
Building Blocks of Generative AI Course - Day 2

Utility functions for working with LLM APIs in the prompt engineering exercise.
"""

import os
import sys
import json
import requests
from IPython.display import HTML, display, Markdown

def setup_api_key():
    """
    Set up and validate the API key for OpenAI.
    
    Returns:
        str: The API key or "dummy_key" if not available
    """
    try:
        API_KEY = os.environ.get('OPENAI_API_KEY')
        if API_KEY is None:
            print("Please set your OPENAI_API_KEY environment variable.")
            print("You can do this by running: export OPENAI_API_KEY='your-api-key'")
            API_KEY = "dummy_key"  # For testing without making actual API calls
    except Exception as e:
        print(f"Error accessing environment variables: {e}")
        API_KEY = "dummy_key"
    
    return API_KEY

def call_openai_api(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=500):
    """
    Call the OpenAI API with the given prompt.
    
    Args:
        prompt (str): The prompt to send to the API
        model (str): The model to use (default: gpt-3.5-turbo)
        temperature (float): Controls randomness (default: 0.7)
        max_tokens (int): Maximum number of tokens to generate (default: 500)
        
    Returns:
        str: The generated text
    """
    API_KEY = setup_api_key()
    
    # If we're in test mode, return a mock response
    if API_KEY == "dummy_key":
        return f"[TEST MODE] Response to: {prompt[:50]}..."
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    except Exception as e:
        return f"Exception occurred: {str(e)}"

def display_response(response, title="Model Response"):
    """
    Display a model response with formatting.
    
    Args:
        response (str): The response to display
        title (str): Title for the response section
    """
    display(Markdown(f"## {title}\n\n{response}"))

def compare_responses(responses, labels):
    """
    Display multiple responses side by side for comparison.
    
    Args:
        responses (list): List of model responses
        labels (list): Labels for each response
    """
    assert len(responses) == len(labels), "Number of responses must match number of labels"
    
    # Create HTML table for side-by-side comparison
    html = "<table style='width:100%; border-collapse: collapse;'>"
    
    # Add header row
    html += "<tr>"
    for label in labels:
        html += f"<th style='border: 1px solid black; padding: 8px; background-color: #f2f2f2;'>{label}</th>"
    html += "</tr>"
    
    # Add response row
    html += "<tr>"
    for response in responses:
        # Replace newlines with HTML breaks
        formatted_response = response.replace("\n", "<br>")
        html += f"<td style='border: 1px solid black; padding: 8px; vertical-align: top;'>{formatted_response}</td>"
    html += "</tr>"
    
    html += "</table>"
    
    # Display the table
    display(HTML(html))

def evaluate_response(response, criteria):
    """
    Manually evaluate a model response based on given criteria.
    
    Args:
        response (str): The model's response
        criteria (list): List of criteria to evaluate
        
    Returns:
        dict: Evaluation scores
    """
    print("\n" + "="*50)
    print("Model Response:")
    print("-"*50)
    print(response)
    print("="*50)
    
    scores = {}
    for criterion in criteria:
        while True:
            try:
                score = float(input(f"Score for {criterion} (0-10): "))
                if 0 <= score <= 10:
                    scores[criterion] = score
                    break
                else:
                    print("Please enter a score between 0 and 10.")
            except ValueError:
                print("Please enter a valid number.")
    
    return scores
