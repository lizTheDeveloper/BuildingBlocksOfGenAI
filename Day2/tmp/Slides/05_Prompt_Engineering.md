# Prompt Engineering: Guiding LLM Behavior

## Introduction to Prompt Engineering (15 minutes)

- Definition: Crafting inputs to elicit desired outputs from LLMs
- Why it matters: Bridge between model capabilities and applications
- Treating LLMs as reasoning engines with the right prompts
- Prompt engineering vs. traditional programming

## Basic Prompting Strategies

- Clear, specific requests
- Contextual information
- Structured format specifications
- Role definition ("You are an expert...")
- Step-by-step guidance
- Reference examples

## Zero-shot vs. Few-shot Prompting

```python
# Zero-shot prompting example
zero_shot_prompt = """
Classify this movie review as positive or negative:
"I couldn't stand the slow pacing and confusing plot. The characters were flat and uninteresting."
"""

# Few-shot prompting example
few_shot_prompt = """
Classify these movie reviews as positive or negative:

Review: "The cinematography was breathtaking, and the soundtrack perfectly complemented the mood."
Classification: Positive

Review: "While the acting was decent, the dialogue felt forced and unnatural."
Classification: Negative

Review: "I couldn't stand the slow pacing and confusing plot. The characters were flat and uninteresting."
Classification:
"""
```

## Task Decomposition in Prompts

- Breaking complex tasks into manageable steps
- Chain of thought prompting
- Benefits:
  - Improved reasoning accuracy
  - Better handling of complex problems
  - More transparent decision process

```python
# Chain of thought prompting example
cot_prompt = """
Solve the following word problem step by step:

Problem: If a shirt originally costs $25 and is marked down by 20%, 
then marked up by 15% from the sale price, what is the final price?

Step 1: Calculate the discount amount.
20% of $25 = 0.2 × $25 = $5

Step 2: Subtract the discount to get the sale price.
$25 - $5 = $20

Step 3: Calculate the markup amount.
15% of $20 = 0.15 × $20 = $3

Step 4: Add the markup to get the final price.
$20 + $3 = $23

Therefore, the final price of the shirt is $23.

Problem: Roger has 5 tennis balls. He buys 2 more cans of tennis balls, with 3 balls in each can.
How many tennis balls does he have now?

Step 1:
"""
```

## Advanced Prompt Engineering Techniques

- In-context learning through examples
- Self-criticism and refinement
- Ensemble prompting: combining multiple approaches
- Persona crafting: specifying expertise and communication style
- Constraint specification: rules and limitations

## Structured Output Engineering

```python
# JSON output specification
json_prompt = """
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
"""
```

## Prompt Templates and Variables

- Creating reusable prompt templates
- Interpolating variables for dynamic content
- Example implementation with Python string formatting

```python
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
```

## Handling Common Prompt Engineering Challenges

- Dealing with context length limitations
- Managing bias in prompts
- Ensuring consistency across multiple outputs
- Addressing hallucinations and factual accuracy
- Iterative refinement of prompts

## Hands-on Prompt Engineering Exercise

- We'll practice different prompting techniques
- Focus on summarization and text correction tasks
- Compare effectiveness of different approaches
- Experiment with:
  - Zero-shot vs. few-shot learning
  - Chain of thought prompting
  - Role and format specification
  - Output constraints
