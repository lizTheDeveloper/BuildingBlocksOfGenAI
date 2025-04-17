"""
LLM Overview Visualization
Building Blocks of Generative AI Course - Day 2

This script visualizes key concepts in Large Language Models (LLMs) including
the evolution from RNNs to Transformers, and demonstrates the concept of 
self-attention and tokens.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.colors as colors

# Set random seed for reproducibility
np.random.seed(42)

def plot_llm_evolution():
    """
    Visualize the evolution of language models from RNNs to Transformers.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axis_off()
    
    # Title
    plt.suptitle("Evolution of Language Models", fontsize=20, y=0.98)
    
    # Create a timeline
    timeline_y = 0.1
    ax.axhline(y=timeline_y, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=2)
    
    # Add timeline points and labels
    timeline_points = [
        (0.1, "2013: Word2Vec"),
        (0.25, "2014: Seq2Seq + RNNs"),
        (0.4, "2015: Attention Mechanism"),
        (0.55, "2017: Transformer Architecture"),
        (0.7, "2019: BERT & GPT-2"),
        (0.85, "2022: ChatGPT & Large LLMs")
    ]
    
    for x, label in timeline_points:
        # Add point
        ax.plot(x, timeline_y, 'o', markersize=10, color='blue')
        
        # Add label
        year = label.split(":")[0]
        model = label.split(":")[1].strip()
        ax.text(x, timeline_y-0.05, year, horizontalalignment='center', fontsize=12)
        ax.text(x, timeline_y-0.08, model, horizontalalignment='center', fontsize=10)
    
    # Add model architecture diagrams above the timeline
    # Word2Vec
    x, y = 0.1, 0.3
    ax.text(x, y+0.25, "Word2Vec", fontsize=14, horizontalalignment='center', fontweight='bold')
    ax.text(x, y+0.2, "Word Embeddings", fontsize=10, horizontalalignment='center')
    
    # Draw Word2Vec diagram
    rect_width, rect_height = 0.06, 0.04
    ax.add_patch(patches.Rectangle((x-0.03, y), rect_width, rect_height, 
                                   facecolor='lightblue', edgecolor='black'))
    ax.text(x, y+0.02, "Input", horizontalalignment='center', fontsize=8)
    
    ax.arrow(x, y+rect_height, 0, 0.05, head_width=0.01, head_length=0.01, 
             fc='black', ec='black')
    
    ax.add_patch(patches.Rectangle((x-0.03, y+rect_height+0.05), rect_width, rect_height, 
                                  facecolor='lightgreen', edgecolor='black'))
    ax.text(x, y+rect_height+0.05+0.02, "Embedding", horizontalalignment='center', fontsize=8)
    
    # Seq2Seq + RNNs
    x = 0.25
    ax.text(x, y+0.25, "Seq2Seq + RNNs", fontsize=14, horizontalalignment='center', fontweight='bold')
    ax.text(x, y+0.2, "Sequential Processing", fontsize=10, horizontalalignment='center')
    
    # Draw RNN diagram
    circle_radius = 0.02
    circle_x = [x-0.04, x, x+0.04]
    circle_y = y+0.05
    colors = ['lightblue', 'lightblue', 'lightblue']
    
    for i, (cx, color) in enumerate(zip(circle_x, colors)):
        circle = plt.Circle((cx, circle_y), circle_radius, fill=True, color=color, edgecolor='black')
        ax.add_patch(circle)
        
        # Add connections between RNN cells
        if i < len(circle_x) - 1:
            ax.arrow(cx+circle_radius, circle_y, 
                    circle_x[i+1]-(cx+circle_radius*2), 0, 
                    head_width=0.01, head_length=0.01, fc='black', ec='black')
    
    # Add Input arrows
    for cx in circle_x:
        ax.arrow(cx, y, 0, circle_y-y-circle_radius, head_width=0.01, head_length=0.01, 
                fc='black', ec='black')
    
    # Add Output arrows
    for cx in circle_x:
        ax.arrow(cx, circle_y+circle_radius, 0, 0.05, head_width=0.01, head_length=0.01, 
                fc='black', ec='black')
    
    # Attention Mechanism
    x = 0.4
    ax.text(x, y+0.25, "Attention Mechanism", fontsize=14, horizontalalignment='center', fontweight='bold')
    ax.text(x, y+0.2, "Focused Connections", fontsize=10, horizontalalignment='center')
    
    # Draw Attention diagram - simplified as connections between two sets of nodes
    src_y = y+0.02
    tgt_y = y+0.12
    
    # Source nodes
    src_nodes_x = [x-0.05, x, x+0.05]
    for i, sx in enumerate(src_nodes_x):
        circle = plt.Circle((sx, src_y), 0.015, fill=True, color='lightblue', edgecolor='black')
        ax.add_patch(circle)
        ax.text(sx, src_y-0.03, f"s{i+1}", horizontalalignment='center', fontsize=8)
    
    # Target nodes
    tgt_nodes_x = [x-0.03, x+0.03]
    for i, tx in enumerate(tgt_nodes_x):
        circle = plt.Circle((tx, tgt_y), 0.015, fill=True, color='lightgreen', edgecolor='black')
        ax.add_patch(circle)
        ax.text(tx, tgt_y+0.03, f"t{i+1}", horizontalalignment='center', fontsize=8)
    
    # Attention connections - thicker lines represent stronger attention
    line_styles = ['-', '--', ':']
    line_widths = [2, 1, 0.5]
    
    for sx in src_nodes_x:
        for tx, ls, lw in zip(tgt_nodes_x, line_styles, line_widths):
            plt.plot([sx, tx], [src_y, tgt_y], ls, color='red', linewidth=lw, alpha=0.7)
    
    # Transformer Architecture
    x = 0.55
    ax.text(x, y+0.25, "Transformer", fontsize=14, horizontalalignment='center', fontweight='bold')
    ax.text(x, y+0.2, "Parallelized Attention", fontsize=10, horizontalalignment='center')
    
    # Draw simplified Transformer diagram
    block_width, block_height = 0.08, 0.04
    
    # Encoder blocks
    ax.add_patch(patches.Rectangle((x-0.04, y), block_width, block_height, 
                                  facecolor='lightblue', edgecolor='black'))
    ax.text(x, y+0.02, "Self-Attention", horizontalalignment='center', fontsize=8)
    
    ax.arrow(x, y+block_height, 0, 0.02, head_width=0.01, head_length=0.01, 
             fc='black', ec='black')
    
    ax.add_patch(patches.Rectangle((x-0.04, y+block_height+0.02), block_width, block_height, 
                                  facecolor='lightblue', edgecolor='black'))
    ax.text(x, y+block_height+0.02+0.02, "Feed Forward", horizontalalignment='center', fontsize=8)
    
    # BERT & GPT
    x = 0.7
    ax.text(x, y+0.25, "BERT & GPT", fontsize=14, horizontalalignment='center', fontweight='bold')
    ax.text(x, y+0.2, "Pre-trained Models", fontsize=10, horizontalalignment='center')
    
    # Draw BERT & GPT diagram
    ax.add_patch(patches.Rectangle((x-0.06, y), 0.05, block_height*2+0.02, 
                                  facecolor='#d4a76a', edgecolor='black'))
    ax.text(x-0.035, y+block_height+0.02, "BERT", horizontalalignment='center', fontsize=8, rotation=90)
    
    ax.add_patch(patches.Rectangle((x+0.01, y), 0.05, block_height*2+0.02, 
                                  facecolor='#76b5c5', edgecolor='black'))
    ax.text(x+0.035, y+block_height+0.02, "GPT", horizontalalignment='center', fontsize=8, rotation=90)
    
    ax.arrow(x, y-0.05, 0, 0.05, head_width=0.01, head_length=0.01, 
             fc='black', ec='black')
    ax.text(x, y-0.06, "Huge Text Corpus", horizontalalignment='center', fontsize=8)
    
    # Modern LLMs
    x = 0.85
    ax.text(x, y+0.25, "Modern LLMs", fontsize=14, horizontalalignment='center', fontweight='bold')
    ax.text(x, y+0.2, "Billions of Parameters", fontsize=10, horizontalalignment='center')
    
    # Draw Modern LLM diagram - represented as a stack of many transformer layers
    layer_height = 0.01
    num_layers = 10
    layer_width = 0.08
    
    for i in range(num_layers):
        layer_y = y + i * layer_height
        ax.add_patch(patches.Rectangle((x-layer_width/2, layer_y), layer_width, layer_height-0.001, 
                                      facecolor=plt.cm.viridis(i/num_layers), edgecolor='black', linewidth=0.5))
    
    ax.text(x, y+num_layers*layer_height+0.02, "Many Layers", horizontalalignment='center', fontsize=8)
    
    # Add arrows showing scaling up
    ax.arrow(x-0.06, y+num_layers*layer_height/2, 0.02, 0, head_width=0.01, head_length=0.01, 
             fc='black', ec='black')
    ax.text(x-0.08, y+num_layers*layer_height/2, "Scale Up", horizontalalignment='right', fontsize=8)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('llm_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_self_attention():
    """
    Visualize how self-attention works in a transformer by showing attention maps.
    """
    # Sample sentence
    sentence = "The cat sat on the mat."
    tokens = sentence.split()
    
    # Create a fake attention matrix
    n_tokens = len(tokens)
    attention = np.zeros((n_tokens, n_tokens))
    
    # Set attention patterns
    # 'The' attends to 'cat'
    attention[0, 1] = 0.7
    attention[0, 0] = 0.3
    
    # 'cat' attends to itself and 'sat'
    attention[1, 1] = 0.6
    attention[1, 2] = 0.4
    
    # 'sat' attends to 'cat' and 'on'
    attention[2, 1] = 0.5
    attention[2, 3] = 0.5
    
    # 'on' attends to 'sat' and 'the'
    attention[3, 2] = 0.3
    attention[3, 4] = 0.7
    
    # 'the' attends to 'on' and 'mat'
    attention[4, 3] = 0.4
    attention[4, 5] = 0.6
    
    # 'mat' attends to itself and 'the'
    attention[5, 4] = 0.3
    attention[5, 5] = 0.7
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot attention heatmap
    im = ax1.imshow(attention, cmap='viridis')
    
    # Add colorbar
    cbar = ax1.figure.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax1.set_xticks(np.arange(n_tokens))
    ax1.set_yticks(np.arange(n_tokens))
    ax1.set_xticklabels(tokens)
    ax1.set_yticklabels(tokens)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(n_tokens):
        for j in range(n_tokens):
            text = ax1.text(j, i, f"{attention[i, j]:.1f}",
                          ha="center", va="center", color="w" if attention[i, j] > 0.5 else "black")
    
    ax1.set_title("Self-Attention Weights")
    
    # Plot token relationships as a graph
    token_positions = np.array([
        [1, 3],   # The
        [2, 2],   # cat
        [3, 3],   # sat
        [4, 2],   # on
        [5, 3],   # the
        [6, 2],   # mat
    ])
    
    # Normalize token positions
    token_positions = token_positions / np.max(token_positions)
    
    # Plot tokens as nodes
    ax2.scatter(token_positions[:, 0], token_positions[:, 1], s=1000, c='lightblue', edgecolor='black', zorder=2)
    
    # Add token labels
    for i, (x, y) in enumerate(token_positions):
        ax2.text(x, y, tokens[i], ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw attention connections
    for i in range(n_tokens):
        for j in range(n_tokens):
            if attention[i, j] > 0.1:  # Only draw significant connections
                # Draw an arrow from token i to token j
                arrow_width = attention[i, j] * 2  # Scale arrow width by attention weight
                ax2.arrow(token_positions[i, 0], token_positions[i, 1],
                         token_positions[j, 0] - token_positions[i, 0], 
                         token_positions[j, 1] - token_positions[i, 1],
                         head_width=0.02, head_length=0.02, 
                         fc='red', ec='red', alpha=attention[i, j], 
                         length_includes_head=True,
                         linewidth=arrow_width,
                         zorder=1)
    
    ax2.set_title("Token Relationships via Self-Attention")
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(0, 1.2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('self_attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_tokenization():
    """
    Visualize how text is tokenized in modern LLMs.
    """
    # Sample text
    text = "Transformers are neural networks that learn context and understanding."
    
    # Different tokenization schemes (for demonstration)
    char_tokens = list(text)
    word_tokens = text.split()
    subword_tokens = [
        "Transform", "ers", "are", "neural", "networks", "that", "learn", "context", "and", "understand", "ing", "."
    ]
    
    # Set up the figure
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    # Helper function to visualize token boundaries
    def draw_tokens(ax, tokens, title):
        ax.set_xlim(0, len(text))
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        
        # Display the original text at the top
        ax.text(0, 0.8, text, fontsize=14)
        
        # Calculate token positions
        token_positions = []
        pos = 0
        for token in tokens:
            token_len = len(token)
            if token in text[pos:]:
                start_idx = text.index(token, pos)
                token_positions.append((start_idx, start_idx + token_len, token))
                pos = start_idx + token_len
        
        # Draw token boundaries
        for i, (start, end, token) in enumerate(token_positions):
            # Draw bounding box
            rect = patches.Rectangle((start, 0.3), end-start, 0.2, 
                                    facecolor=plt.cm.tab10(i % 10), alpha=0.3)
            ax.add_patch(rect)
            
            # Add token label
            ax.text((start + end) / 2, 0.2, token, ha='center', va='center', fontsize=10)
            
            # Add token index
            ax.text((start + end) / 2, 0.1, f"Token {i}", ha='center', va='center', fontsize=8)
    
    # Visualize character tokenization
    draw_tokens(axs[0], char_tokens, "Character Tokenization")
    
    # Visualize word tokenization
    draw_tokens(axs[1], word_tokens, "Word Tokenization")
    
    # Visualize subword tokenization
    draw_tokens(axs[2], subword_tokens, "Subword Tokenization (WordPiece/BPE)")
    
    # Add explanation text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    axs[0].text(len(text)*0.7, 0.5, "Character tokens: Very fine-grained\nMany tokens per word\nNo OOV issues", 
               fontsize=10, verticalalignment='center', bbox=props)
    
    axs[1].text(len(text)*0.7, 0.5, "Word tokens: Natural boundaries\nFewer tokens overall\nOOV words are a problem", 
               fontsize=10, verticalalignment='center', bbox=props)
    
    axs[2].text(len(text)*0.7, 0.5, "Subword tokens: Balance between\ncharacter and word tokenization\nHandles rare words better", 
               fontsize=10, verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig('tokenization_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_prompt_engineering():
    """
    Visualize how prompt engineering works with LLMs.
    """
    # Sample prompts and outputs
    prompts = [
        "Tell me about Mars",
        "Write a detailed report on Mars' atmosphere, temperature, and geological features",
        "You are a NASA scientist writing for 10-year-old children. Explain Mars in simple terms.",
        "Step by step, analyze the composition of Mars' atmosphere and compare it to Earth's."
    ]
    
    outputs = [
        "Mars is the fourth planet from the Sun...",
        "Mars Atmospheric Report:\nAtmosphere: Primarily CO2 (95%)...\nTemperature: Ranges from -140°C to 30°C...\nGeology: Features include Olympus Mons...",
        "Hi kids! Mars is the red planet. It looks red because it has rust in its soil! Mars is smaller than Earth...",
        "Step 1: Mars' atmosphere is 95% carbon dioxide, 2.8% nitrogen, and 2% argon...\nStep 2: Earth's atmosphere is 78% nitrogen, 21% oxygen...\nStep 3: Comparing these, we see Mars has much more CO2..."
    ]
    
    output_styles = [
        "Basic factual information",
        "Detailed technical report",
        "Child-friendly explanation",
        "Structured analytical response"
    ]
    
    prompt_techniques = [
        "Simple query",
        "Format specification",
        "Role/audience definition",
        "Process instruction"
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()
    
    # Title
    plt.suptitle("Prompt Engineering Techniques", fontsize=20, y=0.98)
    
    # Set up grid for prompt-output pairs
    grid_height = 0.85 / len(prompts)
    
    for i, (prompt, output, style, technique) in enumerate(zip(prompts, outputs, output_styles, prompt_techniques)):
        y_pos = 0.9 - i * grid_height
        
        # Add technique label
        ax.text(0.02, y_pos, f"Technique {i+1}:", fontsize=14, fontweight='bold')
        ax.text(0.02, y_pos - 0.03, technique, fontsize=12)
        
        # Draw prompt box
        prompt_box = patches.FancyBboxPatch(
            (0.2, y_pos - 0.12), 0.25, 0.1, 
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor='lightblue', alpha=0.4
        )
        ax.add_patch(prompt_box)
        ax.text(0.325, y_pos - 0.07, "Prompt:", ha='center', fontsize=12, fontweight='bold')
        
        # Add prompt text (shortened for display)
        ax.text(0.325, y_pos - 0.095, prompt[:30] + "..." if len(prompt) > 30 else prompt, 
               ha='center', fontsize=10, va='center', wrap=True)
        
        # Draw arrow
        ax.arrow(0.45, y_pos - 0.07, 0.05, 0, head_width=0.01, head_length=0.01, 
                fc='black', ec='black')
        
        # Draw LLM box
        llm_box = patches.FancyBboxPatch(
            (0.5, y_pos - 0.1), 0.1, 0.06, 
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor='lightgreen', alpha=0.4
        )
        ax.add_patch(llm_box)
        ax.text(0.55, y_pos - 0.07, "LLM", ha='center', fontsize=12, fontweight='bold')
        
        # Draw arrow
        ax.arrow(0.6, y_pos - 0.07, 0.05, 0, head_width=0.01, head_length=0.01, 
                fc='black', ec='black')
        
        # Draw output box
        output_box = patches.FancyBboxPatch(
            (0.65, y_pos - 0.12), 0.25, 0.1, 
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor='lightyellow', alpha=0.4
        )
        ax.add_patch(output_box)
        ax.text(0.775, y_pos - 0.07, "Output:", ha='center', fontsize=12, fontweight='bold')
        
        # Add output text (shortened for display)
        ax.text(0.775, y_pos - 0.095, output[:30] + "..." if len(output) > 30 else output, 
               ha='center', fontsize=10, va='center', wrap=True)
        
        # Add output style
        ax.text(0.9, y_pos - 0.07, style, fontsize=10, fontweight='bold', color='darkred')
    
    # Add explanation box at the bottom
    explanation = ("Prompt engineering is the process of crafting inputs to get desired outputs from LLMs.\n"
                  "Effective prompts can specify format, tone, audience, and step-by-step instructions.\n"
                  "The same query with different prompting techniques produces very different results.")
    
    explanation_box = patches.FancyBboxPatch(
        (0.1, 0.1), 0.8, 0.1, 
        boxstyle=patches.BoxStyle("Round", pad=0.02),
        facecolor='lightgray', alpha=0.4
    )
    ax.add_patch(explanation_box)
    ax.text(0.5, 0.15, "Understanding Prompt Engineering", ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.12, explanation, ha='center', fontsize=11, va='center')
    
    plt.tight_layout()
    plt.savefig('prompt_engineering.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Visualizing LLM evolution...")
    plot_llm_evolution()
    
    print("Visualizing self-attention mechanism...")
    visualize_self_attention()
    
    print("Visualizing tokenization methods...")
    visualize_tokenization()
    
    print("Visualizing prompt engineering techniques...")
    plot_prompt_engineering()
    
    print("Visualization complete! Check the output images.")
