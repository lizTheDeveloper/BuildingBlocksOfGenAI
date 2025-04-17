"""
Attention Mechanism Visualization
Building Blocks of Generative AI Course - Day 2

This script visualizes different types of attention mechanisms used in neural networks,
including Bahdanau attention and scaled dot-product attention.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
np.random.seed(42)

def visualize_bahdanau_attention():
    """
    Visualize Bahdanau (additive) attention mechanism.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Title
    plt.suptitle("Bahdanau (Additive) Attention Mechanism", fontsize=16)
    
    # Define colors
    encoder_color = "#AED6F1"  # Light blue
    decoder_color = "#A2D9CE"  # Light green
    attention_color = "#F5B7B1"  # Light red
    
    # Define positions
    encoder_y = 0.7
    decoder_y = 0.3
    attention_y = 0.5
    
    # Number of units
    n_encoder = 5
    n_decoder = 3
    
    # Setup axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Draw encoder states
    encoder_x = np.linspace(0.2, 0.8, n_encoder)
    for i, x in enumerate(encoder_x):
        circle = plt.Circle((x, encoder_y), 0.05, color=encoder_color, ec='black')
        ax.add_patch(circle)
        ax.text(x, encoder_y - 0.08, f"h{i+1}", ha='center')
    
    # Draw decoder state
    decoder_x = 0.5
    circle = plt.Circle((decoder_x, decoder_y), 0.05, color=decoder_color, ec='black')
    ax.add_patch(circle)
    ax.text(decoder_x, decoder_y - 0.08, "s_t", ha='center')
    
    # Draw attention mechanism
    attention_x = 0.5
    rect = patches.Rectangle((attention_x - 0.15, attention_y - 0.07), 0.3, 0.14, 
                            color=attention_color, ec='black')
    ax.add_patch(rect)
    ax.text(attention_x, attention_y, "Attention", ha='center')
    
    # Draw arrows from encoder to attention
    for x in encoder_x:
        ax.arrow(x, encoder_y - 0.05, 
                 attention_x - x, attention_y - encoder_y + 0.12, 
                 head_width=0.01, head_length=0.01, fc='black', ec='black')
    
    # Draw arrow from decoder to attention
    ax.arrow(decoder_x, decoder_y + 0.05, 
             0, attention_y - decoder_y - 0.12, 
             head_width=0.01, head_length=0.01, fc='black', ec='black')
    
    # Draw context vector arrow
    ax.arrow(attention_x, attention_y - 0.07, 
             0, decoder_y - attention_y + 0.12, 
             head_width=0.01, head_length=0.01, fc='blue', ec='blue')
    
    # Add context vector label
    ax.text(attention_x + 0.08, (attention_y + decoder_y) / 2, "Context Vector", 
           color='blue', ha='left', va='center')
    
    # Attention weights display
    for i, x in enumerate(encoder_x):
        weight = np.random.rand() * 0.9 + 0.1  # Random weight between 0.1 and 1.0
        width = max(0.01, weight * 0.03)  # Scale line width with weight
        alpha = weight  # Scale transparency with weight
        
        # Draw weighted connections
        plt.plot([attention_x, x], [attention_y + 0.07, encoder_y - 0.05], 
                color='red', linewidth=width * 10, alpha=alpha)
        
        # Display weights
        ax.text((attention_x + x) / 2, (attention_y + encoder_y) / 2 + 0.05, 
               f"{weight:.2f}", ha='center', va='center', color='red')
    
    # Add formula explanation
    formula = r"$score(s_t, h_i) = v_a^T \tanh(W_a s_t + U_a h_i)$"
    ax.text(0.5, 0.9, formula, ha='center', fontsize=14)
    
    # Explanation text
    explanation = """
    Bahdanau attention (2014) calculates attention weights by:
    1. Passing encoder (h_i) and decoder (s_t) states through a small neural network
    2. Computing a score for each encoder state
    3. Normalizing scores with softmax to get attention weights
    4. Creating a context vector as weighted sum of encoder states
    """
    
    ax.text(0.02, 0.02, explanation, fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.savefig('bahdanau_attention.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_scaled_dot_product_attention():
    """
    Visualize scaled dot-product attention as used in Transformers.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Title
    plt.suptitle("Scaled Dot-Product Attention", fontsize=16)
    
    # Define colors
    query_color = "#AED6F1"  # Light blue
    key_color = "#A2D9CE"    # Light green
    value_color = "#F5B7B1"  # Light red
    output_color = "#D7BDE2"  # Light purple
    
    # Create a custom colormap for attention weights
    attention_cmap = LinearSegmentedColormap.from_list(
        "attention_cmap", ["white", "red"])
    
    # Number of positions in the sequence
    n_positions = 5
    
    # Setup axes for matrices
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Draw the matrices
    cell_size = 0.06
    
    # Helper function to draw matrices
    def draw_matrix(top_left, rows, cols, values=None, cmap=None, title=None):
        x, y = top_left
        for i in range(rows):
            for j in range(cols):
                color = 'white'
                if values is not None and cmap is not None:
                    color = cmap(values[i, j])
                
                rect = patches.Rectangle(
                    (x + j * cell_size, y - i * cell_size), 
                    cell_size, cell_size, 
                    fill=True, color=color, ec='black'
                )
                ax.add_patch(rect)
                
                if values is not None:
                    ax.text(x + j * cell_size + cell_size/2, 
                           y - i * cell_size - cell_size/2, 
                           f"{values[i, j]:.1f}", ha='center', va='center', fontsize=8)
        
        if title:
            ax.text(x + (cols * cell_size) / 2, y + 0.02, title, ha='center', fontsize=10)
    
    # Generate some example values
    query = np.random.rand(1, n_positions, 3)  # Single head, 5 positions, d_k=3
    key = np.random.rand(1, n_positions, 3)
    value = np.random.rand(1, n_positions, 3)
    
    # Calculate attention scores (dot product of query and key)
    scores = np.zeros((n_positions, n_positions))
    for i in range(n_positions):
        for j in range(n_positions):
            scores[i, j] = np.dot(query[0, i], key[0, j]) / np.sqrt(3)  # Scaled by sqrt(d_k)
    
    # Apply softmax to get attention weights
    exp_scores = np.exp(scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculate output (weighted sum of values)
    output = np.zeros((n_positions, 3))
    for i in range(n_positions):
        for j in range(n_positions):
            output[i] += attention_weights[i, j] * value[0, j]
    
    # Draw the Query matrix
    draw_matrix((0.1, 0.8), n_positions, 3, title="Query (Q)")
    
    # Draw the Key matrix
    draw_matrix((0.35, 0.8), n_positions, 3, title="Key (K)")
    
    # Draw the Value matrix
    draw_matrix((0.6, 0.8), n_positions, 3, title="Value (V)")
    
    # Draw the attention weights matrix
    draw_matrix((0.35, 0.5), n_positions, n_positions, 
               values=attention_weights, cmap=plt.cm.Reds, 
               title="Attention Weights")
    
    # Draw the output matrix
    draw_matrix((0.6, 0.5), n_positions, 3, title="Output")
    
    # Draw arrows to show data flow
    ax.arrow(0.16 + 3*cell_size/2, 0.78, 0.05, -0.18, head_width=0.01, 
            head_length=0.02, fc='black', ec='black')
    ax.arrow(0.41 + 3*cell_size/2, 0.78, -0.05, -0.18, head_width=0.01, 
            head_length=0.02, fc='black', ec='black')
    ax.arrow(0.41 + 5*cell_size/2, 0.48, 0.04, 0, head_width=0.01, 
            head_length=0.02, fc='black', ec='black')
    ax.arrow(0.66 + 3*cell_size/2, 0.78, 0.08, -0.38, head_width=0.01, 
            head_length=0.02, fc='black', ec='black', linestyle='--')
    
    # Add formula
    formula = r"$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$"
    ax.text(0.5, 0.92, formula, ha='center', fontsize=14)
    
    # Add explanation
    explanation = """
    Scaled Dot-Product Attention (2017) from the Transformer paper:
    
    1. Calculate similarity between Query (Q) and Key (K) vectors with dot product
    2. Scale by square root of key dimension (d_k) to prevent vanishing gradients
    3. Apply softmax to get attention weights
    4. Multiply Value (V) vectors by attention weights to get the output
    
    This attention mechanism is more computationally efficient than Bahdanau attention
    because it can be implemented using highly optimized matrix multiplication.
    """
    
    ax.text(0.02, 0.02, explanation, fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.savefig('scaled_dot_product_attention.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_multi_head_attention():
    """
    Visualize multi-head attention as used in Transformers.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Title
    plt.suptitle("Multi-Head Attention Mechanism", fontsize=16)
    
    # Setup axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Define colors
    input_color = "#AED6F1"  # Light blue
    head_colors = ["#F5B7B1", "#FADBD8", "#F9E79F", "#D5F5E3"]  # Different colors for heads
    output_color = "#D7BDE2"  # Light purple
    
    # Draw the input
    input_rect = patches.Rectangle((0.1, 0.7), 0.8, 0.1, 
                                  facecolor=input_color, edgecolor='black')
    ax.add_patch(input_rect)
    ax.text(0.5, 0.75, "Input Sequence", ha='center', fontsize=12)
    
    # Draw the projection arrows
    num_heads = 4
    head_width = 0.15
    head_height = 0.15
    head_y = 0.5
    
    for i in range(num_heads):
        head_x = 0.2 + i * 0.2
        
        # Draw an arrow from input to this head
        ax.arrow(head_x, 0.7, 0, -0.05, head_width=0.01, head_length=0.01, 
                fc='black', ec='black')
        
        # Draw the attention head
        head_rect = patches.Rectangle((head_x - head_width/2, head_y - head_height/2), 
                                     head_width, head_height, 
                                     facecolor=head_colors[i], edgecolor='black')
        ax.add_patch(head_rect)
        
        # Label the head
        ax.text(head_x, head_y, f"Head {i+1}", ha='center', va='center', fontsize=10)
        
        # Draw the Q, K, V projections
        ax.text(head_x - head_width/2 + 0.02, head_y + 0.05, "Q", fontsize=8)
        ax.text(head_x, head_y + 0.05, "K", fontsize=8)
        ax.text(head_x + head_width/2 - 0.02, head_y + 0.05, "V", fontsize=8)
        
        # Draw an arrow from this head to the concatenated output
        ax.arrow(head_x, head_y - head_height/2, 0, -0.05, head_width=0.01, head_length=0.01, 
                fc='black', ec='black')
    
    # Draw the concatenation operation
    concat_y = 0.25
    concat_rect = patches.Rectangle((0.3, concat_y - 0.05), 0.4, 0.1, 
                                   facecolor='lightgray', edgecolor='black')
    ax.add_patch(concat_rect)
    ax.text(0.5, concat_y, "Concatenate", ha='center', va='center', fontsize=10)
    
    # Draw the final linear projection
    linear_y = 0.1
    linear_rect = patches.Rectangle((0.3, linear_y - 0.05), 0.4, 0.1, 
                                   facecolor=output_color, edgecolor='black')
    ax.add_patch(linear_rect)
    ax.text(0.5, linear_y, "Linear Projection", ha='center', va='center', fontsize=10)
    
    # Connect concatenation to linear projection
    ax.arrow(0.5, concat_y - 0.05, 0, -0.05, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Add formula
    formula = r"$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$"
    ax.text(0.5, 0.95, formula, ha='center', fontsize=14)
    
    # Add sub-formula
    sub_formula = r"$where\ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$"
    ax.text(0.5, 0.9, sub_formula, ha='center', fontsize=12)
    
    # Add explanation
    explanation = """
    Multi-Head Attention (2017) from the Transformer paper:
    
    1. Project the input into multiple sets of Query (Q), Key (K), and Value (V) matrices
    2. Perform scaled dot-product attention in parallel on each projection
    3. Concatenate the outputs from each attention head
    4. Apply a final linear transformation to produce the final output
    
    This allows the model to jointly attend to information from different representation subspaces.
    """
    
    ax.text(0.05, 0.02, explanation, fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_self_attention_vs_cross_attention():
    """
    Visualize the difference between self-attention and cross-attention.
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Titles
    ax1.set_title("Self-Attention", fontsize=14)
    ax2.set_title("Cross-Attention", fontsize=14)
    
    # Turn off axes
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    # Setup axes limits
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Define colors
    sequence_color = "#AED6F1"  # Light blue
    other_sequence_color = "#A2D9CE"  # Light green
    attention_color = "#F5B7B1"  # Light red
    
    # Self-Attention diagram
    # Draw sequence tokens
    token_radius = 0.05
    token_positions = np.linspace(0.2, 0.8, 5)
    token_y = 0.7
    
    for i, x in enumerate(token_positions):
        circle = plt.Circle((x, token_y), token_radius, color=sequence_color, ec='black')
        ax1.add_patch(circle)
        ax1.text(x, token_y, f"t{i+1}", ha='center', va='center')
    
    # Draw self-attention connections
    # Token 3 attends to all other tokens
    focus_token_idx = 2  # t3
    focus_x = token_positions[focus_token_idx]
    
    # Draw attention weights
    for i, x in enumerate(token_positions):
        if i != focus_token_idx:
            # Calculate a random attention weight
            weight = np.random.rand() * 0.8 + 0.2
            
            # Draw the connection with width proportional to weight
            arrow_width = weight * 2
            ax1.arrow(focus_x, token_y - token_radius, 
                     x - focus_x, 0, 
                     head_width=0.02, head_length=0.02, 
                     width=arrow_width * 0.005,
                     fc=attention_color, ec=attention_color, 
                     alpha=weight, length_includes_head=True)
            
            # Show the weight
            ax1.text((focus_x + x) / 2, token_y - token_radius - 0.05, 
                    f"{weight:.2f}", ha='center', va='top', color='red')
    
    # Circle the focus token
    highlight_circle = plt.Circle((focus_x, token_y), token_radius + 0.01, 
                                 fill=False, ec='red', linewidth=2)
    ax1.add_patch(highlight_circle)
    
    # Add explanation
    ax1.text(0.5, 0.2, "In self-attention, tokens attend to\nother tokens in the same sequence.", 
           ha='center', fontsize=10)
    ax1.text(0.5, 0.1, "This allows the model to capture\nrelationships within a sequence.", 
           ha='center', fontsize=10)
    
    # Cross-Attention diagram
    # Draw source sequence tokens
    src_token_positions = np.linspace(0.2, 0.8, 5)
    src_token_y = 0.8
    
    for i, x in enumerate(src_token_positions):
        circle = plt.Circle((x, src_token_y), token_radius, color=other_sequence_color, ec='black')
        ax2.add_patch(circle)
        ax2.text(x, src_token_y, f"s{i+1}", ha='center', va='center')
    
    # Draw target sequence tokens
    tgt_token_positions = np.linspace(0.3, 0.7, 3)
    tgt_token_y = 0.4
    
    for i, x in enumerate(tgt_token_positions):
        circle = plt.Circle((x, tgt_token_y), token_radius, color=sequence_color, ec='black')
        ax2.add_patch(circle)
        ax2.text(x, tgt_token_y, f"t{i+1}", ha='center', va='center')
    
    # Draw cross-attention connections
    # Target token 2 attends to source tokens
    focus_tgt_idx = 1  # t2
    focus_tgt_x = tgt_token_positions[focus_tgt_idx]
    
    # Draw attention weights
    for i, x in enumerate(src_token_positions):
        # Calculate a random attention weight
        weight = np.random.rand() * 0.8 + 0.2
        
        # Draw the connection with width proportional to weight
        arrow_width = weight * 2
        ax2.arrow(focus_tgt_x, tgt_token_y + token_radius, 
                 x - focus_tgt_x, src_token_y - tgt_token_y - 2*token_radius, 
                 head_width=0.02, head_length=0.02, 
                 width=arrow_width * 0.005,
                 fc=attention_color, ec=attention_color, 
                 alpha=weight, length_includes_head=True)
        
        # Show the weight
        mid_x = (focus_tgt_x + x) / 2
        mid_y = (tgt_token_y + token_radius + src_token_y - token_radius) / 2
        ax2.text(mid_x, mid_y, f"{weight:.2f}", ha='center', va='center', color='red')
    
    # Circle the focus token
    highlight_circle = plt.Circle((focus_tgt_x, tgt_token_y), token_radius + 0.01, 
                                 fill=False, ec='red', linewidth=2)
    ax2.add_patch(highlight_circle)
    
    # Add explanation
    ax2.text(0.5, 0.2, "In cross-attention, tokens from one sequence\nattend to tokens in another sequence.", 
           ha='center', fontsize=10)
    ax2.text(0.5, 0.1, "This allows the model to capture relationships\nbetween different sequences (e.g., encoder-decoder).", 
           ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('self_vs_cross_attention.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Visualizing Bahdanau attention...")
    visualize_bahdanau_attention()
    
    print("Visualizing scaled dot-product attention...")
    visualize_scaled_dot_product_attention()
    
    print("Visualizing multi-head attention...")
    visualize_multi_head_attention()
    
    print("Visualizing self-attention vs cross-attention...")
    visualize_self_attention_vs_cross_attention()
    
    print("Visualization complete! Check the output images.")
