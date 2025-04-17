"""
GAN Theory Visualization
Building Blocks of Generative AI Course - Day 2

This script visualizes the adversarial process in GANs by showing the generator
and discriminator distributions evolving over training iterations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# Set random seed for reproducibility
np.random.seed(42)

def visualize_gan_training():
    """
    Visualize the GAN training process with evolving distributions.
    Shows how generator distribution gets closer to the real data distribution
    while the discriminator tries to differentiate between them.
    """
    # Parameters
    num_iterations = 6
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
    
    # Set up subplots for different aspects of GAN training
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    ax4 = plt.subplot(gs[2, :])
    
    # Distribution parameters
    real_mean = 4.0
    real_std = 1.0
    
    # Create x-axis for plotting distributions
    x = np.linspace(-2, 10, 1000)
    
    # Real data distribution (Gaussian)
    real_dist = np.exp(-0.5 * ((x - real_mean) / real_std) ** 2) / (real_std * np.sqrt(2 * np.pi))
    
    # Generator starting distribution (Gaussian, but wrong mean)
    gen_means = np.linspace(0, real_mean, num_iterations)
    gen_stds = np.linspace(2.0, real_std, num_iterations)
    
    # Plot real data distribution (fixed)
    ax1.plot(x, real_dist, 'b', linewidth=2, label='Real Data Distribution')
    ax1.fill_between(x, real_dist, alpha=0.3, color='blue')
    ax1.set_title('Real Data Distribution')
    ax1.legend()
    ax1.set_xlabel('Data Value')
    ax1.set_ylabel('Probability Density')
    
    # Create animation frames for the training process
    for i in range(num_iterations):
        # Generator distribution at this iteration
        gen_dist = np.exp(-0.5 * ((x - gen_means[i]) / gen_stds[i]) ** 2) / (gen_stds[i] * np.sqrt(2 * np.pi))
        
        # Clear previous generator plot
        ax2.clear()
        
        # Plot generator distribution
        ax2.plot(x, gen_dist, 'g', linewidth=2, label=f'Generator (Iteration {i+1})')
        ax2.fill_between(x, gen_dist, alpha=0.3, color='green')
        ax2.set_title(f'Generator Distribution (Iteration {i+1})')
        ax2.legend()
        ax2.set_xlabel('Data Value')
        ax2.set_ylabel('Probability Density')
        
        # Calculate discriminator response
        # A simple sigmoid to represent discriminator's classification of real vs fake
        discriminator = 1 / (1 + np.exp(-(x - (gen_means[i] + (real_mean - gen_means[i]) / 2))))
        
        # Clear previous discriminator plot
        ax3.clear()
        
        # Plot discriminator response
        ax3.plot(x, discriminator, 'r', linewidth=2, label='Discriminator Output')
        ax3.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Decision Boundary')
        ax3.set_title(f'Discriminator Response (Iteration {i+1})')
        ax3.legend()
        ax3.set_xlabel('Data Value')
        ax3.set_ylabel('Probability of Real')
        ax3.set_ylim(0, 1)
        
        # Clear previous combined plot
        ax4.clear()
        
        # Plot both distributions together to show progress
        ax4.plot(x, real_dist, 'b', linewidth=2, label='Real Data Distribution')
        ax4.plot(x, gen_dist, 'g', linewidth=2, label=f'Generator (Iteration {i+1})')
        ax4.fill_between(x, real_dist, alpha=0.15, color='blue')
        ax4.fill_between(x, gen_dist, alpha=0.15, color='green')
        
        # Highlight the area where distributions overlap
        min_dist = np.minimum(gen_dist, real_dist)
        ax4.fill_between(x, min_dist, alpha=0.5, color='purple')
        
        # Show JS divergence approximation
        js_div = np.sum((real_dist - gen_dist) ** 2) / len(x)
        ax4.set_title(f'Distribution Comparison (Iteration {i+1})\nDivergence: {js_div:.4f}')
        ax4.legend()
        ax4.set_xlabel('Data Value')
        ax4.set_ylabel('Probability Density')
        
        # Save the current state as an image for each iteration
        plt.tight_layout()
        plt.savefig(f'gan_training_iteration_{i+1}.png')
        
        # Pause to show the progress
        plt.pause(1.0)
    
    # Show the final plot
    plt.tight_layout()
    plt.savefig('gan_training_final.png')
    plt.show()

def plot_gan_architecture():
    """Plot a diagram of the GAN architecture"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Hide axes
    ax.set_axis_off()
    
    # Draw components
    # Latent Space
    ax.text(0.1, 0.8, "Latent Space\n(Random Noise)", 
           ha="center", va="center", fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="black"))
    
    # Generator
    ax.text(0.1, 0.5, "Generator\nNetwork", 
           ha="center", va="center", fontsize=14, 
           bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="black"))
    
    # Fake Data
    ax.text(0.3, 0.5, "Generated\nData", 
           ha="center", va="center", fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="black"))
    
    # Real Data
    ax.text(0.3, 0.2, "Real\nData", 
           ha="center", va="center", fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="black"))
    
    # Discriminator
    ax.text(0.5, 0.35, "Discriminator\nNetwork", 
           ha="center", va="center", fontsize=14, 
           bbox=dict(boxstyle="round,pad=0.5", fc="salmon", ec="black"))
    
    # Classification
    ax.text(0.7, 0.35, "Classification\nReal/Fake", 
           ha="center", va="center", fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="black"))
    
    # Add arrows for connections
    ax.arrow(0.15, 0.8, 0, -0.2, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.15, 0.5, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.35, 0.5, 0.1, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.35, 0.2, 0.1, 0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.55, 0.35, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Add feedback loop for generator training
    ax.arrow(0.7, 0.4, -0.2, 0.15, head_width=0.02, head_length=0.02, fc='blue', ec='blue', linestyle='--')
    ax.text(0.6, 0.5, "Update\nGenerator", color='blue', ha="center", va="center", fontsize=10)
    
    # Add feedback loop for discriminator training
    ax.arrow(0.7, 0.3, -0.2, -0.1, head_width=0.02, head_length=0.02, fc='red', ec='red', linestyle='--')
    ax.text(0.6, 0.2, "Update\nDiscriminator", color='red', ha="center", va="center", fontsize=10)
    
    # Draw the GAN game
    ax.text(0.9, 0.7, "GAN Game:", fontsize=14, fontweight='bold')
    ax.text(0.9, 0.6, "Generator: Produce data\nthat fools the discriminator", 
           ha="center", va="center", fontsize=10, color='green')
    ax.text(0.9, 0.4, "Discriminator: Learn to\ndistinguish real from\ngenerated data", 
           ha="center", va="center", fontsize=10, color='red')
    ax.text(0.9, 0.2, "Nash Equilibrium:\nGenerator produces\nindistinguishable data", 
           ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"))
    
    plt.title("Generative Adversarial Network Architecture", fontsize=16)
    plt.tight_layout()
    plt.savefig('gan_architecture.png')
    plt.show()

if __name__ == "__main__":
    print("Visualizing GAN architecture...")
    plot_gan_architecture()
    
    print("Visualizing GAN training process...")
    visualize_gan_training()
    
    print("Visualization complete! Check the output images.")
