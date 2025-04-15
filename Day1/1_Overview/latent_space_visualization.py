"""
Latent Space Visualization
Building Blocks of Generative AI Course - Day 1

This script visualizes the concept of latent space in generative models
using face image data and PCA for dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

def visualize_latent_space():
    """Visualize the concept of latent space in generative models"""
    # Load Olivetti faces dataset 
    faces = fetch_olivetti_faces()
    X = faces.data
    
    # Apply PCA to get 2D latent space for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    
    # Plot the latent space with sample images
    ax1 = plt.subplot(gs[0, :])
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=faces.target, cmap='viridis', 
                          s=100, alpha=0.8, edgecolors='w')
    ax1.set_title('2D Latent Space of Face Images')
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    
    # Add a color bar to show identity
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Identity')
    
    # Display a few original face images
    sample_indices = [0, 30, 60, 90, 120, 150]
    for i, idx in enumerate(sample_indices):
        ax_img = plt.subplot(gs[1, i % 3])
        ax_img.imshow(faces.images[idx], cmap='gray')
        ax_img.axis('off')
        ax_img.set_title(f"Image {idx}")
        
        # Add a connection line from latent point to image
        ax1.plot([X_pca[idx, 0], X_pca[idx, 0]], 
                 [X_pca[idx, 1], X_pca[idx, 1]], 
                 'r*', markersize=15)
    
    plt.tight_layout()
    plt.savefig('latent_space.png')
    plt.show()

def visualize_latent_interpolation():
    """Visualize the concept of latent space interpolation between faces"""
    # Load Olivetti faces dataset 
    faces = fetch_olivetti_faces()
    X = faces.data
    
    # Apply PCA to get 2D latent space for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Visualize the concept of latent space interpolation
    plt.figure(figsize=(15, 5))
    
    # Choose two face images to interpolate between
    face1_idx = 0
    face2_idx = 10
    
    # Get their latent representations
    z1 = X_pca[face1_idx]
    z2 = X_pca[face2_idx]
    
    # Generate points along the interpolation path
    num_steps = 5
    alphas = np.linspace(0, 1, num_steps)
    
    # Plot the original faces and interpolation in latent space
    for i, alpha in enumerate(alphas):
        # Weighted average in latent space
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # For visualization, we'll use a simple weighted average of pixels
        # (In a real VAE or GAN, we would decode from the latent space)
        face_interp = (1 - alpha) * faces.images[face1_idx] + alpha * faces.images[face2_idx]
        
        # Plot
        plt.subplot(1, num_steps, i + 1)
        plt.imshow(face_interp, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(f"Original Face 1")
        elif i == num_steps - 1:
            plt.title(f"Original Face 2")
        else:
            plt.title(f"Interpolation\nα={alpha:.2f}")
    
    plt.tight_layout()
    plt.savefig('latent_interpolation.png')
    plt.show()

if __name__ == "__main__":
    print("Visualizing latent space of face images...")
    visualize_latent_space()
    
    print("\nVisualizing latent space interpolation between faces...")
    visualize_latent_interpolation()
    
    print("\nVisualization complete! Check the output images:")
    print("- latent_space.png")
    print("- latent_interpolation.png")
