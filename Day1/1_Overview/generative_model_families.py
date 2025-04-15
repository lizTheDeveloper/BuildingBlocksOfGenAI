"""
Generative Model Families
Building Blocks of Generative AI Course - Day 1

This script visualizes the different families of generative models including:
1. Autoregressive Models
2. Variational Autoencoders (VAEs)
3. Generative Adversarial Networks (GANs)
4. Diffusion Models
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

def visualize_autoregressive_model():
    """Visualize a simple autoregressive model on time series data"""
    # Create a 1D time series
    t = np.linspace(0, 4*np.pi, 100)
    signal = np.sin(t) + 0.2 * np.sin(3*t) + np.random.normal(0, 0.1, size=len(t))
    
    # Simulate an autoregressive prediction
    ar_signal = np.zeros_like(signal)
    ar_signal[0] = signal[0]
    for i in range(1, len(signal)):
        if i < 3:
            ar_signal[i] = signal[i]
        else:
            # Simple AR(3) model for demonstration
            ar_signal[i] = 0.8 * signal[i-1] + 0.1 * signal[i-2] + 0.05 * signal[i-3] + np.random.normal(0, 0.05)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, 'b-', label='Original Signal')
    plt.plot(t, ar_signal, 'r--', label='Autoregressive Prediction')
    plt.title('Autoregressive Model')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('autoregressive_model.png')
    plt.show()

def visualize_vae_concept():
    """Visualize the concept of a Variational Autoencoder (VAE)"""
    # Create a simple moon dataset
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # Apply a simple PCA to simulate the latent space and reconstruction
    pca = PCA(n_components=2)
    X_latent = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_latent)
    
    # Add Gaussian noise to the latent space to simulate VAE sampling
    X_latent_noisy = X_latent + np.random.normal(0, 0.1, size=X_latent.shape)
    X_gen = pca.inverse_transform(X_latent_noisy)
    
    plt.figure(figsize=(10, 8))
    
    # Plot the original data and latent space
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.7, label='Original Data')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Plot the latent space
    plt.subplot(2, 2, 2)
    plt.scatter(X_latent[:, 0], X_latent[:, 1], c='red', alpha=0.7, label='Latent Space')
    plt.title('Latent Space (Encoding)')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    
    # Plot the reconstructed data
    plt.subplot(2, 2, 3)
    plt.scatter(X_recon[:, 0], X_recon[:, 1], c='green', alpha=0.7, label='Reconstructed')
    plt.title('Reconstructed Data (Decoding)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Plot the generated data (from noisy latent)
    plt.subplot(2, 2, 4)
    plt.scatter(X_gen[:, 0], X_gen[:, 1], c='purple', alpha=0.7, label='Generated')
    plt.title('Generated Data (Sampling)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_concept.png')
    plt.show()

def visualize_gan_concept():
    """Visualize the concept of a Generative Adversarial Network (GAN)"""
    # Create a simple moon dataset
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # Simulate generated samples with an offset and different distribution
    gen_samples = X + np.random.normal(0, 0.15, size=X.shape) + np.array([0.5, 0.5])
    
    plt.figure(figsize=(10, 8))
    
    # Plot the real and generated data
    plt.scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.7, label='Real Data')
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], c='green', s=50, alpha=0.7, label='Generated Data')
    
    # Simulate discriminator decision boundary
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Simple decision boundary for demonstration
    def discriminator_boundary(x, y):
        # This is just a visual approximation of what a discriminator might learn
        center_x, center_y = np.mean(X, axis=0)
        return 0.5 * ((x - center_x)**2 + (y - center_y)**2)
    
    zz = np.array([discriminator_boundary(p[0], p[1]) for p in grid])
    plt.contour(xx, yy, zz.reshape(xx.shape), levels=[0.4], colors='red', linewidths=2)
    
    plt.title('Generative Adversarial Network (GAN) Concept')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(['Real Data', 'Generated Data', 'Discriminator Boundary'])
    plt.grid(True, alpha=0.3)
    plt.savefig('gan_concept.png')
    plt.show()

def visualize_diffusion_concept():
    """Visualize the concept of a Diffusion Model"""
    # Create a simple pattern
    t = np.linspace(0, 1, 100)
    x_pattern = np.sin(2 * np.pi * t)
    y_pattern = np.cos(2 * np.pi * t)
    
    plt.figure(figsize=(10, 8))
    
    # Plot the original pattern
    plt.plot(x_pattern, y_pattern, 'b-', linewidth=3, label='Original Pattern')
    
    # Visualize different noise levels (diffusion process)
    noise_levels = [0.1, 0.3, 0.6, 1.0]
    for i, noise in enumerate(noise_levels):
        x_noisy = x_pattern + np.random.normal(0, noise, size=len(x_pattern))
        y_noisy = y_pattern + np.random.normal(0, noise, size=len(y_pattern))
        plt.plot(x_noisy, y_noisy, 'o', alpha=0.5, markersize=2, label=f'Noise Level {noise}')
    
    plt.title('Diffusion Model Concept')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('diffusion_concept.png')
    plt.show()

if __name__ == "__main__":
    print("1. Visualizing Autoregressive Model concept...")
    visualize_autoregressive_model()
    
    print("\n2. Visualizing Variational Autoencoder (VAE) concept...")
    visualize_vae_concept()
    
    print("\n3. Visualizing Generative Adversarial Network (GAN) concept...")
    visualize_gan_concept()
    
    print("\n4. Visualizing Diffusion Model concept...")
    visualize_diffusion_concept()
    
    print("\nVisualization complete! Check the output images:")
    print("- autoregressive_model.png")
    print("- vae_concept.png")
    print("- gan_concept.png")
    print("- diffusion_concept.png")
