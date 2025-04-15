"""
Probability Distributions and Sampling Techniques
Building Blocks of Generative AI Course - Day 1

This script demonstrates various probability distributions and sampling techniques
that are fundamental to generative models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def visualize_common_distributions():
    """Visualize common probability distributions used in generative models"""
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # 1. Gaussian (Normal) Distribution
    plt.subplot(2, 2, 1)
    x = np.linspace(-5, 5, 1000)
    for mu, sigma in [(0, 1), (0, 0.5), (2, 1)]:
        y = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, y, label=f'μ={mu}, σ={sigma}')
    plt.title('Gaussian (Normal) Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Multivariate Gaussian
    plt.subplot(2, 2, 2)
    x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    pos = np.dstack((x, y))
    
    # Standard multivariate normal
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    rv = stats.multivariate_normal(mean, cov)
    z = rv.pdf(pos)
    
    plt.contourf(x, y, z, levels=20, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.title('Multivariate Gaussian Distribution')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    
    # 3. Mixture of Gaussians
    plt.subplot(2, 2, 3)
    xs = np.linspace(-10, 10, 1000)
    # Define mixture components
    mus = [-3, 0, 3]
    sigmas = [1, 0.5, 1.5]
    weights = [0.3, 0.4, 0.3]
    
    # Plot each component
    y = np.zeros_like(xs)
    for i, (mu, sigma, weight) in enumerate(zip(mus, sigmas, weights)):
        component = stats.norm.pdf(xs, mu, sigma) * weight
        plt.plot(xs, component, '--', label=f'Component {i+1}', alpha=0.6)
        y += component
    
    # Plot mixture
    plt.plot(xs, y, 'k-', label='Mixture')
    plt.title('Gaussian Mixture Model')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Uniform Distribution
    plt.subplot(2, 2, 4)
    x = np.linspace(-1, 3, 1000)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        y = stats.uniform.pdf(x, a, b-a)
        plt.plot(x, y, label=f'Uniform({a}, {b})')
    plt.title('Uniform Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("common_distributions.png")
    plt.show()

def demonstrate_sampling_techniques():
    """Demonstrate different sampling techniques used in generative models"""
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # 1. Direct Sampling from Gaussian
    plt.subplot(2, 2, 1)
    samples = np.random.normal(0, 1, 1000)
    plt.hist(samples, bins=30, density=True, alpha=0.7)
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
    plt.title('Direct Sampling from Gaussian')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    
    # 2. Rejection Sampling
    plt.subplot(2, 2, 2)
    
    # Target distribution: Bimodal Gaussian mixture
    def target_pdf(x):
        return 0.5 * stats.norm.pdf(x, -2, 0.8) + 0.5 * stats.norm.pdf(x, 2, 0.8)
    
    # Proposal distribution: Gaussian
    def proposal_pdf(x):
        return stats.norm.pdf(x, 0, 3)
    
    # Find scaling factor M
    x_range = np.linspace(-6, 6, 1000)
    M = max(target_pdf(x_range) / proposal_pdf(x_range)) * 1.2  # Add margin
    
    # Rejection sampling
    accepted_samples = []
    proposal_samples = []
    accept_idx = []
    reject_idx = []
    
    # Generate samples
    for i in range(5000):
        # Sample from proposal
        x = np.random.normal(0, 3)
        proposal_samples.append(x)
        
        # Accept or reject
        u = np.random.uniform(0, 1)
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            accepted_samples.append(x)
            accept_idx.append(i)
        else:
            reject_idx.append(i)
    
    # Plot results
    plt.hist(accepted_samples, bins=30, density=True, alpha=0.7)
    plt.plot(x_range, target_pdf(x_range), 'r-', lw=2, label='Target')
    plt.plot(x_range, M * proposal_pdf(x_range), 'g--', lw=2, label='Scaled Proposal')
    plt.title('Rejection Sampling')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Importance Sampling
    plt.subplot(2, 2, 3)
    
    # Target: Bimodal Gaussian mixture (same as before)
    # Proposal: Gaussian (same as before)
    
    # Generate proposal samples
    proposal_samples = np.random.normal(0, 3, 1000)
    
    # Calculate importance weights
    weights = target_pdf(proposal_samples) / proposal_pdf(proposal_samples)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Plot weighted histogram
    plt.hist(proposal_samples, bins=30, weights=weights, density=True, alpha=0.7)
    plt.plot(x_range, target_pdf(x_range), 'r-', lw=2, label='Target')
    plt.plot(x_range, proposal_pdf(x_range), 'g--', lw=2, label='Proposal')
    plt.title('Importance Sampling')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. MCMC Sampling (Metropolis-Hastings)
    plt.subplot(2, 2, 4)
    
    # Target: Bimodal Gaussian mixture (unnormalized)
    def target_unnormalized(x):
        return np.exp(-(x + 2)**2 / (2 * 0.8**2)) + np.exp(-(x - 2)**2 / (2 * 0.8**2))
    
    # Metropolis-Hastings algorithm
    def metropolis_hastings(n_samples, proposal_width=1.0):
        samples = np.zeros(n_samples)
        # Start at a random point
        current = np.random.normal(0, 1)
        
        for i in range(n_samples):
            # Propose a new point
            proposal = current + np.random.normal(0, proposal_width)
            
            # Calculate acceptance probability
            current_prob = target_unnormalized(current)
            proposal_prob = target_unnormalized(proposal)
            
            # Accept or reject
            if np.random.uniform(0, 1) < proposal_prob / current_prob:
                current = proposal
            
            samples[i] = current
            
        return samples
    
    # Generate MCMC samples
    mcmc_samples = metropolis_hastings(10000)
    
    # Plot results
    plt.hist(mcmc_samples, bins=30, density=True, alpha=0.7)
    plt.plot(x_range, target_pdf(x_range) / max(target_pdf(x_range)) * max(plt.gca().get_ylim()), 
             'r-', lw=2, label='Target (scaled)')
    plt.title('MCMC Sampling (Metropolis-Hastings)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("sampling_techniques.png")
    plt.show()

def demonstrate_reparameterization_trick():
    """
    Demonstrate the reparameterization trick used in VAEs
    
    The reparameterization trick allows us to backpropagate through a random sampling
    operation by expressing it as a deterministic function of the distribution
    parameters and a separate noise variable.
    """
    plt.figure(figsize=(15, 10))
    
    # Define original distribution parameters
    mu_original = 2.0
    sigma_original = 1.5
    
    # Generate grid for visualizing distributions
    x = np.linspace(-5, 10, 1000)
    
    # Original distribution
    plt.subplot(2, 2, 1)
    plt.plot(x, stats.norm.pdf(x, mu_original, sigma_original))
    plt.axvline(mu_original, color='r', linestyle='--', label='μ')
    plt.axvline(mu_original - sigma_original, color='g', linestyle=':', label='μ-σ')
    plt.axvline(mu_original + sigma_original, color='g', linestyle=':', label='μ+σ')
    plt.title(f'Original Distribution N({mu_original}, {sigma_original}²)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Direct sampling (not differentiable)
    plt.subplot(2, 2, 2)
    direct_samples = np.random.normal(mu_original, sigma_original, 1000)
    plt.hist(direct_samples, bins=30, density=True, alpha=0.7)
    plt.plot(x, stats.norm.pdf(x, mu_original, sigma_original), 'r-', lw=2)
    plt.title('Direct Sampling (Not Differentiable)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    
    # Reparameterization trick
    plt.subplot(2, 2, 3)
    
    # Standard normal noise (ε ~ N(0, 1))
    epsilon = np.random.normal(0, 1, 1000)
    
    # Reparameterized samples: x = μ + σ * ε
    reparam_samples = mu_original + sigma_original * epsilon
    
    plt.hist(reparam_samples, bins=30, density=True, alpha=0.7)
    plt.plot(x, stats.norm.pdf(x, mu_original, sigma_original), 'r-', lw=2)
    plt.title('Reparameterized Sampling (x = μ + σ * ε)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    
    # Visualize gradient flow with different mu values
    plt.subplot(2, 2, 4)
    
    # Generate samples with different mu values but same noise
    mu_values = [0.0, 2.0, 4.0]
    
    # Keep the same noise values for all
    epsilon_fixed = np.random.normal(0, 1, 1000)
    
    for mu in mu_values:
        # Get samples using the same epsilon but different mu
        samples = mu + sigma_original * epsilon_fixed
        
        # Plot the resulting distribution
        plt.hist(samples, bins=30, density=True, alpha=0.5, label=f'μ={mu}')
    
    plt.title('Effect of μ with Fixed Noise (ε)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reparameterization_trick.png")
    plt.show()

def demonstrate_latent_manifold_learning():
    """
    Demonstrate the concept of latent manifold learning in generative models
    using a 2D toy example with Swiss Roll data.
    """
    plt.figure(figsize=(15, 12))
    
    # 1. Generate Swiss Roll data
    n_samples = 1000
    noise = 0.1
    
    # Swiss roll in 3D space
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n_samples)
    z = t * np.sin(t)
    
    data_3d = np.vstack((x, y, z)).T
    data_3d += noise * np.random.randn(n_samples, 3)
    
    # 2. Visualize 3D data
    from mpl_toolkits.mplot3d import Axes3D
    
    ax = plt.subplot(2, 2, 1, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], 
               c=t, cmap=plt.cm.viridis)
    ax.set_title('Original 3D Data (Swiss Roll)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 3. PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_3d)
    
    ax = plt.subplot(2, 2, 2)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=t, cmap=plt.cm.viridis)
    plt.title('PCA (Linear Dimensionality Reduction)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    
    # 4. t-SNE for non-linear dimensionality reduction
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data_3d)
    
    ax = plt.subplot(2, 2, 3)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=t, cmap=plt.cm.viridis)
    plt.title('t-SNE (Non-Linear Dimensionality Reduction)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    
    # 5. UMAP for non-linear dimensionality reduction
    try:
        import umap
        
        um = umap.UMAP(n_components=2, random_state=42)
        data_umap = um.fit_transform(data_3d)
        
        ax = plt.subplot(2, 2, 4)
        plt.scatter(data_umap[:, 0], data_umap[:, 1], c=t, cmap=plt.cm.viridis)
        plt.title('UMAP (Non-Linear Dimensionality Reduction)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
    except ImportError:
        # If UMAP is not installed, use MDS instead
        from sklearn.manifold import MDS
        
        mds = MDS(n_components=2, random_state=42)
        data_mds = mds.fit_transform(data_3d)
        
        ax = plt.subplot(2, 2, 4)
        plt.scatter(data_mds[:, 0], data_mds[:, 1], c=t, cmap=plt.cm.viridis)
        plt.title('MDS (Non-Linear Dimensionality Reduction)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("latent_manifold_learning.png")
    plt.show()

if __name__ == "__main__":
    print("Visualizing common probability distributions...")
    visualize_common_distributions()
    
    print("\nDemonstrating sampling techniques...")
    demonstrate_sampling_techniques()
    
    print("\nDemonstrating the reparameterization trick...")
    demonstrate_reparameterization_trick()
    
    print("\nDemonstrating latent manifold learning...")
    demonstrate_latent_manifold_learning()
    
    print("\nVisualization complete! Check the output images:")
    print("- common_distributions.png")
    print("- sampling_techniques.png")
    print("- reparameterization_trick.png")
    print("- latent_manifold_learning.png")
