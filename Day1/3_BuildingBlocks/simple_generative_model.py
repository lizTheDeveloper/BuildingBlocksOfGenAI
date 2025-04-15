"""
Simple Generative Model Exercise - Gaussian Mixture Model
Building Blocks of Generative AI Course - Day 1

This script demonstrates a simple generative model using a Gaussian Mixture Model (GMM)
to generate synthetic data points.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_clustered_data(n_samples=1000):
    """Generate synthetic data with 3 clusters to train our GMM"""
    # Create 3 cluster centers
    centers = [
        (0, 0),  # Cluster 1 center
        (5, 5),  # Cluster 2 center
        (0, 5)   # Cluster 3 center
    ]
    
    # Standard deviations for each cluster
    stds = [1.0, 1.5, 0.8]
    
    # Generate sample counts for each cluster
    cluster_sizes = [int(n_samples * 0.4), int(n_samples * 0.3), int(n_samples * 0.3)]
    
    # Generate the data
    data = []
    labels = []
    
    for i, (center, std, size) in enumerate(zip(centers, stds, cluster_sizes)):
        x = np.random.normal(center[0], std, size)
        y = np.random.normal(center[1], std, size)
        data.append(np.column_stack([x, y]))
        labels.extend([i] * size)
    
    return np.vstack(data), np.array(labels)

def train_gmm(data, n_components=3):
    """Train a Gaussian Mixture Model on the provided data"""
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    return gmm

def generate_samples(gmm, n_samples=1000):
    """Generate new samples using the trained GMM"""
    samples, _ = gmm.sample(n_samples)
    return samples

def plot_data_and_samples(original_data, original_labels, generated_samples):
    """Visualize original and generated data"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original data
    scatter = axes[0].scatter(original_data[:, 0], original_data[:, 1], c=original_labels, 
                  alpha=0.6, cmap='viridis')
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    legend1 = axes[0].legend(*scatter.legend_elements(),
                            title="Clusters")
    axes[0].add_artist(legend1)
    
    # Plot generated samples
    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                   alpha=0.6, color='coral')
    axes[1].set_title("Generated Samples")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Set same axis limits for comparison
    xlim = [min(original_data[:, 0].min(), generated_samples[:, 0].min()) - 1,
            max(original_data[:, 0].max(), generated_samples[:, 0].max()) + 1]
    ylim = [min(original_data[:, 1].min(), generated_samples[:, 1].min()) - 1,
            max(original_data[:, 1].max(), generated_samples[:, 1].max()) + 1]
    
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig('gmm_comparison.png')
    plt.show()

def visualize_gmm_components(gmm, data):
    """Visualize the learned GMM components over the original data"""
    plt.figure(figsize=(10, 8))
    
    # Create a meshgrid to plot the decision boundaries
    x = np.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 100)
    y = np.linspace(data[:, 1].min() - 1, data[:, 1].max() + 1, 100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    
    # Predict probabilities on the meshgrid
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)
    
    # Plot the data points
    plt.scatter(data[:, 0], data[:, 1], s=30, alpha=0.4, color='coral')
    
    # Plot contour for the GMM's learned decision boundaries
    levels = np.linspace(Z.min(), Z.max(), 10)
    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    
    # Plot the means of each Gaussian component
    means = gmm.means_
    plt.scatter(means[:, 0], means[:, 1], s=200, color='red', marker='*',
                edgecolor='black', linewidth=1.5, label='GMM Means')
    
    # Add visualizations for covariances
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        # For full covariance matrices
        if covar.shape == (2, 2):  
            v, w = np.linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = plt.matplotlib.patches.Ellipse(
                mean, v[0], v[1], 180 + angle, color='black')
            ell.set_alpha(0.3)
            plt.gca().add_artist(ell)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('GMM Components and Density Contours')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2') 
    plt.legend()
    plt.tight_layout()
    plt.savefig('gmm_components.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Generating synthetic data...")
    data, labels = generate_clustered_data(n_samples=1000)
    
    print("Training Gaussian Mixture Model...")
    gmm = train_gmm(data, n_components=3)
    
    print("Generating new samples from the trained model...")
    samples = generate_samples(gmm, n_samples=1000)
    
    print("Visualizing results...")
    plot_data_and_samples(data, labels, samples)
    visualize_gmm_components(gmm, data)
    
    print("Exercise complete!")
