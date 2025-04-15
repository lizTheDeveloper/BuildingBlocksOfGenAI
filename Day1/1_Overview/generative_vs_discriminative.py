"""
Generative vs Discriminative Models Visualization
Building Blocks of Generative AI Course - Day 1

This script visualizes the difference between generative and discriminative approaches
to classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(42)

def visualize_generative_vs_discriminative():
    """Visualize the difference between generative and discriminative models"""
    # Create a simple 2D dataset with two classes
    X, y = make_blobs(n_samples=300, centers=2, random_state=42)
    
    # Create a figure
    plt.figure(figsize=(15, 6))
    
    # Plot original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolor='k', s=80)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Generate grid for the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Discriminative approach (classify directly based on decision boundary)
    grid = np.c_[xx.ravel(), yy.ravel()]
    discriminative_boundary = np.zeros(len(grid))
    
    # Simple linear boundary for demonstration
    slope = 1.2
    intercept = 0
    for i, point in enumerate(grid):
        discriminative_boundary[i] = 1 if point[1] > slope * point[0] + intercept else 0
        
    # Generative approach (model the distribution of each class)
    # For demonstration, we'll use a simple multivariate normal for each class
    means = [X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)]
    covs = [np.cov(X[y == 0].T), np.cov(X[y == 1].T)]
    
    # Calculate log probability for each class
    generative_probs = np.zeros((len(grid), 2))
    for i, point in enumerate(grid):
        for c in range(2):
            # Mahalanobis distance (log probability)
            diff = point - means[c]
            generative_probs[i, c] = -0.5 * diff.dot(np.linalg.inv(covs[c])).dot(diff)
    
    generative_boundary = np.argmax(generative_probs, axis=1)
    
    # Plot discriminative vs generative boundaries
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, edgecolor='k', s=80)
    
    # Plot the decision boundaries
    plt.contour(xx, yy, discriminative_boundary.reshape(xx.shape), 
                levels=[0.5], colors='red', linestyles='--', linewidths=2)
    plt.contour(xx, yy, generative_boundary.reshape(xx.shape), 
                levels=[0.5], colors='blue', linewidths=2)
    
    plt.title('Discriminative vs Generative Approach')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(['Class 0', 'Class 1', 'Discriminative Boundary', 'Generative Boundary'], 
              loc='lower right')
    
    plt.tight_layout()
    plt.savefig('generative_vs_discriminative.png')
    plt.show()

if __name__ == "__main__":
    visualize_generative_vs_discriminative()
    print("Visualization complete! Check the output image: generative_vs_discriminative.png")
