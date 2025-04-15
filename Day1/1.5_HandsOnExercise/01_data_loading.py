"""
Fashion MNIST Data Loading
Building Blocks of Generative AI Course - Day 1

This script loads and prepares the Fashion MNIST dataset for training.
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def load_fashion_mnist(batch_size=64):
    """
    Load the Fashion MNIST dataset and create data loaders.
    
    Args:
        batch_size: Batch size for the data loaders
        
    Returns:
        train_loader: DataLoader for the training set
        test_loader: DataLoader for the test set
        class_names: Dictionary mapping class indices to class names
    """
    # Define transformations for the training and testing data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
    ])
    
    # Load the Fashion MNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                          download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                         download=True, transform=transform)
    
    # Create data loaders for batching
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Define the class labels for Fashion MNIST
    class_names = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    
    return train_loader, test_loader, class_names

def visualize_dataset_samples(dataset, num_samples=10, class_names=None):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to visualize
        class_names: Dictionary mapping class indices to class names
    """
    fig = plt.figure(figsize=(12, 4))
    
    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]
        
        # Add subplot
        ax = fig.add_subplot(1, num_samples, i + 1)
        
        # Convert tensor to numpy and reshape for display
        img = img.numpy()[0]  # Get the first channel (grayscale)
        
        # Display the image
        ax.imshow(img, cmap='gray')
        
        # Set title to the class name if provided
        if class_names:
            ax.set_title(class_names[label])
        else:
            ax.set_title(f"Label: {label}")
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader, class_names = load_fashion_mnist()
    
    # Get the first batch
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    print(f"Batch shape: {example_data.shape}")
    print(f"Labels shape: {example_targets.shape}")
    
    # Visualize the first batch
    plt.figure(figsize=(12, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Label: {class_names[example_targets[i].item()]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
