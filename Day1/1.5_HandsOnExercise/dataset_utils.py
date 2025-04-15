"""
Dataset Utilities for Hands-On Neural Network Exercise
Building Blocks of Generative AI Course - Day 1

This module provides utility functions for loading and visualizing datasets
for the hands-on neural network exercise.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# Define the class labels for Fashion MNIST
FASHION_MNIST_LABELS = {
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

def load_fashion_mnist(batch_size=64):
    """
    Load the Fashion MNIST dataset and create data loaders.
    
    Args:
        batch_size: Batch size for training
        
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
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
    
    return train_loader, test_loader

def visualize_dataset_samples(dataset, num_samples=10, class_names=None):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: The dataset to visualize
        num_samples: Number of samples to display
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

def visualize_predictions(model, device, test_loader, class_names, num_samples=10):
    """
    Visualize model predictions on random test samples.
    
    Args:
        model: Trained model
        device: Device to run the model on (CPU or CUDA)
        test_loader: DataLoader for test set
        class_names: Dictionary mapping class indices to class names
        num_samples: Number of samples to display
    """
    model.eval()  # Set the model to evaluation mode
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Select a subset of images
    indices = np.random.choice(len(images), num_samples, replace=False)
    images = images[indices].to(device)
    labels = labels[indices].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Plot the images and their predictions
    fig = plt.figure(figsize=(15, 6))
    
    for i in range(num_samples):
        # Plot original image
        ax = fig.add_subplot(2, num_samples, i + 1)
        img = images[i].cpu().numpy()[0]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {class_names[labels[i].item()]}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot prediction
        ax = fig.add_subplot(2, num_samples, i + 1 + num_samples)
        ax.imshow(img, cmap='gray')
        # Color the title based on correctness
        title_color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {class_names[predicted[i].item()]}", color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def show_available_datasets():
    """
    Print information about available datasets for the exercise.
    """
    datasets_info = [
        {
            "name": "Fashion MNIST",
            "description": "70,000 grayscale images (28x28 pixels) of clothing items across 10 categories",
            "ideal_for": "Convolutional Neural Networks (CNNs), classification tasks",
            "availability": "torchvision.datasets.FashionMNIST"
        },
        {
            "name": "MNIST",
            "description": "70,000 grayscale images (28x28 pixels) of handwritten digits (0-9)",
            "ideal_for": "Basic image classification, introduction to CNNs",
            "availability": "torchvision.datasets.MNIST"
        },
        {
            "name": "CIFAR-10",
            "description": "60,000 color images (32x32 pixels) across 10 classes (airplanes, cars, birds, etc.)",
            "ideal_for": "More complex image classification tasks",
            "availability": "torchvision.datasets.CIFAR10"
        },
        {
            "name": "Digits Dataset (Scikit-learn)",
            "description": "1,797 images of handwritten digits (8x8 pixels)",
            "ideal_for": "Quick prototyping, introductory ML models",
            "availability": "sklearn.datasets.load_digits()"
        },
        {
            "name": "California Housing Dataset",
            "description": "Housing price data with features like median income, house age, etc.",
            "ideal_for": "Regression tasks, predicting continuous values",
            "availability": "sklearn.datasets.fetch_california_housing()"
        },
        {
            "name": "IMDb Movie Reviews",
            "description": "50,000 movie reviews labeled as positive or negative",
            "ideal_for": "Text classification, sentiment analysis, NLP models",
            "availability": "torchtext.datasets.IMDB"
        },
        {
            "name": "Breast Cancer Wisconsin Dataset",
            "description": "569 instances with 30 attributes for tumor classification (malignant/benign)",
            "ideal_for": "Binary classification, medical diagnostics",
            "availability": "sklearn.datasets.load_breast_cancer()"
        }
    ]
    
    print("=" * 80)
    print("Available Datasets for the Neural Network Exercise")
    print("=" * 80)
    
    for i, dataset in enumerate(datasets_info, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Ideal For: {dataset['ideal_for']}")
        print(f"   Availability: {dataset['availability']}")
        print("-" * 80)

if __name__ == "__main__":
    # Show available datasets
    show_available_datasets()
    
    # Load and visualize Fashion MNIST dataset
    train_loader, test_loader = load_fashion_mnist()
    
    # Visualize some examples
    print("\nDisplaying random samples from Fashion MNIST:")
    visualize_dataset_samples(train_loader.dataset, class_names=FASHION_MNIST_LABELS)
