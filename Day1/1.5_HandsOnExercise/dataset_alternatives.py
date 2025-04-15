"""
Alternative Datasets for Neural Network Exercises
Building Blocks of Generative AI Course - Day 1

This script provides examples of how to load and visualize alternative datasets
that can be used for the neural network exercises.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets as sklearn_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_fashion_mnist():
    """Load Fashion MNIST dataset"""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                          download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                                         download=True, transform=transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Class names
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
    
    return train_loader, test_loader, train_dataset, test_dataset, class_names

def load_cifar10():
    """Load CIFAR-10 dataset"""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Class names
    class_names = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    
    return train_loader, test_loader, train_dataset, test_dataset, class_names

def load_sklearn_digits():
    """Load scikit-learn Digits dataset"""
    # Load dataset
    digits = sklearn_datasets.load_digits()
    
    # Prepare data
    X = digits.data.astype('float32') / 16.0  # Normalize pixel values to [0,1]
    y = digits.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Class names
    class_names = {i: str(i) for i in range(10)}
    
    return train_loader, test_loader, train_dataset, test_dataset, class_names

def load_california_housing():
    """Load California Housing dataset (for regression tasks)"""
    # Load dataset
    housing = sklearn_datasets.fetch_california_housing()
    
    # Prepare data
    X = housing.data
    y = housing.target
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)
    
    # Feature names
    feature_names = housing.feature_names
    
    return train_loader, test_loader, train_dataset, test_dataset, feature_names

def load_breast_cancer():
    """Load Breast Cancer Wisconsin dataset (for binary classification)"""
    # Load dataset
    cancer = sklearn_datasets.load_breast_cancer()
    
    # Prepare data
    X = cancer.data
    y = cancer.target
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Class names
    class_names = {0: 'malignant', 1: 'benign'}
    
    # Feature names
    feature_names = cancer.feature_names
    
    return train_loader, test_loader, train_dataset, test_dataset, class_names, feature_names

def visualize_image_dataset(dataset, class_names, num_samples=10):
    """Visualize examples from an image dataset"""
    # Set up the figure
    fig = plt.figure(figsize=(12, 4))
    
    # Select random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get the image and label
        img, label = dataset[idx]
        
        # Add subplot
        ax = fig.add_subplot(1, num_samples, i + 1)
        
        # Handle different image formats
        if img.shape[0] == 1:  # Grayscale
            img_np = img.numpy()[0]
            ax.imshow(img_np, cmap='gray')
        else:  # RGB
            img_np = img.numpy().transpose((1, 2, 0))  # Change from CxHxW to HxWxC
            # Unnormalize
            img_np = img_np * 0.5 + 0.5
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
        
        # Set title and remove ticks
        if class_names:
            ax.set_title(class_names[label])
        else:
            ax.set_title(f"Label: {label}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def visualize_regression_data(X, y, feature_idx1=0, feature_idx2=1, feature_names=None):
    """Visualize regression data (2D scatter plot of two features vs target)"""
    plt.figure(figsize=(12, 5))
    
    # Plot feature 1 vs target
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, feature_idx1], y, alpha=0.5)
    if feature_names:
        plt.xlabel(feature_names[feature_idx1])
    else:
        plt.xlabel(f"Feature {feature_idx1}")
    plt.ylabel("Target")
    plt.title("Feature vs Target")
    plt.grid(True, alpha=0.3)
    
    # Plot feature 2 vs target
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, feature_idx2], y, alpha=0.5)
    if feature_names:
        plt.xlabel(feature_names[feature_idx2])
    else:
        plt.xlabel(f"Feature {feature_idx2}")
    plt.ylabel("Target")
    plt.title("Feature vs Target")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_classification_data(X, y, feature_idx1=0, feature_idx2=1, class_names=None, feature_names=None):
    """Visualize classification data (2D scatter plot of two features colored by class)"""
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(X[:, feature_idx1], X[:, feature_idx2], c=y, cmap='viridis', 
                          alpha=0.8, edgecolors='w')
    
    # Set labels
    if feature_names:
        plt.xlabel(feature_names[feature_idx1])
        plt.ylabel(feature_names[feature_idx2])
    else:
        plt.xlabel(f"Feature {feature_idx1}")
        plt.ylabel(f"Feature {feature_idx2}")
    
    plt.title("Feature Space Visualization")
    plt.grid(True, alpha=0.3)
    
    # Add legend
    if class_names:
        legend_labels = [class_names[i] for i in range(len(class_names))]
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, 
                   title="Classes", loc="upper right")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Dataset Demo: Available Datasets for Neural Network Exercises")
    print("-" * 70)
    
    # 1. Fashion MNIST
    print("\n1. Fashion MNIST Dataset:")
    train_loader, test_loader, train_dataset, test_dataset, class_names = load_fashion_mnist()
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Testing samples: {len(test_dataset)}")
    print(f"  - Input shape: {train_dataset[0][0].shape}")
    print(f"  - Classes: {list(class_names.values())}")
    print("\nVisualizing Fashion MNIST examples:")
    visualize_image_dataset(train_dataset, class_names)
    
    # 2. CIFAR-10
    print("\n2. CIFAR-10 Dataset:")
    train_loader, test_loader, train_dataset, test_dataset, class_names = load_cifar10()
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Testing samples: {len(test_dataset)}")
    print(f"  - Input shape: {train_dataset[0][0].shape}")
    print(f"  - Classes: {list(class_names.values())}")
    print("\nVisualizing CIFAR-10 examples:")
    visualize_image_dataset(train_dataset, class_names)
    
    # 3. Digits Dataset (scikit-learn)
    print("\n3. Digits Dataset (scikit-learn):")
    train_loader, test_loader, train_dataset, test_dataset, class_names = load_sklearn_digits()
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Testing samples: {len(test_dataset)}")
    print(f"  - Input shape: {train_dataset[0][0].shape}")
    print(f"  - Classes: {list(class_names.values())}")
    
    # 4. California Housing Dataset
    print("\n4. California Housing Dataset:")
    train_loader, test_loader, train_dataset, test_dataset, feature_names = load_california_housing()
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Testing samples: {len(test_dataset)}")
    print(f"  - Input shape: {train_dataset[0][0].shape}")
    print(f"  - Features: {feature_names}")
    print("\nVisualizing California Housing data:")
    X_train = train_dataset.tensors[0].numpy()
    y_train = train_dataset.tensors[1].numpy().flatten()
    visualize_regression_data(X_train, y_train, feature_names=feature_names)
    
    # 5. Breast Cancer Wisconsin Dataset
    print("\n5. Breast Cancer Wisconsin Dataset:")
    train_loader, test_loader, train_dataset, test_dataset, class_names, feature_names = load_breast_cancer()
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Testing samples: {len(test_dataset)}")
    print(f"  - Input shape: {train_dataset[0][0].shape}")
    print(f"  - Classes: {list(class_names.values())}")
    print("\nVisualizing Breast Cancer data:")
    X_train = train_dataset.tensors[0].numpy()
    y_train = train_dataset.tensors[1].numpy()
    visualize_classification_data(X_train, y_train, class_names=class_names, feature_names=feature_names)
