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

def describe_alternative_datasets():
    """
    Print information about alternative datasets that can be used for similar exercises.
    """
    print("\n===== Alternative Datasets for Neural Network Exercises =====\n")
    
    datasets_info = [
        {
            "name": "Fashion-MNIST",
            "description": "70,000 grayscale images (28x28 pixels) of clothing items across 10 categories.",
            "ideal_for": "Convolutional Neural Networks (CNNs), classification tasks",
            "why_great": "More challenging and modern alternative to MNIST."
        },
        {
            "name": "Digits Dataset (Scikit-learn)",
            "description": "1,797 images of handwritten digits (8x8 pixels) with labels from 0 to 9.",
            "ideal_for": "Introductory machine learning models, quick prototyping",
            "why_great": "Small size allows for quick experimentation."
        },
        {
            "name": "California Housing Dataset",
            "description": "Housing prices in California with features like median income, house age, etc.",
            "ideal_for": "Linear regression, decision trees, and other regression algorithms",
            "why_great": "Perfect for regression tasks and predicting continuous values."
        },
        {
            "name": "IMDb Movie Reviews",
            "description": "50,000 movie reviews labeled as positive or negative.",
            "ideal_for": "Text classification, sentiment analysis, NLP models",
            "why_great": "Standard benchmark for sentiment analysis."
        },
        {
            "name": "Sentiment140",
            "description": "1.6 million tweets labeled for sentiment (positive, negative, neutral).",
            "ideal_for": "NLP, sentiment analysis, handling noisy text data",
            "why_great": "Real-world data for practicing sentiment analysis on social media text."
        },
        {
            "name": "Breast Cancer Wisconsin Diagnostic Dataset",
            "description": "569 instances with 30 attributes related to cell nuclei, labeled as malignant or benign.",
            "ideal_for": "Classification algorithms, medical diagnosis models",
            "why_great": "Classic dataset for binary classification tasks in the medical domain."
        },
        {
            "name": "Mall Customers Dataset",
            "description": "Data on customers' annual income, spending score, and age.",
            "ideal_for": "K-means clustering, customer segmentation analysis",
            "why_great": "Useful for clustering and customer segmentation exercises."
        },
        {
            "name": "Oxford-IIIT Pet Dataset",
            "description": "7,349 images of 37 pet breeds, with annotations for breed, head pose, and bounding boxes.",
            "ideal_for": "CNNs, transfer learning, object detection tasks",
            "why_great": "Excellent for practicing image classification and object detection."
        },
        {
            "name": "CIFAR-10",
            "description": "60,000 32x32 color images across 10 classes, including airplanes, cars, and birds.",
            "ideal_for": "CNNs, image classification, deep learning models",
            "why_great": "A step up from MNIST, offering more complex images for classification tasks."
        },
        {
            "name": "UrbanSound8K",
            "description": "8,732 labeled sound excerpts from 10 urban sound classes (sirens, dog barks, etc.).",
            "ideal_for": "Audio classification, feature extraction, spectrogram analysis",
            "why_great": "Introduces learners to audio data processing and classification."
        }
    ]
    
    for i, dataset in enumerate(datasets_info, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Ideal for: {dataset['ideal_for']}")
        print(f"   Why it's great: {dataset['why_great']}")
        print()
    
    print("Each dataset offers unique learning opportunities and can be used for similar neural network exercises.")
    print("The choice of dataset depends on the specific concepts you want to explore and the type of data you're interested in working with.")

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
    
    # Print info about alternative datasets
    describe_alternative_datasets()
