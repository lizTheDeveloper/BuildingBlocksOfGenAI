"""
Fashion MNIST Training Functions
Building Blocks of Generative AI Course - Day 1

This script provides specialized functions for training and evaluating neural networks
on the Fashion MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

# Fashion MNIST class names
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def load_fashion_mnist(batch_size=64, num_workers=2):
    """
    Load the Fashion MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, test_loader, class_names)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load Fashion MNIST datasets
    train_dataset = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, FASHION_MNIST_CLASSES

def train_fashion_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Train a model for one epoch on Fashion MNIST data.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        device: Device to use for computation (CPU or GPU)
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for progress bar
    for inputs, targets in tqdm(train_loader, desc="Training on Fashion MNIST", leave=False):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Calculate epoch statistics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def evaluate_fashion(model, test_loader, criterion, device='cpu'):
    """
    Evaluate a model on the Fashion MNIST test set.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for computation (CPU or GPU)
    
    Returns:
        tuple: (average_loss, accuracy, per_class_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = torch.zeros(10, device=device)
    class_total = torch.zeros(10, device=device)
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating on Fashion MNIST", leave=False):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update per-class accuracy
            for c in range(10):
                class_mask = (targets == c)
                class_correct[c] += (predicted[class_mask] == c).sum().item()
                class_total[c] += class_mask.sum().item()
    
    # Calculate statistics
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    # Calculate per-class accuracy
    per_class_accuracy = 100. * class_correct / class_total
    
    return avg_loss, accuracy, per_class_accuracy

def train_fashion_model(model, train_loader, test_loader, criterion, optimizer, 
                       scheduler=None, num_epochs=5, device='cpu', visualize=False):
    """
    Train a model for multiple epochs on Fashion MNIST.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train
        device: Device to use for computation (CPU or GPU)
        visualize: Whether to visualize training progress
    
    Returns:
        dict: Training history including losses and accuracies
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'per_class_acc': [],
        'epoch_times': []
    }
    
    # Training loop
    print(f"Training on Fashion MNIST for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_fashion_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on test set
        val_loss, val_acc, per_class_acc = evaluate_fashion(model, test_loader, criterion, device)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - start_time
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['per_class_acc'].append(per_class_acc.cpu().numpy())
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"Time: {epoch_time:.2f}s")
    
    # Visualize training progress if requested
    if visualize:
        plot_fashion_training_history(history)
        plot_fashion_class_accuracy(history['per_class_acc'][-1], FASHION_MNIST_CLASSES)
    
    return history

def plot_fashion_training_history(history):
    """
    Plot the training history for Fashion MNIST.
    
    Args:
        history: Dictionary containing training history
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Fashion MNIST Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Fashion MNIST Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_fashion_class_accuracy(class_accuracies, class_names=FASHION_MNIST_CLASSES):
    """
    Visualize accuracy for each Fashion MNIST class.
    
    Args:
        class_accuracies: Array of class accuracies
        class_names: List of class names
    """
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_accuracies)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Class')
    plt.title('Fashion MNIST Accuracy by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_fashion_confusion_matrix(model, test_loader, device='cpu'):
    """
    Calculate confusion matrix for Fashion MNIST.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to use for computation (CPU or GPU)
    
    Returns:
        confusion_matrix: PyTorch tensor containing the confusion matrix
    """
    model.eval()
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(10, 10, device=device)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Update confusion matrix
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix

def visualize_fashion_confusion_matrix(confusion_matrix, class_names=FASHION_MNIST_CLASSES):
    """
    Visualize the confusion matrix for Fashion MNIST.
    
    Args:
        confusion_matrix: PyTorch tensor containing the confusion matrix
        class_names: List of class names
    """
    # Convert to numpy if tensor
    if torch.is_tensor(confusion_matrix):
        confusion_matrix = confusion_matrix.cpu().numpy()
    
    # Normalize the confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Fashion MNIST Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

def analyze_fashion_misclassifications(model, test_loader, device='cpu', num_samples=25):
    """
    Analyze misclassifications on Fashion MNIST.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to use for computation (CPU or GPU)
        num_samples: Number of misclassified samples to show
    """
    model.eval()
    
    # Collect misclassified samples
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples
            incorrect_mask = (preds != targets)
            if incorrect_mask.sum().item() > 0:
                misclassified_images.append(inputs[incorrect_mask])
                misclassified_labels.append(targets[incorrect_mask])
                misclassified_preds.append(preds[incorrect_mask])
            
            # Check if we have enough samples
            if len(misclassified_images) > 0 and sum(len(batch) for batch in misclassified_images) >= num_samples:
                break
    
    # Concatenate batches
    if misclassified_images:
        misclassified_images = torch.cat(misclassified_images)
        misclassified_labels = torch.cat(misclassified_labels)
        misclassified_preds = torch.cat(misclassified_preds)
        
        # Select a subset
        indices = torch.randperm(len(misclassified_images))[:num_samples]
        misclassified_images = misclassified_images[indices]
        misclassified_labels = misclassified_labels[indices]
        misclassified_preds = misclassified_preds[indices]
        
        # Plot the misclassified examples
        n_cols = 5
        n_rows = (num_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        
        for i, ax in enumerate(axes.flat):
            if i < len(misclassified_images):
                img = misclassified_images[i][0].cpu().numpy()
                true_label = misclassified_labels[i].item()
                pred_label = misclassified_preds[i].item()
                
                ax.imshow(img, cmap='gray')
                ax.set_title(f'True: {FASHION_MNIST_CLASSES[true_label]}\nPred: {FASHION_MNIST_CLASSES[pred_label]}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No misclassifications found.")

def visualize_fashion_item(loader, class_names=FASHION_MNIST_CLASSES, num_samples=10):
    """
    Visualize random Fashion MNIST items with their labels.
    
    Args:
        loader: DataLoader for Fashion MNIST data
        class_names: List of class names
        num_samples: Number of samples to show
    """
    # Get a batch of data
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Show images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images[i][0], cmap='gray')
            ax.set_title(class_names[labels[i]])
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Fashion MNIST dataset
    train_loader, test_loader, class_names = load_fashion_mnist(batch_size=64)
    
    # Visualize some examples
    visualize_fashion_item(train_loader, class_names)
    
    # Define a simple model for testing
    class FashionMNISTModel(nn.Module):
        def __init__(self):
            super(FashionMNISTModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create model, loss function, and optimizer
    model = FashionMNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    print("Testing training functionality with a subset of Fashion MNIST...")
    
    # Use a small subset for quick testing
    subset_size = 1000
    indices = torch.randperm(len(train_loader.dataset))[:subset_size]
    train_subset = torch.utils.data.Subset(train_loader.dataset, indices)
    train_subset_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    # Train for a few epochs
    history = train_fashion_model(
        model=model,
        train_loader=train_subset_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3,
        device=device,
        visualize=True
    )
    
    # Analyze misclassifications
    analyze_fashion_misclassifications(model, test_loader, device=device)
    
    # Get and visualize confusion matrix
    confusion_matrix = get_fashion_confusion_matrix(model, test_loader, device=device)
    visualize_fashion_confusion_matrix(confusion_matrix, class_names)
