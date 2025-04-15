"""
Training Functions for Neural Networks
Building Blocks of Generative AI Course - Day 1

This script provides functions for training and evaluating neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

def train_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Train a model for one epoch.
    
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
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
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

def evaluate(model, test_loader, criterion, device='cpu'):
    """
    Evaluate a model on a test set.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for computation (CPU or GPU)
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", leave=False):
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
    
    # Calculate statistics
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                scheduler=None, num_epochs=5, device='cpu', visualize=False):
    """
    Train a model for multiple epochs.
    
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
        'epoch_times': []
    }
    
    # Training loop
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on test set
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - start_time
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"Time: {epoch_time:.2f}s")
    
    # Visualize training progress if requested
    if visualize:
        plot_training_history(history)
    
    return history

def plot_training_history(history):
    """
    Plot the training history.
    
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
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def get_class_accuracy(model, test_loader, num_classes=10, device='cpu'):
    """
    Calculate accuracy for each class.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        num_classes: Number of classes
        device: Device to use for computation (CPU or GPU)
    
    Returns:
        tuple: (class_accuracies, confusion_matrix)
    """
    model.eval()
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Update confusion matrix
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # Calculate class accuracy
    class_accuracies = confusion_matrix.diag() / confusion_matrix.sum(1)
    
    return class_accuracies, confusion_matrix

def visualize_class_accuracy(class_accuracies, class_names=None):
    """
    Visualize accuracy for each class.
    
    Args:
        class_accuracies: Tensor of class accuracies
        class_names: List of class names (optional)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(class_accuracies):
        class_accuracies = class_accuracies.cpu().numpy()
    
    # Create class labels
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(class_accuracies))]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_accuracies * 100)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Class')
    plt.title('Accuracy by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_confusion_matrix(confusion_matrix, class_names=None):
    """
    Visualize the confusion matrix.
    
    Args:
        confusion_matrix: PyTorch tensor containing the confusion matrix
        class_names: List of class names (optional)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(confusion_matrix):
        confusion_matrix = confusion_matrix.cpu().numpy()
    
    # Create class labels
    if class_names is None:
        class_names = [f"Class {i}" for i in range(confusion_matrix.shape[0])]
    
    # Normalize the confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
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

def create_optimizer(model, optimizer_type='sgd', learning_rate=0.01, momentum=0.9, weight_decay=0.0):
    """
    Create an optimizer for the model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
        learning_rate: Learning rate
        momentum: Momentum (for SGD)
        weight_decay: Weight decay (L2 penalty)
    
    Returns:
        optimizer: PyTorch optimizer
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer

def create_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.1, 
                    num_epochs=None, last_epoch=-1):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')
        step_size: Period of learning rate decay (for StepLR)
        gamma: Multiplicative factor of learning rate decay
        num_epochs: Total number of epochs (for CosineAnnealingLR)
        last_epoch: The index of the last epoch
    
    Returns:
        scheduler: PyTorch learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, 
                                             gamma=gamma, last_epoch=last_epoch)
    elif scheduler_type == 'cosine':
        if num_epochs is None:
            raise ValueError("num_epochs must be specified for cosine scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, 
                                                        last_epoch=last_epoch)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=gamma, patience=5, 
                                                        threshold=0.001, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, verbose=True)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Define a simple model for testing
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super(SimpleTestModel, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create model, loss function, and optimizer
    model = SimpleTestModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, 'sgd', learning_rate=0.01)
    scheduler = create_scheduler(optimizer, 'step', step_size=1, gamma=0.95)
    
    # Train the model for a few epochs
    print("Testing training functionality with a small subset of data...")
    
    # Use a small subset for quick testing
    subset_size = 1000
    indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = torch.utils.data.Subset(train_dataset, indices)
    train_subset_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    # Train for a few epochs
    history = train_model(
        model=model,
        train_loader=train_subset_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=3,
        device=device,
        visualize=True
    )
    
    # Get class accuracy
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    class_accuracies, confusion_matrix = get_class_accuracy(model, test_loader, device=device)
    
    # Visualize class accuracy
    visualize_class_accuracy(class_accuracies, class_names)
    
    # Visualize confusion matrix
    visualize_confusion_matrix(confusion_matrix, class_names)
