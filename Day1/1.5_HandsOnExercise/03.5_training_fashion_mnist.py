"""
Training Functions for Fashion MNIST
Building Blocks of Generative AI Course - Day 1

This script provides specialized training and evaluation functions
specifically optimized for the Fashion MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Fashion MNIST class names for reference
FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_fashion_mnist(batch_size=64, data_augmentation=False):
    """
    Load the Fashion MNIST dataset with optional data augmentation.
    
    Args:
        batch_size: Batch size for the data loaders
        data_augmentation: Whether to use data augmentation for training
        
    Returns:
        tuple: (train_loader, test_loader, class_names)
    """
    # Define base transformations
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Define augmentation transformations if requested
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = base_transform
    
    # Load datasets
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=base_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, FASHION_MNIST_CLASSES

def train_fashion_mnist_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Train a model for one epoch on Fashion MNIST data.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        device: Device to use for computation (CPU or GPU)
    
    Returns:
        tuple: (average_loss, accuracy, class_accuracies)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize per-class tracking
    class_correct = [0] * 10
    class_total = [0] * 10
    
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
        
        # Track global statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Track per-class statistics
        for i in range(targets.size(0)):
            label = targets[i].item()
            prediction = predicted[i].item()
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1
    
    # Calculate epoch statistics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Calculate class accuracies
    class_accuracies = []
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies.append(100. * class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0)
    
    return avg_loss, accuracy, class_accuracies

def evaluate_fashion_mnist(model, test_loader, criterion, device='cpu'):
    """
    Evaluate a model on the Fashion MNIST test set.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for computation (CPU or GPU)
    
    Returns:
        tuple: (average_loss, accuracy, class_accuracies, confusion_matrix)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize per-class tracking
    class_correct = [0] * 10
    class_total = [0] * 10
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(10, 10)
    
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
            
            # Track per-class statistics
            for i in range(targets.size(0)):
                label = targets[i].item()
                prediction = predicted[i].item()
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1
                
                # Update confusion matrix
                confusion_matrix[label][prediction] += 1
    
    # Calculate statistics
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    # Calculate class accuracies
    class_accuracies = []
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies.append(100. * class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0)
    
    return avg_loss, accuracy, class_accuracies, confusion_matrix

def train_fashion_mnist_model(model, train_loader, test_loader, num_epochs=10, 
                             learning_rate=0.001, optimizer_type='adam', 
                             weight_decay=0.0, scheduler_type=None, 
                             device='cpu', visualize=True):
    """
    Train a model on the Fashion MNIST dataset with comprehensive analysis.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
        weight_decay: Weight decay (L2 penalty)
        scheduler_type: Learning rate scheduler type (None, 'step', 'cosine', 'plateau')
        device: Device to use for computation (CPU or GPU)
        visualize: Whether to show visualizations during training
        
    Returns:
        tuple: (model, training_history)
    """
    # Move model to device
    model = model.to(device)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Set up scheduler
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    else:
        scheduler = None
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'class_accuracies': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    # Initial evaluation
    print("Initial model evaluation...")
    val_loss, val_acc, class_accs, conf_matrix = evaluate_fashion_mnist(model, test_loader, criterion, device)
    print(f"Initial Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    # Training loop
    print(f"Training for {num_epochs} epochs...")
    best_acc = val_acc
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs} - Learning Rate: {current_lr:.6f}")
        
        # Train for one epoch
        train_loss, train_acc, _ = train_fashion_mnist_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on test set
        val_loss, val_acc, class_accs, conf_matrix = evaluate_fashion_mnist(model, test_loader, criterion, device)
        
        # Update learning rate if using plateau scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - start_time
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['class_accuracies'].append(class_accs)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"Time: {epoch_time:.2f}s")
        
        # Check if this is the best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best model with validation accuracy: {val_acc:.2f}%")
        
        # Visualize progress if requested and at appropriate intervals
        if visualize and (epoch == 1 or epoch % 5 == 0 or epoch == num_epochs):
            plot_fashion_mnist_progress(history, epoch)
    
    # Final evaluation and visualization
    if visualize:
        plot_fashion_mnist_results(history, FASHION_MNIST_CLASSES)
        plot_fashion_mnist_confusion_matrix(conf_matrix, FASHION_MNIST_CLASSES)
    
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
    return model, history

def plot_fashion_mnist_progress(history, current_epoch):
    """
    Plot the training progress for Fashion MNIST.
    
    Args:
        history: Dictionary containing training history
        current_epoch: Current training epoch
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Epoch {current_epoch})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['val_acc'], 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training and Validation Accuracy (Epoch {current_epoch})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_fashion_mnist_results(history, class_names):
    """
    Plot comprehensive results for Fashion MNIST training.
    
    Args:
        history: Dictionary containing training history
        class_names: List of class names
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot loss
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot learning rate
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(history['learning_rates'], 'g-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(alpha=0.3)
    
    # Plot per-class accuracy (from the last epoch)
    ax4 = fig.add_subplot(2, 2, 4)
    final_class_accs = history['class_accuracies'][-1]
    x = np.arange(len(class_names))
    ax4.bar(x, final_class_accs)
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Final Per-Class Accuracy')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_fashion_mnist_confusion_matrix(confusion_matrix, class_names):
    """
    Plot a confusion matrix for Fashion MNIST results.
    
    Args:
        confusion_matrix: Tensor containing the confusion matrix
        class_names: List of class names
    """
    # Convert to numpy if it's a tensor
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()
    
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_cm = confusion_matrix / row_sums
    
    plt.figure(figsize=(12, 10))
    plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f'
    thresh = normalized_cm.max() / 2.
    for i in range(normalized_cm.shape[0]):
        for j in range(normalized_cm.shape[1]):
            plt.text(j, i, format(normalized_cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if normalized_cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def analyze_fashion_mnist_errors(model, test_loader, class_names, device='cpu', num_examples=16):
    """
    Analyze and visualize misclassified Fashion MNIST examples.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        class_names: List of class names
        device: Device to use for computation (CPU or GPU)
        num_examples: Number of misclassified examples to show
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples
            mask = preds != labels
            misclassified_idx = torch.where(mask)[0]
            
            for idx in misclassified_idx:
                if len(misclassified_images) < num_examples:
                    misclassified_images.append(images[idx].cpu())
                    misclassified_labels.append(labels[idx].item())
                    misclassified_preds.append(preds[idx].item())
                else:
                    break
            
            if len(misclassified_images) >= num_examples:
                break
    
    # Visualize misclassified examples
    fig = plt.figure(figsize=(15, 10))
    for i in range(len(misclassified_images)):
        plt.subplot(4, 4, i+1)
        plt.imshow(misclassified_images[i].squeeze().numpy(), cmap='gray')
        plt.title(f'True: {class_names[misclassified_labels[i]]}\nPred: {class_names[misclassified_preds[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Misclassified Fashion MNIST Examples", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

def analyze_fashion_mnist_similar_classes(confusion_matrix, class_names):
    """
    Analyze which Fashion MNIST classes are frequently confused with each other.
    
    Args:
        confusion_matrix: Tensor containing the confusion matrix
        class_names: List of class names
    
    Returns:
        list: Tuples of (class_a, class_b, confusion_score) for most confused pairs
    """
    # Convert to numpy if it's a tensor
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()
    
    # Normalize by row (true class)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    norm_confusion = confusion_matrix / row_sums
    
    # Find the most confused pairs (excluding diagonal)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:  # Skip diagonal
                confused_pairs.append((i, j, norm_confusion[i, j]))
    
    # Sort by confusion score
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Print the top confused pairs
    print("Top confused class pairs:")
    for true_class, pred_class, score in confused_pairs[:5]:
        print(f"True: {class_names[true_class]}, Predicted: {class_names[pred_class]}, Score: {score:.4f}")
    
    # Create a bar chart of the top confused pairs
    plt.figure(figsize=(12, 6))
    top_pairs = confused_pairs[:10]
    pair_names = [f"{class_names[true][:3]}/{class_names[pred][:3]}" for true, pred, _ in top_pairs]
    scores = [score for _, _, score in top_pairs]
    
    plt.bar(range(len(top_pairs)), scores)
    plt.xlabel('Class Pairs')
    plt.ylabel('Confusion Score')
    plt.title('Top Confused Class Pairs in Fashion MNIST')
    plt.xticks(range(len(top_pairs)), pair_names, rotation=45)
    plt.tight_layout()
    plt.show()
    
    return confused_pairs

if __name__ == "__main__":
    print("Fashion MNIST Training Module - Building Blocks of Generative AI Course")
    
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Fashion MNIST dataset with data augmentation
    train_loader, test_loader, class_names = load_fashion_mnist(batch_size=64, data_augmentation=True)
    print(f"Dataset loaded with classes: {class_names}")
    
    # Define a simple model for testing
    class FashionMNISTModel(nn.Module):
        def __init__(self):
            super(FashionMNISTModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    # Create and train the model
    model = FashionMNISTModel()
    
    # Example quick training for demonstration
    print("\nTraining a small sample for demonstration:")
    _, sample_history = train_fashion_mnist_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=2,  # Low number for demonstration
        learning_rate=0.001,
        optimizer_type='adam',
        device=device,
        visualize=True
    )
    
    print("\nAnalyzing errors and confused classes:")
    # Get the latest confusion matrix
    _, _, _, confusion_matrix = evaluate_fashion_mnist(model, test_loader, nn.CrossEntropyLoss(), device)
    
    # Analyze which classes are most often confused
    analyze_fashion_mnist_similar_classes(confusion_matrix, class_names)
    
    # Show examples of misclassifications
    analyze_fashion_mnist_errors(model, test_loader, class_names, device)
