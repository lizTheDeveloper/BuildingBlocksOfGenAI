"""
Fashion MNIST Visualization Tools
Building Blocks of Generative AI Course - Day 1

This script provides specialized visualization functions for the Fashion MNIST dataset,
helping students understand neural network behavior and model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision.utils as vutils
import seaborn as sns
from itertools import product

# Fashion MNIST class names for reference
FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def visualize_fashion_mnist_samples(dataloader, num_images=25, class_names=None):
    """
    Visualize a grid of random Fashion MNIST images from a dataloader.
    
    Args:
        dataloader: DataLoader containing Fashion MNIST images
        num_images: Number of images to display
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = FASHION_MNIST_CLASSES
    
    # Get a batch of images
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Limit to num_images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Create a grid of images
    grid_size = int(np.ceil(np.sqrt(num_images)))
    plt.figure(figsize=(12, 12))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        if i >= num_images:
            break
            
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.title(class_names[label])
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Fashion MNIST Samples", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

def visualize_fashion_mnist_class_distribution(dataloader, class_names=None):
    """
    Visualize the distribution of classes in a Fashion MNIST dataset.
    
    Args:
        dataloader: DataLoader containing Fashion MNIST data
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = FASHION_MNIST_CLASSES
    
    # Count instances of each class
    class_counts = [0] * len(class_names)
    
    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), class_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Fashion MNIST Class Distribution')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print class statistics
    total = sum(class_counts)
    print("Class distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"{name}: {count} samples ({100 * count / total:.2f}%)")

def visualize_fashion_mnist_feature_maps(model, image, layer_names=None, device='cpu'):
    """
    Visualize feature maps (activations) of a CNN model for a Fashion MNIST image.
    
    Args:
        model: PyTorch CNN model
        image: Input image tensor (1, 1, 28, 28)
        layer_names: Names of layers to visualize (optional)
        device: Device to use for computation (CPU or GPU)
    """
    # Move image to device
    image = image.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Dictionary to store activations
    activations = {}
    
    # Hook function to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    if layer_names is None:
        # Auto-detect convolutional layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(get_activation(name)))
    else:
        # Use provided layer names
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Forward pass to get activations
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Display the original image
    plt.figure(figsize=(5, 5))
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    # Display feature maps for each layer
    for name, activation in activations.items():
        if len(activation.shape) != 4:  # Skip non-convolutional layers
            continue
            
        # Get the number of feature maps
        num_features = min(activation.shape[1], 16)  # Show up to 16 feature maps
        
        # Create a grid of feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(16):
            if i < num_features:
                # Get feature map
                feature_map = activation[0, i].numpy()
                
                # Display feature map
                im = axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'Filter {i+1}')
                axes[i].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            else:
                axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps: {name}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

def visualize_fashion_mnist_filters(model, layer_idx=0, figsize=(12, 12)):
    """
    Visualize the filters (weights) of a CNN model for Fashion MNIST.
    
    Args:
        model: PyTorch CNN model
        layer_idx: Index of the convolutional layer to visualize
        figsize: Figure size
    """
    # Get all convolutional layers
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    
    if not conv_layers:
        print("No convolutional layers found in the model.")
        return
    
    if layer_idx >= len(conv_layers):
        print(f"Layer index {layer_idx} out of range. Model has {len(conv_layers)} convolutional layers.")
        return
    
    # Get the weights of the specified layer
    weights = conv_layers[layer_idx].weight.data.cpu()
    num_filters = weights.shape[0]
    num_channels = weights.shape[1]
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    # Create a figure
    plt.figure(figsize=figsize)
    
    # Plot each filter
    for i in range(num_filters):
        plt.subplot(grid_size, grid_size, i+1)
        
        if num_channels == 1:
            # For single-channel input (grayscale)
            plt.imshow(weights[i, 0], cmap='viridis')
        else:
            # For multi-channel input, take mean across channels
            plt.imshow(weights[i].mean(dim=0), cmap='viridis')
            
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    
    plt.suptitle(f'Filters of Convolutional Layer {layer_idx+1}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def visualize_fashion_mnist_tsne(model, dataloader, num_samples=1000, perplexity=30, device='cpu'):
    """
    Visualize Fashion MNIST embeddings using t-SNE for dimensionality reduction.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing Fashion MNIST data
        num_samples: Number of samples to visualize
        perplexity: Perplexity parameter for t-SNE
        device: Device to use for computation (CPU or GPU)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Extract features from the model
    features = []
    labels = []
    sample_count = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            # Check if we've collected enough samples
            if sample_count >= num_samples:
                break
                
            # Limit batch size if needed
            batch_size = min(images.shape[0], num_samples - sample_count)
            images = images[:batch_size].to(device)
            targets = targets[:batch_size]
            
            # Forward pass to get features
            # Here we use the penultimate layer (before the final classification layer)
            # This requires a custom forward hook or modifying the model
            # For simplicity, let's use the output logits
            outputs = model(images)
            
            # Store features and labels
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
            
            sample_count += batch_size
    
    # Concatenate all features and labels
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Performing t-SNE on {len(features)} samples with perplexity {perplexity}...")
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Create a scatter plot with different colors for each class
    plt.figure(figsize=(12, 10))
    
    # Define colors for the scatter plot
    colors = plt.cm.rainbow(np.linspace(0, 1, len(FASHION_MNIST_CLASSES)))
    
    for i, label in enumerate(FASHION_MNIST_CLASSES):
        idx = (labels == i)
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], c=[colors[i]], label=label, alpha=0.7)
    
    plt.legend()
    plt.title(f't-SNE Visualization of Fashion MNIST (Perplexity: {perplexity})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_fashion_mnist_predictions(model, images, labels, predictions=None, class_names=None):
    """
    Visualize model predictions on Fashion MNIST images.
    
    Args:
        model: PyTorch model
        images: Batch of images (B, 1, 28, 28)
        labels: Ground truth labels
        predictions: Model predictions (optional, will be computed if not provided)
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = FASHION_MNIST_CLASSES
    
    # Set model to evaluation mode
    model.eval()
    
    # Compute predictions if not provided
    if predictions is None:
        with torch.no_grad():
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
    
    # Display images with predictions
    num_images = min(len(images), 25)  # Show up to 25 images
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    plt.figure(figsize=(15, 15))
    
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        
        # Green for correct predictions, red for incorrect
        color = 'green' if predictions[i] == labels[i] else 'red'
        
        plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}', 
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Fashion MNIST Predictions", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

def visualize_fashion_mnist_attention(model, images, class_names=None, device='cpu'):
    """
    Visualize where the model is 'looking' (paying attention) in the images using
    gradient-based class activation mapping.
    
    Args:
        model: PyTorch CNN model
        images: Batch of images to visualize
        class_names: List of class names (optional)
        device: Device to use for computation (CPU or GPU)
    """
    if class_names is None:
        class_names = FASHION_MNIST_CLASSES
    
    # Set model to evaluation mode
    model.eval()
    
    # Move images to device
    images = images.to(device)
    
    # Number of images to visualize
    num_images = min(len(images), 9)
    
    # Create a figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        # Get a single image
        img = images[i:i+1].clone().requires_grad_(True)
        
        # Forward pass
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        
        # Target is the predicted class
        target_class = predicted.item()
        
        # Zero gradients
        model.zero_grad()
        
        # Backward pass for the target class
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        outputs.backward(gradient=one_hot)
        
        # Get gradients of the input image
        gradients = img.grad.data.abs()
        
        # Average gradients across channels
        gradients = gradients.mean(dim=1).squeeze().cpu().numpy()
        
        # Normalize gradients for visualization
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        # Display original image
        axes[i].imshow(img.squeeze().detach().cpu().numpy(), cmap='gray', alpha=0.7)
        
        # Overlay attention map
        im = axes[i].imshow(gradients, cmap='jet', alpha=0.5)
        axes[i].set_title(f'Predicted: {class_names[target_class]}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Attention Maps for Fashion MNIST", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def visualize_fashion_mnist_decision_regions(model, feature_dim=2, resolution=100, device='cpu'):
    """
    Visualize the decision regions of a model for Fashion MNIST in 2D space.
    This requires dimensionality reduction as Fashion MNIST is high-dimensional.
    
    Args:
        model: PyTorch model
        feature_dim: Dimensionality reduction target (must be 2)
        resolution: Resolution of the decision boundary grid
        device: Device to use for computation (CPU or GPU)
    """
    # Currently, this function only supports 2D visualization
    if feature_dim != 2:
        print("This function only supports 2D visualization (feature_dim=2).")
        return
    
    # Load a small subset of Fashion MNIST
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load test dataset
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    # Use a subset for dimensionality reduction
    subset_size = 1000
    indices = torch.randperm(len(test_dataset))[:subset_size]
    test_subset = Subset(test_dataset, indices)
    data_loader = DataLoader(test_subset, batch_size=subset_size, shuffle=False)
    
    # Extract features and labels
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            break
    
    # Apply PCA for dimensionality reduction
    print("Applying PCA for dimensionality reduction...")
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Create a mesh grid for decision boundary visualization
    x_min, x_max = features_pca[:, 0].min() - 1, features_pca[:, 0].max() + 1
    y_min, y_max = features_pca[:, 1].min() - 1, features_pca[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                          np.linspace(y_min, y_max, resolution))
    
    # Create a simple model for the reduced features
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleClassifier, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    # Train a classifier on the reduced features
    simple_model = SimpleClassifier(2, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
    
    # Convert features and labels to PyTorch tensors
    train_features = torch.tensor(features_pca, dtype=torch.float32).to(device)
    train_labels = torch.tensor(labels, dtype=torch.long).to(device)
    
    # Train the simple model
    print("Training a simple classifier on the reduced features...")
    simple_model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = simple_model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == train_labels).float().mean().item()
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")
    
    # Evaluate on the mesh grid
    simple_model.eval()
    with torch.no_grad():
        # Reshape the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_tensor = torch.tensor(mesh_points, dtype=torch.float32).to(device)
        
        # Predict class for each point
        z = simple_model(mesh_points_tensor)
        _, z = torch.max(z, 1)
        z = z.cpu().numpy()
    
    # Reshape z to match the mesh grid
    z = z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(12, 10))
    
    # Plot decision regions
    cmap = plt.cm.rainbow
    plt.contourf(xx, yy, z, alpha=0.7, cmap=cmap)
    
    # Plot data points
    for i in range(10):
        idx = (labels == i)
        plt.scatter(features_pca[idx, 0], features_pca[idx, 1], 
                   c=cmap(i/10), label=FASHION_MNIST_CLASSES[i], 
                   edgecolors='black', alpha=0.7)
    
    plt.title('Decision Regions for Fashion MNIST (PCA-reduced)')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_fashion_mnist_similarity_matrix(model, dataloader, num_samples=100, device='cpu'):
    """
    Visualize a similarity matrix between Fashion MNIST images based on feature space distances.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing Fashion MNIST data
        num_samples: Number of samples to include in the matrix
        device: Device to use for computation (CPU or GPU)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Extract features and labels
    features = []
    labels = []
    sample_count = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            # Check if we've collected enough samples
            if sample_count >= num_samples:
                break
                
            # Limit batch size if needed
            batch_size = min(images.shape[0], num_samples - sample_count)
            images = images[:batch_size].to(device)
            targets = targets[:batch_size]
            
            # Forward pass to get features
            outputs = model(images)
            
            # Store features and labels
            features.append(outputs.cpu())
            labels.append(targets)
            
            sample_count += batch_size
    
    # Concatenate all features and labels
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    # Compute pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(features)
    
    # Create a sorted similarity matrix by class
    # Get sorting indices by class
    sort_idx = np.argsort(labels)
    sorted_similarity = similarity_matrix[sort_idx][:, sort_idx]
    sorted_labels = labels[sort_idx]
    
    # Create class boundary lines
    class_boundaries = [0]
    for i in range(10):
        class_count = np.sum(sorted_labels == i)
        class_boundaries.append(class_boundaries[-1] + class_count)
    
    # Visualize the similarity matrix
    plt.figure(figsize=(12, 10))
    im = plt.imshow(sorted_similarity, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Cosine Similarity')
    
    # Add class boundary lines
    for boundary in class_boundaries[1:-1]:
        plt.axhline(y=boundary-0.5, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=boundary-0.5, color='r', linestyle='-', alpha=0.3)
    
    # Add class labels
    for i in range(10):
        mid_point = (class_boundaries[i] + class_boundaries[i+1]) / 2
        plt.text(-5, mid_point, FASHION_MNIST_CLASSES[i], 
                 ha='right', va='center', fontsize=8)
        plt.text(mid_point, -5, FASHION_MNIST_CLASSES[i], 
                 ha='center', va='top', fontsize=8, rotation=90)
    
    plt.title('Feature Similarity Matrix for Fashion MNIST Classes')
    plt.tight_layout()
    plt.show()

def visualize_fashion_mnist_augmentations(image, label, class_names=None):
    """
    Visualize different data augmentations on a single Fashion MNIST image.
    
    Args:
        image: Single image tensor (1, 28, 28)
        label: Label of the image
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = FASHION_MNIST_CLASSES
    
    # Define augmentations
    augmentations = [
        ("Original", transforms.Compose([])),
        ("Horizontal Flip", transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])),
        ("Rotation (15°)", transforms.Compose([transforms.RandomRotation(15)])),
        ("Rotation (30°)", transforms.Compose([transforms.RandomRotation(30)])),
        ("Brightness", transforms.Compose([transforms.ColorJitter(brightness=0.5)])),
        ("Contrast", transforms.Compose([transforms.ColorJitter(contrast=0.5)])),
        ("Shear (20°)", transforms.Compose([transforms.RandomAffine(0, shear=20)])),
        ("Translation", transforms.Compose([transforms.RandomAffine(0, translate=(0.2, 0.2))])),
        ("Scale", transforms.Compose([transforms.RandomAffine(0, scale=(0.8, 1.2))])),
    ]
    
    # Apply augmentations
    augmented_images = []
    plt.figure(figsize=(15, 10))
    
    for i, (aug_name, aug_transform) in enumerate(augmentations):
        # Convert tensor to PIL image, apply augmentation, convert back to tensor
        pil_image = transforms.ToPILImage()(image.squeeze())
        aug_image = aug_transform(pil_image)
        aug_tensor = transforms.ToTensor()(aug_image)
        
        # Display augmented image
        plt.subplot(3, 3, i+1)
        plt.imshow(aug_tensor.squeeze().numpy(), cmap='gray')
        plt.title(aug_name)
        plt.axis('off')
    
    plt.suptitle(f'Data Augmentations for {class_names[label]}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    print("Fashion MNIST Visualization Tools - Building Blocks of Generative AI Course")
    
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Fashion MNIST dataset
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Define a simple CNN model for testing
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
    
    # Create a model instance
    model = FashionMNISTModel().to(device)
    
    # Get sample batch for testing
    sample_batch, sample_labels = next(iter(test_loader))
    
    # Example visualizations
    print("\nVisualizing sample Fashion MNIST images:")
    visualize_fashion_mnist_samples(test_loader, num_images=16)
    
    print("\nVisualizing class distribution:")
    visualize_fashion_mnist_class_distribution(train_loader)
    
    print("\nVisualizing data augmentations on a sample image:")
    visualize_fashion_mnist_augmentations(sample_batch[0], sample_labels[0])
    
    # Note: The following functions require a trained model and are commented out
    """
    print("\nVisualizing feature maps:")
    visualize_fashion_mnist_feature_maps(model, sample_batch[0], device=device)
    
    print("\nVisualizing model filters:")
    visualize_fashion_mnist_filters(model)
    
    print("\nVisualizing t-SNE embeddings:")
    visualize_fashion_mnist_tsne(model, test_loader, num_samples=500, device=device)
    
    print("\nVisualizing model predictions:")
    with torch.no_grad():
        outputs = model(sample_batch.to(device))
        _, preds = torch.max(outputs, 1)
    visualize_fashion_mnist_predictions(model, sample_batch, sample_labels, preds)
    
    print("\nVisualizing attention maps:")
    visualize_fashion_mnist_attention(model, sample_batch[:9], device=device)
    
    print("\nVisualizing decision regions:")
    visualize_fashion_mnist_decision_regions(model, device=device)
    
    print("\nVisualizing similarity matrix:")
    visualize_fashion_mnist_similarity_matrix(model, test_loader, num_samples=200, device=device)
    """
