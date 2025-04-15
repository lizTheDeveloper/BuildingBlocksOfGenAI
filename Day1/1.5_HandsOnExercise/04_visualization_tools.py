"""
Visualization Tools for Neural Networks
Building Blocks of Generative AI Course - Day 1

This script provides visualization functions to help understand neural networks and their training.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_decision_boundary(model, X, y, h=0.02, cmap_light=None, cmap_bold=None):
    """
    Visualize the decision boundary of a model for 2D data.
    
    Args:
        model: Trained PyTorch model
        X: Feature data (numpy array)
        y: Target labels (numpy array)
        h: Step size for the meshgrid
        cmap_light: Colormap for the decision regions
        cmap_bold: Colormap for the data points
    """
    if cmap_light is None:
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    if cmap_bold is None:
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Convert to PyTorch tensors if necessary
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    
    # Predict class for each point in the meshgrid
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
        if isinstance(Z, tuple):
            Z = Z[0]  # Some models return multiple outputs
        _, Z = torch.max(Z, 1)
        Z = Z.numpy()
    
    # Reshape the prediction to match the meshgrid shape
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # Plot the data points
    for i in np.unique(y):
        idx = (y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=cmap_bold(i), edgecolor='k', 
                   s=20, label=f'Class {i}')
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary")
    plt.legend()
    plt.show()

def visualize_weights(model, layer_index=0, figsize=(10, 5)):
    """
    Visualize the weights of a given layer as an image.
    
    Args:
        model: PyTorch model
        layer_index: Index of the linear layer to visualize
        figsize: Figure size
    """
    # Extract weights from the model
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    if layer_index >= len(linear_layers):
        print(f"Layer index {layer_index} out of range. Model has {len(linear_layers)} linear layers.")
        return
    
    # Get the weights of the specified layer
    weights = linear_layers[layer_index].weight.data.cpu().numpy()
    
    # Determine whether these are input weights for MNIST (784 input features)
    if weights.shape[1] == 784:  # First layer with flattened MNIST input
        # Reshape to 28x28 for visualization
        fig, axes = plt.subplots(4, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Plot first 16 filters
        num_filters = min(16, weights.shape[0])
        for i in range(num_filters):
            filter_weights = weights[i].reshape(28, 28)
            axes[i].imshow(filter_weights, cmap='viridis')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f'Weights of Layer {layer_index+1} (First 16 Neurons)', y=0.98)
        plt.tight_layout()
        plt.show()
    else:
        # For other layers, display as a heatmap
        plt.figure(figsize=figsize)
        plt.imshow(weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Weights of Layer {layer_index+1} (Shape: {weights.shape})')
        plt.xlabel('Input Features')
        plt.ylabel('Output Neurons')
        plt.tight_layout()
        plt.show()

def visualize_activations(model, data_loader, class_names=None, device='cpu'):
    """
    Visualize the activations of a neural network for a batch of inputs.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader containing the input data
        class_names: Dictionary mapping class indices to class names
        device: Device to use for inference (CPU or GPU)
    """
    # Get a single batch
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Select a single image for visualization
    img_idx = 0
    image = images[img_idx:img_idx+1]
    label = labels[img_idx].item()
    
    # Store activations
    activations = {}
    
    # Define forward hook
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks for all linear layers
    hooks = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(get_activation(f"{name}_{layer_idx}")))
            layer_idx += 1
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()
    
    # Remove the hooks
    for hook in hooks:
        hook.remove()
    
    # Display the original image
    plt.figure(figsize=(5, 5))
    img = image[0].cpu().numpy()
    if img.shape[0] == 1:  # For grayscale images
        plt.imshow(img[0], cmap='gray')
    else:  # For RGB images
        plt.imshow(np.transpose(img, (1, 2, 0)))
    
    label_text = f"Label: {label}"
    if class_names:
        label_text = f"Label: {class_names[label]}"
    
    pred_text = f"Prediction: {prediction}"
    if class_names:
        pred_text = f"Prediction: {class_names[prediction]}"
    
    plt.title(f"{label_text}\n{pred_text}")
    plt.axis('off')
    plt.show()
    
    # Visualize activations for each layer
    for name, activation in activations.items():
        if len(activation.shape) == 4:  # Convolutional layer output
            # Plot first 16 feature maps
            activation = activation[0]  # Remove batch dimension
            num_features = min(16, activation.shape[0])
            
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            axes = axes.flatten()
            
            for i in range(num_features):
                axes[i].imshow(activation[i], cmap='viridis')
                axes[i].set_title(f'Channel {i+1}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_features, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Activations: {name} (Shape: {activation.shape})', y=0.98)
            plt.tight_layout()
            plt.show()
        else:  # Fully connected layer output
            activation = activation[0]  # Remove batch dimension
            
            plt.figure(figsize=(10, 3))
            plt.plot(activation.numpy())
            plt.title(f'Activations: {name} (Shape: {activation.shape})')
            plt.xlabel('Neuron Index')
            plt.ylabel('Activation Value')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

def visualize_gradients(model, data, target, criterion, device='cpu'):
    """
    Visualize the gradients of a neural network for a given input and target.
    
    Args:
        model: PyTorch model
        data: Input data (batch)
        target: Target labels
        criterion: Loss function
        device: Device to use for computation (CPU or GPU)
    """
    # Ensure the model is in training mode
    model.train()
    
    # Move data to device
    data, target = data.to(device), target.to(device)
    
    # Zero the gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
    
    # Forward pass
    output = model(data)
    
    # Compute loss
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad.cpu().numpy()
    
    # Visualize gradients for each layer
    for name, grad in gradients.items():
        if 'weight' in name:  # Skip bias parameters for brevity
            plt.figure(figsize=(10, 5))
            
            # Flatten the gradient for visualization
            grad_flat = grad.flatten()
            
            # Plot the gradient distribution
            plt.subplot(1, 2, 1)
            plt.hist(grad_flat, bins=50)
            plt.title(f'Gradient Distribution: {name}')
            plt.xlabel('Gradient Value')
            plt.ylabel('Frequency')
            
            # Plot the gradient heatmap
            plt.subplot(1, 2, 2)
            plt.imshow(grad, cmap='coolwarm', aspect='auto')
            plt.title(f'Gradient Heatmap: {name}')
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()

def visualize_feature_space(model, data_loader, device='cpu', num_samples=1000, method='tsne'):
    """
    Visualize the feature space of a neural network using t-SNE or PCA.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader containing the input data
        device: Device to use for inference (CPU or GPU)
        num_samples: Number of samples to visualize
        method: 'tsne' or 'pca'
    """
    # Set model to evaluation mode
    model.eval()
    
    # Collect features and labels
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Limit number of samples
            if len(features) * images.shape[0] >= num_samples:
                break
                
            images = images.to(device)
            
            # Get the output of the second-to-last layer
            # This requires modifying the model or using a hook
            # For simplicity, we'll use the logits (final layer output)
            outputs = model(images)
            
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    
    # Concatenate all features and labels
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization of Feature Space'
    else:  # PCA
        reducer = PCA(n_components=2)
        title = 'PCA Visualization of Feature Space'
    
    # Reduce dimensionality
    embedded_features = reducer.fit_transform(features)
    
    # Visualize the embedded features
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with a different color for each class
    for i in np.unique(labels):
        idx = (labels == i)
        plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], label=f'Class {i}')
    
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def visualize_gradient_flow(model, data, target, criterion, device='cpu'):
    """
    Visualize the gradient flow (magnitude) through a neural network.
    
    Args:
        model: PyTorch model
        data: Input data (batch)
        target: Target labels
        criterion: Loss function
        device: Device to use for computation (CPU or GPU)
    """
    # Ensure the model is in training mode
    model.train()
    
    # Move data to device
    data, target = data.to(device), target.to(device)
    
    # Forward pass
    output = model(data)
    
    # Compute loss
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Collect gradient norms
    grad_norms = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            layer_names.append(name)
    
    # Visualize gradient norms
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(grad_norms)), grad_norms, align='center')
    plt.yticks(range(len(grad_norms)), layer_names)
    plt.xlabel('Gradient Norm')
    plt.title('Gradient Flow')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load a subset of MNIST for quick testing
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Use a small subset for testing
    subset_size = 100
    indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = torch.utils.data.Subset(train_dataset, indices)
    
    # Create data loader
    data_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    
    # Define a simple model for testing
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super(SimpleTestModel, self).__init__()
            self.fc1 = nn.Linear(28*28, 64)
            self.fc2 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create model, loss function, and optimizer
    model = SimpleTestModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Get a batch for testing
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Example: Visualize weights
    print("Visualizing initial random weights:")
    visualize_weights(model, layer_index=0)
    
    # Example: Visualize activations
    print("Visualizing activations for a sample input:")
    visualize_activations(model, data_loader, device=device)
    
    # Example: Visualize gradients
    print("Visualizing gradients for a sample input:")
    visualize_gradients(model, images, labels, criterion, device=device)
    
    # Example: Visualize gradient flow
    print("Visualizing gradient flow:")
    visualize_gradient_flow(model, images, labels, criterion, device=device)
