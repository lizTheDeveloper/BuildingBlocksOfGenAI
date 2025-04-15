"""
Neural Network Model for Fashion MNIST
Building Blocks of Generative AI Course - Day 1

This script defines a simple neural network model for classifying Fashion MNIST images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """
    A simple neural network with fully connected layers.
    
    Architecture:
    - Input layer: 784 neurons (28x28 flattened images)
    - Hidden layer 1: 128 neurons with ReLU activation
    - Hidden layer 2: 64 neurons with ReLU activation
    - Output layer: 10 neurons (one for each class)
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # Define the network layers
        self.fc1 = nn.Linear(28*28, 128)  # Input to hidden layer 1
        self.fc2 = nn.Linear(128, 64)     # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(64, 10)      # Hidden layer 2 to output
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            Tensor of shape [batch_size, 10] containing class logits
        """
        # Flatten the input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(-1, 28*28)
        
        # Hidden layer 1 with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Hidden layer 2 with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Output layer (no activation, will be applied in loss function)
        x = self.fc3(x)
        
        return x

class EnhancedNN(nn.Module):
    """
    An enhanced neural network with dropout for regularization.
    
    Architecture:
    - Input layer: 784 neurons (28x28 flattened images)
    - Hidden layer 1: 256 neurons with ReLU activation
    - Dropout layer 1: 20% dropout rate
    - Hidden layer 2: 128 neurons with ReLU activation
    - Dropout layer 2: 20% dropout rate
    - Output layer: 10 neurons (one for each class)
    """
    def __init__(self, dropout_rate=0.2):
        super(EnhancedNN, self).__init__()
        
        # Define the network layers
        self.fc1 = nn.Linear(28*28, 256)  # Input to hidden layer 1
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)    # Hidden layer 1 to hidden layer 2
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)     # Hidden layer 2 to output
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            Tensor of shape [batch_size, 10] containing class logits
        """
        # Flatten the input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(-1, 28*28)
        
        # Hidden layer 1 with ReLU activation and dropout
        x = self.dropout1(F.relu(self.fc1(x)))
        
        # Hidden layer 2 with ReLU activation and dropout
        x = self.dropout2(F.relu(self.fc2(x)))
        
        # Output layer (no activation, will be applied in loss function)
        x = self.fc3(x)
        
        return x

class ConvNN(nn.Module):
    """
    A convolutional neural network for image classification.
    
    Architecture:
    - Convolutional layer 1: 32 filters of size 3x3
    - Max pooling layer 1: 2x2 pool size
    - Convolutional layer 2: 64 filters of size 3x3
    - Max pooling layer 2: 2x2 pool size
    - Fully connected layer 1: 128 neurons
    - Output layer: 10 neurons (one for each class)
    """
    def __init__(self):
        super(ConvNN, self).__init__()
        
        # Define the network architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 7x7 is the feature map size after two 2x2 pooling operations on 28x28 input
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            Tensor of shape [batch_size, 10] containing class logits
        """
        # Convolutional layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the feature maps
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model, input_size=(1, 28, 28)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
    """
    # Create a dummy input tensor
    batch_size = 1
    dummy_input = torch.zeros(batch_size, *input_size)
    
    # Register forward hooks to get layer output sizes
    summary = []
    hooks = []
    
    def register_hook(module):
        def hook(module, input, output):
            # For multiple output tensors, just take the first one
            if isinstance(output, (list, tuple)):
                output = output[0]
                
            summary.append({
                'name': module.__class__.__name__,
                'input_shape': list(input[0].size()),
                'output_shape': list(output.size()),
                'params': count_parameters(module)
            })
        
        hooks.append(module.register_forward_hook(hook))
    
    # Register hooks for all modules
    model.apply(register_hook)
    
    # Forward pass
    model(dummy_input)
    
    # Remove the hooks
    for hook in hooks:
        hook.remove()
    
    # Print the summary
    total_params = 0
    print(f"{'Layer':<15} {'Input Shape':<20} {'Output Shape':<20} {'Params':<10}")
    print('-' * 70)
    
    for layer in summary:
        name = layer['name']
        input_shape = str(layer['input_shape'])
        output_shape = str(layer['output_shape'])
        params = layer['params']
        total_params += params
        
        print(f"{name:<15} {input_shape:<20} {output_shape:<20} {params:<10,}")
    
    print('-' * 70)
    print(f"Total Params: {total_params:,}")

if __name__ == "__main__":
    # Create the models
    simple_model = SimpleNN()
    enhanced_model = EnhancedNN()
    conv_model = ConvNN()
    
    # Print model summaries
    print("\nSimple Neural Network:")
    get_model_summary(simple_model)
    
    print("\nEnhanced Neural Network:")
    get_model_summary(enhanced_model)
    
    print("\nConvolutional Neural Network:")
    get_model_summary(conv_model)
