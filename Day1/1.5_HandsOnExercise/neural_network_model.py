"""
Neural Network Model Implementation
Building Blocks of Generative AI Course - Day 1

This module provides a simple neural network implementation for the hands-on exercise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """
    A simple neural network model for image classification.
    
    The network takes a 28x28 grayscale image (as used in MNIST/Fashion MNIST)
    and outputs class probabilities for multi-class classification.
    """
    
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, num_classes=10):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of the input features (default: 28*28=784 for flattened MNIST images)
            hidden1_size: Size of the first hidden layer
            hidden2_size: Size of the second hidden layer
            num_classes: Number of output classes
        """
        super(SimpleNN, self).__init__()
        
        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden1_size)  # Input to hidden layer 1
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)  # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(hidden2_size, num_classes)  # Hidden layer 2 to output
        
        # Define dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               For MNIST/Fashion MNIST: [batch_size, 1, 28, 28]
               
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Flatten the input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(-1, 28*28)
        
        # Apply first hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        
        # Apply second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        
        # Apply output layer (no activation, as we'll use cross-entropy loss)
        x = self.fc3(x)
        
        return x

class CustomNeuralNetwork(nn.Module):
    """
    A customizable neural network for students to experiment with different architectures.
    
    This class allows students to modify the network architecture by changing
    the number of layers, layer sizes, and activation functions.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, 
                 activation=F.relu, dropout_rate=0.2):
        """
        Initialize a custom neural network.
        
        Args:
            input_size: Size of the input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            activation: Activation function to use (default: ReLU)
            dropout_rate: Dropout rate for regularization
        """
        super(CustomNeuralNetwork, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Add the first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Add additional hidden layers
        for i in range(len(hidden_sizes) - 1):
            # Add activation and dropout after each layer
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Add the next hidden layer
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Add dropout before the output layer
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Convert the list of layers into a sequential model
        self.layers = nn.ModuleList(layers)
        
        # Store the activation function
        self.activation = activation
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Flatten the input if it's an image
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply activation to all but the last layer
            if i < len(self.layers) - 1 and isinstance(layer, nn.Linear):
                x = self.activation(x)
        
        return x

# Example: Create and inspect a model
if __name__ == "__main__":
    # Create a simple neural network
    simple_model = SimpleNN()
    print("Simple Neural Network:")
    print(simple_model)
    
    # Create a custom neural network with 3 hidden layers
    custom_model = CustomNeuralNetwork(
        hidden_sizes=[256, 128, 64],
        dropout_rate=0.3,
        activation=F.leaky_relu
    )
    print("\nCustom Neural Network:")
    print(custom_model)
    
    # Test with a random input
    x = torch.randn(16, 1, 28, 28)  # [batch_size, channels, height, width]
    
    # Forward pass through simple model
    output_simple = simple_model(x)
    print(f"\nSimple model output shape: {output_simple.shape}")
    
    # Forward pass through custom model
    output_custom = custom_model(x)
    print(f"Custom model output shape: {output_custom.shape}")
