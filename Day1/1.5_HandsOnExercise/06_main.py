"""
Main Script for Hands-On Neural Network Exercise
Building Blocks of Generative AI Course - Day 1

This script serves as an entry point for the hands-on exercise, combining all the components
from the other modules.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Import from other modules
from data_loading import load_fashion_mnist, visualize_dataset_samples, describe_alternative_datasets
from neural_network_model import SimpleNN, EnhancedNN, ConvNN, get_model_summary
from training_functions import train_model, plot_training_results, visualize_predictions
from visualization_tools import visualize_weights, visualize_activations, visualize_feature_space

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural Network Exercise')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'enhanced', 'conv'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--visualize', action='store_true', help='Enable additional visualizations')
    
    return parser.parse_args()

def main():
    """Main function to run the exercise"""
    # Parse arguments
    try:
        args = parse_arguments()
    except:
        # For environments that don't support argparse (like notebooks)
        class Args:
            model = 'simple'
            batch_size = 64
            epochs = 5
            lr = 0.01
            momentum = 0.9
            visualize = True
        args = Args()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the data
    print("\n1. Loading Fashion MNIST dataset...")
    train_loader, test_loader, class_names = load_fashion_mnist(batch_size=args.batch_size)
    
    # Display some sample images
    # Get examples from the train loader
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples, "
          f"{len(test_loader.dataset)} test samples")
    
    print("\n2. Visualizing sample images...")
    visualize_dataset_samples(train_loader.dataset, num_samples=8, class_names=class_names)
    
    # Create the model
    print(f"\n3. Creating {args.model} neural network model...")
    if args.model == 'simple':
        model = SimpleNN().to(device)
    elif args.model == 'enhanced':
        model = EnhancedNN().to(device)
    else:  # conv
        model = ConvNN().to(device)
    
    # Display model summary
    get_model_summary(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Train the model
    print(f"\n4. Training the model for {args.epochs} epochs...")
    train_losses, test_losses, test_accuracies = train_model(
        model, device, train_loader, test_loader, optimizer, criterion, args.epochs
    )
    
    # Plot training results
    print("\n5. Plotting training results...")
    plot_training_results(train_losses, test_losses, test_accuracies)
    
    # Visualize predictions
    print("\n6. Visualizing model predictions...")
    visualize_predictions(model, device, test_loader, class_names, num_samples=8)
    
    # Additional visualizations if requested
    if args.visualize:
        print("\n7. Visualizing model weights...")
        visualize_weights(model, layer_index=0)
        
        print("\n8. Visualizing activations for a sample input...")
        visualize_activations(model, test_loader, class_names, device)
        
        print("\n9. Visualizing feature space with t-SNE...")
        visualize_feature_space(model, test_loader, device, num_samples=500, method='tsne')
    
    # Mention alternative datasets
    print("\n10. Alternative datasets for similar exercises:")
    describe_alternative_datasets()

if __name__ == "__main__":
    main()
