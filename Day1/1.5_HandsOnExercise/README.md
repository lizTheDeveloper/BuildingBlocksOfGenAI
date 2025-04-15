# Hands-On Neural Network Exercise
## Building Blocks of Generative AI Course - Day 1

This folder contains a hands-on exercise for understanding neural networks, gradient descent, and backpropagation - core concepts that form the foundation of generative AI models.

## Overview

In this exercise, you will:

1. Work with the Fashion MNIST dataset
2. Build and train neural networks from scratch and with PyTorch
3. Visualize the training process, weights, and decision boundaries
4. See how gradient descent optimizes model parameters
5. Learn about various activation functions and loss functions

## Files in this Exercise

- `01_data_loading.py` - Functions for loading and visualizing the Fashion MNIST dataset
- `02_neural_network_model.py` - Neural network model architectures (simple, enhanced, and convolutional)
- `03_training_functions.py` - Functions for training and evaluating neural networks
- `04_visualization_tools.py` - Tools for visualizing various aspects of neural networks
- `05_gradient_descent_demo.py` - Demonstration of gradient descent from scratch
- `06_main.py` - Main script to run the exercise

## Getting Started

### Option 1: Google Colab (Recommended)

1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Create a new notebook
3. Upload the Python files from this folder
4. Run the following commands:

```python
# Install required packages
!pip install torch torchvision matplotlib scikit-learn numpy

# Import the main script and run it
from google.colab import files
files.upload()  # Upload the Python files

# Run the main script
%run 06_main.py --model simple --epochs 5 --visualize
```

### Option 2: Local Development

1. Install the required packages:

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

2. Run the main script:

```bash
python 06_main.py --model simple --epochs 5 --visualize
```

## Command Line Arguments

The main script accepts the following arguments:

- `--model`: Model architecture to use (`simple`, `enhanced`, or `conv`)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs to train (default: 5)
- `--lr`: Learning rate (default: 0.01)
- `--momentum`: Momentum for SGD optimizer (default: 0.9)
- `--visualize`: Enable additional visualizations

## Gradient Descent Demo

The gradient descent demo provides a separate example that implements a neural network from scratch without using automatic differentiation. To run it:

```bash
python 05_gradient_descent_demo.py
```

This will demonstrate gradient descent on:
- A spiral classification problem using a neural network
- A linear regression problem
- A logistic regression problem

## Alternative Datasets

This exercise uses Fashion MNIST by default, but you can explore other datasets:

1. Digits Dataset (scikit-learn)
2. CIFAR-10
3. California Housing Dataset
4. And many others

See the output of `describe_alternative_datasets()` function for more options.

## Learning Objectives

By the end of this exercise, you should understand:

1. The basic structure of neural networks
2. How forward and backward passes work
3. How parameters are updated using gradient descent
4. How to implement, train, and evaluate neural networks in PyTorch
5. How to visualize and interpret neural network training

This knowledge forms the foundation for understanding more complex generative models like GANs, VAEs, and Transformers that we'll cover in the upcoming sections.
