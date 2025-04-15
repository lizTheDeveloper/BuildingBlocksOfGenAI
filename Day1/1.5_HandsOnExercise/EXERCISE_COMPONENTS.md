# Exercise Components - Neural Networks and Gradient Descent

This document explains the components of the hands-on exercise for the 7-8am time block on Day 1 of the Building Blocks of Generative AI course.

## Exercise Structure

The exercise is broken down into modular components to facilitate understanding and to make it easier to integrate with Google Colab. The recommended order for exploring these components is:

1. Data Loading
2. Neural Network Models
3. Training Functions
4. Visualization Tools
5. Gradient Descent Demo
6. Main Script Integration

## Component Descriptions

### 1. Data Loading Module (`01_data_loading.py`)

**Purpose:** Provides utilities for loading and visualizing the Fashion MNIST dataset.

**Key Features:**
- Loads Fashion MNIST using PyTorch's datasets and transforms
- Creates DataLoader objects for batched training
- Includes visualization tools for displaying sample images
- Contains a comprehensive list of 10 alternative datasets with descriptions

**Educational Value:**
- Introduces students to working with image data
- Demonstrates data preprocessing for neural networks
- Shows how to create efficient data pipelines with batching

### 2. Neural Network Models Module (`02_neural_network_model.py`)

**Purpose:** Defines neural network architectures of increasing complexity.

**Key Features:**
- SimpleNN: Basic fully-connected network with 2 hidden layers
- EnhancedNN: Adds dropout regularization to improve generalization
- ConvNN: Implements a convolutional neural network for better image processing
- Includes utilities for displaying model summary statistics

**Educational Value:**
- Demonstrates how to define neural networks using PyTorch
- Shows different model architectures and their components
- Introduces regularization techniques to prevent overfitting

### 3. Training Functions Module (`03_training_functions.py`)

**Purpose:** Implements the training and evaluation process for neural networks.

**Key Features:**
- Functions for training a model for one epoch
- Evaluation function for testing model performance
- Full training loop with metrics tracking
- Visualization functions for displaying training progress
- Tools for visualizing model predictions on test data

**Educational Value:**
- Shows the training loop structure (forward pass, loss computation, backward pass, weight updates)
- Demonstrates how to track and visualize metrics during training
- Explains model evaluation techniques

### 4. Visualization Tools Module (`04_visualization_tools.py`)

**Purpose:** Provides advanced visualization functions for neural network internals.

**Key Features:**
- Functions to visualize weights and filters
- Tools to display activations across the network
- Dimensionality reduction visualizations of feature spaces (t-SNE, PCA)
- Gradient visualization tools
- Decision boundary visualizations

**Educational Value:**
- Helps students understand what's happening inside the "black box" of neural networks
- Demonstrates how features are learned and transformed through the network
- Shows how the model represents data in different spaces

### 5. Gradient Descent Demo (`05_gradient_descent_demo.py`)

**Purpose:** Implements neural networks, linear regression, and logistic regression from scratch to demonstrate gradient descent.

**Key Features:**
- Neural network implementation using only NumPy (no automatic differentiation)
- Manual forward and backward propagation
- Animated visualizations of the training process
- Implementations for three types of problems:
  - Neural network classification
  - Linear regression
  - Logistic regression

**Educational Value:**
- Provides a deep understanding of how gradient descent works
- Shows backpropagation calculations step by step
- Demonstrates the core mathematics without relying on automatic differentiation

### 6. Main Script (`06_main.py`)

**Purpose:** Integrates all components into a complete workflow.

**Key Features:**
- Command-line argument parsing for flexible experimentation
- Step-by-step execution of the full machine learning pipeline
- Integration of all visualization components
- Summary of alternative datasets for further exploration

**Educational Value:**
- Shows how to integrate all components into a cohesive workflow
- Demonstrates best practices for organizing machine learning code
- Provides a complete example that students can extend and modify

## Learning Path

The exercise is designed to build understanding progressively:

1. Start with the **data loading** to understand the dataset
2. Explore the **model architectures** to understand neural network structure
3. Examine the **training functions** to see how optimization works
4. Use the **visualization tools** to peek inside the networks
5. Study the **gradient descent implementation** to understand the math from first principles
6. Run the **main script** to see everything working together

This progressive approach helps students build intuition for how neural networks, gradient descent, and backpropagation work—all foundational concepts necessary for understanding the generative AI models covered later in the course.
