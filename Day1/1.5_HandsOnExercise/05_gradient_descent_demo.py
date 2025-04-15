"""
Gradient Descent Demonstration from Scratch
Building Blocks of Generative AI Course - Day 1

This script demonstrates gradient descent optimization from scratch without using
automatic differentiation or deep learning frameworks. It focuses on the core
principles of gradient descent and backpropagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SimpleNeuralNetwork:
    """
    A simple 2-layer neural network implemented from scratch.
    
    Architecture:
    - Input layer: input_size neurons
    - Hidden layer: hidden_size neurons with ReLU activation
    - Output layer: output_size neurons with softmax activation
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes
            learning_rate: Learning rate for gradient descent
        """
        # Initialize weights and biases
        # Xavier/Glorot initialization for better convergence
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Learning rate
        self.learning_rate = learning_rate
        
        # Store for visualization
        self.loss_history = []
        self.weights_history = []
        self.save_weights()
    
    def save_weights(self):
        """Save the current weights for visualization"""
        self.weights_history.append({
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        })
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        Compute cross entropy loss.
        
        Args:
            y_pred: Predicted probabilities (after softmax)
            y_true: One-hot encoded true labels
            
        Returns:
            Average cross entropy loss
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape [batch_size, input_size]
            
        Returns:
            Tuple of (hidden layer activations, output probabilities)
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a1, self.a2
    
    def backward(self, X, y):
        """
        Backward pass to compute gradients.
        
        Args:
            X: Input data of shape [batch_size, input_size]
            y: One-hot encoded target labels
        """
        batch_size = X.shape[0]
        
        # Gradient for output layer
        dz2 = self.a2 - y  # Derivative of softmax+cross-entropy
        dW2 = np.dot(self.a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0) / batch_size
        
        # Gradient for hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0) / batch_size
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """
        Update weights using gradients.
        
        Args:
            dW1, db1, dW2, db2: Gradients of the weights and biases
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        
        # Save updated weights
        self.save_weights()
    
    def train(self, X, y, epochs=100, verbose=True):
        """
        Train the neural network.
        
        Args:
            X: Input data of shape [batch_size, input_size]
            y: One-hot encoded target labels
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            Loss history
        """
        for epoch in range(epochs):
            # Forward pass
            _, y_pred = self.forward(X)
            
            # Compute loss
            loss = self.cross_entropy_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y)
            
            # Update weights
            self.update_weights(dW1, db1, dW2, db2)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                # Compute accuracy
                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y, axis=1)
                accuracy = np.mean(predictions == true_labels)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return self.loss_history
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Args:
            X: Input data of shape [batch_size, input_size]
            
        Returns:
            Predicted class labels
        """
        _, y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

def generate_spiral_data(samples_per_class=100, num_classes=3, noise=0.1):
    """
    Generate a spiral dataset for classification.
    
    Args:
        samples_per_class: Number of samples per class
        num_classes: Number of classes
        noise: Amount of noise to add
        
    Returns:
        Tuple of (inputs, one-hot encoded targets)
    """
    n = samples_per_class * num_classes
    X = np.zeros((n, 2))
    y = np.zeros(n, dtype=int)
    
    for j in range(num_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0, 1, samples_per_class)
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    
    # Convert to one-hot encoding
    y_one_hot = np.zeros((n, num_classes))
    y_one_hot[np.arange(n), y] = 1
    
    return X, y_one_hot, y

def visualize_decision_boundary(X, y, model, title='Decision Boundary'):
    """
    Visualize the decision boundary of a model.
    
    Args:
        X: Input features
        y: Target labels (not one-hot)
        model: Trained model with predict method
        title: Title for the plot
    """
    h = 0.02  # Step size for the meshgrid
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    
    # Predict class labels for the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

def visualize_training_animation(X, y, model, interval=50):
    """
    Create an animation of the training process showing decision boundary changes.
    
    Args:
        X: Input features
        y: Target labels (not one-hot)
        model: Trained model with weights_history
        interval: Interval between frames in milliseconds
    """
    h = 0.02  # Step size for the meshgrid
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    
    # Create a figure for animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
    
    # Plot the initial decision boundary
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    def init():
        # Initial frame
        return [scatter]
    
    def update(frame_idx):
        # Update the model weights with the weights from this frame
        weights = model.weights_history[frame_idx]
        model.W1 = weights['W1']
        model.b1 = weights['b1']
        model.W2 = weights['W2']
        model.b2 = weights['b2']
        
        # Predict class labels for the meshgrid
        Z = model.predict(mesh_points).reshape(xx.shape)
        
        # Clear previous contour plot
        ax.clear()
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
        
        # Plot the training points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f"Training Step {frame_idx}")
        
        return [scatter]
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(model.weights_history),
                       init_func=init, blit=False, interval=interval)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def plot_loss_curve(loss_history):
    """
    Plot the loss curve during training.
    
    Args:
        loss_history: List of loss values during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

def demonstrate_linear_regression():
    """Demonstrate gradient descent on a simple linear regression problem."""
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Initialize parameters
    theta = np.random.randn(2, 1)  # Random initialization
    X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
    n_iterations = 1000
    learning_rate = 0.01
    
    # Store history for visualization
    theta_history = [theta.copy()]
    mse_history = []
    
    # Gradient descent
    for iteration in range(n_iterations):
        # Compute predictions
        y_pred = X_b.dot(theta)
        
        # Compute error
        error = y_pred - y
        
        # Compute gradients
        gradients = 2/100 * X_b.T.dot(error)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Store for visualization
        theta_history.append(theta.copy())
        mse = np.mean(error**2)
        mse_history.append(mse)
    
    # Visualize the data and the regression line
    plt.figure(figsize=(10, 8))
    
    # Original data points
    plt.subplot(2, 1, 1)
    plt.scatter(X, y)
    
    # Final regression line
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(theta)
    plt.plot(X_new, y_predict, 'r-', linewidth=2, label='Predictions')
    
    plt.title('Linear Regression with Gradient Descent')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    # Plot the MSE history
    plt.subplot(2, 1, 2)
    plt.plot(mse_history)
    plt.title('Mean Squared Error During Training')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final parameters: theta_0 = {theta[0][0]:.4f}, theta_1 = {theta[1][0]:.4f}")
    print(f"True parameters: theta_0 = 4, theta_1 = 3")

def demonstrate_logistic_regression():
    """Demonstrate gradient descent on a simple logistic regression problem."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = y.reshape(-1, 1)
    
    # Initialize parameters
    theta = np.random.randn(3, 1)  # Random initialization
    X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
    n_iterations = 1000
    learning_rate = 0.1
    
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Store history for visualization
    theta_history = [theta.copy()]
    loss_history = []
    
    # Gradient descent
    for iteration in range(n_iterations):
        # Compute predictions
        z = X_b.dot(theta)
        y_pred = sigmoid(z)
        
        # Compute error
        error = y_pred - y
        
        # Compute gradients
        gradients = X_b.T.dot(error) / len(y)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Store for visualization
        theta_history.append(theta.copy())
        
        # Compute loss (binary cross entropy)
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        loss_history.append(loss)
    
    # Visualize the data and the decision boundary
    plt.figure(figsize=(10, 8))
    
    # Original data points
    plt.subplot(2, 1, 1)
    plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c='blue', marker='o', label='Class 0')
    plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c='red', marker='x', label='Class 1')
    
    # Decision boundary
    x0_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x1_range = -theta[0] / theta[2] - theta[1] / theta[2] * x0_range  # Solve for x1 where theta0 + theta1*x0 + theta2*x1 = 0
    plt.plot(x0_range, x1_range, 'g-', linewidth=2, label='Decision Boundary')
    
    plt.title('Logistic Regression with Gradient Descent')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    
    # Plot the loss history
    plt.subplot(2, 1, 2)
    plt.plot(loss_history)
    plt.title('Loss During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final parameters: theta_0 = {theta[0][0]:.4f}, theta_1 = {theta[1][0]:.4f}, theta_2 = {theta[2][0]:.4f}")

if __name__ == "__main__":
    # Demonstration options
    demo_option = "nn"  # Choose from: "linear", "logistic", "nn"
    
    if demo_option == "linear":
        print("Demonstrating gradient descent for linear regression...")
        demonstrate_linear_regression()
    
    elif demo_option == "logistic":
        print("Demonstrating gradient descent for logistic regression...")
        demonstrate_logistic_regression()
    
    else:  # Neural network demo
        print("Demonstrating gradient descent for neural network training...")
        
        # Generate synthetic spiral data
        X, y_one_hot, y = generate_spiral_data(samples_per_class=100, num_classes=3, noise=0.1)
        
        # Visualize the data
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
        plt.title("Spiral Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
        
        # Create and train the model
        input_size = 2
        hidden_size = 50
        output_size = 3
        learning_rate = 0.1
        
        model = SimpleNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
        
        print("Training neural network...")
        loss_history = model.train(X, y_one_hot, epochs=300)
        
        # Plot the loss curve
        plot_loss_curve(loss_history)
        
        # Visualize the final decision boundary
        visualize_decision_boundary(X, y, model, title="Neural Network Decision Boundary")
        
        # Optional: Visualize training animation
        # Note: This can be computationally intensive
        visualize_training_animation(X, y, model, interval=100)
