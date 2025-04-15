"""
Deep Learning Primer
Building Blocks of Generative AI Course - Day 1

This script demonstrates fundamental deep learning concepts including:
1. Neural network architecture
2. Gradient descent
3. Backpropagation
4. Basic model training and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    """Load and preprocess the Fashion MNIST dataset"""
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for the model
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    # One-hot encode the labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

class BasicNeuralNetworkVisualizer:
    """Class to build, train, and visualize a basic neural network"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def build_model(self, input_shape, hidden_units=128, dropout_rate=0.2):
        """Build a basic feedforward neural network"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Hidden layers
            layers.Dense(hidden_units, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_units // 2, activation='relu'),
            layers.Dropout(dropout_rate),
            
            # Output layer
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
        """Train the model and record the history"""
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks
        )
        
        self.history = history
        return history
    
    def evaluate_model(self, x_test, y_test):
        """Evaluate the model on test data"""
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        return test_acc
    
    def plot_training_history(self):
        """Plot the training and validation learning curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()
    
    def visualize_predictions(self, x_test, y_test, num_samples=5):
        """Visualize model predictions on random test samples"""
        # Get random indices
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        
        # Get samples
        x_samples = x_test[indices]
        y_true = np.argmax(y_test[indices], axis=1)
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(x_samples), axis=1)
        
        # Plot the results
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(x_samples[i].reshape(28, 28), cmap='gray')
            
            # Add color to title based on prediction correctness
            title_color = 'green' if y_pred[i] == y_true[i] else 'red'
            plt.title(f"True: {self.class_names[y_true[i]]}\nPred: {self.class_names[y_pred[i]]}", 
                      color=title_color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.show()
    
    def visualize_gradients(self):
        """Visualize gradients during backpropagation with a simple example"""
        # Create a small 1D regression problem
        x = np.linspace(-5, 5, 200).reshape(-1, 1)
        y = 0.5 * x**2 + np.random.normal(0, 1, x.shape)
        
        # Create a single-layer model
        simple_model = keras.Sequential([
            layers.Dense(1, input_shape=(1,))
        ])
        
        # Use SGD optimizer with visible learning rate
        opt = keras.optimizers.SGD(learning_rate=0.01)
        simple_model.compile(optimizer=opt, loss='mse')
        
        # Plot the function and gradient descent steps
        plt.figure(figsize=(15, 10))
        
        # Initial model state
        initial_weight = simple_model.layers[0].get_weights()[0][0][0]
        initial_bias = simple_model.layers[0].get_weights()[1][0]
        print(f"Initial weight: {initial_weight:.4f}, bias: {initial_bias:.4f}")
        
        # Store weights and biases during training
        weights = [initial_weight]
        biases = [initial_bias]
        
        # Training loop with gradient visualization
        epochs = 10
        for epoch in range(epochs):
            # Perform one epoch of training
            history = simple_model.fit(x, y, epochs=1, verbose=0)
            
            # Get current weights
            current_weight = simple_model.layers[0].get_weights()[0][0][0]
            current_bias = simple_model.layers[0].get_weights()[1][0]
            weights.append(current_weight)
            biases.append(current_bias)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Weight: {current_weight:.4f}, Bias: {current_bias:.4f}")
            print(f"  Loss: {history.history['loss'][0]:.4f}")
        
        # Plot data points
        plt.subplot(2, 2, 1)
        plt.scatter(x, y, alpha=0.3, label='Data')
        
        # Plot initial model prediction
        x_range = np.linspace(-5, 5, 100).reshape(-1, 1)
        y_init = initial_weight * x_range + initial_bias
        plt.plot(x_range, y_init, 'r--', label='Initial model')
        
        # Plot final model prediction
        y_final = current_weight * x_range + current_bias
        plt.plot(x_range, y_final, 'g-', linewidth=2, label='Final model')
        
        plt.title('Model Before and After Training')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        
        # Plot loss curve
        plt.subplot(2, 2, 2)
        plt.plot(np.arange(0, epochs+1), [np.nan] + history.history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot weight changes
        plt.subplot(2, 2, 3)
        plt.plot(np.arange(0, epochs+1), weights)
        plt.title('Weight Value During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        
        # Plot bias changes
        plt.subplot(2, 2, 4)
        plt.plot(np.arange(0, epochs+1), biases)
        plt.title('Bias Value During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Bias')
        
        plt.tight_layout()
        plt.savefig('gradient_descent.png')
        plt.show()

def demonstrate_optimization_techniques():
    """Demonstrate different optimization techniques"""
    # Create a simple dataset
    np.random.seed(42)
    x = np.random.rand(1000, 20)
    y = 0.5 * np.sum(x, axis=1) + np.random.normal(0, 0.1, size=1000)
    y = y.reshape(-1, 1)
    
    # Split data
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    
    # Define base model architecture
    def get_model():
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(20,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        return model
    
    # List of optimizers to compare
    optimizers = [
        keras.optimizers.SGD(learning_rate=0.01, name="SGD"),
        keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, name="SGD with Momentum"),
        keras.optimizers.Adam(learning_rate=0.001, name="Adam"),
        keras.optimizers.RMSprop(learning_rate=0.001, name="RMSprop"),
        keras.optimizers.Adagrad(learning_rate=0.01, name="Adagrad")
    ]
    
    # Train with each optimizer
    histories = {}
    for opt in optimizers:
        # Create and compile model
        model = get_model()
        model.compile(optimizer=opt, loss='mse')
        
        # Train model
        print(f"Training with {opt.name}...")
        history = model.fit(
            x_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(x_val, y_val),
            verbose=0
        )
        
        # Store history
        histories[opt.name] = history.history
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history['loss'], label=name)
    
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history['val_loss'], label=name)
    
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Split training data for validation
    val_split = int(0.1 * len(x_train))
    x_val, y_val = x_train[:val_split], y_train[:val_split]
    x_train, y_train = x_train[val_split:], y_train[val_split:]
    
    print("Building and training neural network...")
    nn_viz = BasicNeuralNetworkVisualizer()
    nn_viz.build_model(input_shape=(28*28,))
    nn_viz.train_model(x_train, y_train, x_val, y_val, epochs=5)
    
    print("Evaluating model performance...")
    nn_viz.evaluate_model(x_test, y_test)
    
    print("Visualizing training history...")
    nn_viz.plot_training_history()
    
    print("Visualizing model predictions...")
    nn_viz.visualize_predictions(x_test, y_test)
    
    print("Visualizing gradient descent...")
    nn_viz.visualize_gradients()
    
    print("Demonstrating optimization techniques...")
    demonstrate_optimization_techniques()
    
    print("Deep Learning Primer demo complete!")
