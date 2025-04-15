"""
Latent Space and Representation Learning
Building Blocks of Generative AI Course - Day 1

This script demonstrates the concept of latent space and representation learning
using a simple autoencoder model on MNIST digits.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SimpleAutoencoder:
    """
    A simple autoencoder model for demonstrating latent space representation.
    """
    
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
    
    def build_models(self, input_shape=(28, 28, 1)):
        """
        Build the encoder and decoder networks.
        
        Args:
            input_shape: Shape of the input data (height, width, channels)
        """
        # Encoder
        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.Flatten()(encoder_inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        encoder_outputs = layers.Dense(self.latent_dim, activation='linear', name='latent_vector')(x)
        
        self.encoder = keras.Model(encoder_inputs, encoder_outputs, name='encoder')
        
        # Decoder
        decoder_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation='relu')(decoder_inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
        decoder_outputs = layers.Reshape(input_shape)(x)
        
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
        
        # Autoencoder (end-to-end model)
        autoencoder_inputs = keras.Input(shape=input_shape)
        encoded = self.encoder(autoencoder_inputs)
        decoded = self.decoder(encoded)
        
        self.autoencoder = keras.Model(autoencoder_inputs, decoded, name='autoencoder')
        self.autoencoder.compile(
            optimizer='adam',
            loss='binary_crossentropy'
        )
        
        return self.encoder, self.decoder, self.autoencoder
    
    def train(self, x_train, epochs=10, batch_size=128, validation_split=0.2):
        """
        Train the autoencoder model.
        
        Args:
            x_train: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Training history
        """
        if self.autoencoder is None:
            self.build_models()
        
        # Normalize and reshape the data
        x_train = x_train.astype('float32') / 255.0
        x_train = np.expand_dims(x_train, axis=-1)
        
        # Train the model
        history = self.autoencoder.fit(
            x_train, x_train,  # Input is the same as target for autoencoders
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True
        )
        
        return history

def visualize_latent_space(encoder, data, labels, save_path="latent_space_2d.png"):
    """
    Visualize the latent space of the trained encoder.
    
    Args:
        encoder: Trained encoder model
        data: Input data to encode
        labels: Labels for the input data
        save_path: Path to save the visualization
    """
    # Normalize and reshape the data
    data = data.astype('float32') / 255.0
    data = np.expand_dims(data, axis=-1)
    
    # Encode the data
    encoded_data = encoder.predict(data)
    
    plt.figure(figsize=(10, 8))
    
    # If latent space is 2D, plot directly
    if encoded_data.shape[1] == 2:
        plt.scatter(
            encoded_data[:, 0], encoded_data[:, 1],
            c=labels, cmap='tab10', alpha=0.8, s=10
        )
        plt.colorbar()
        plt.grid(True, alpha=0.3)
        plt.title('2D Latent Space')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        
    # If latent space is higher dimensional, use dimensionality reduction
    else:
        # Use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        encoded_data_2d = tsne.fit_transform(encoded_data)
        
        plt.scatter(
            encoded_data_2d[:, 0], encoded_data_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.8, s=10
        )
        plt.colorbar()
        plt.grid(True, alpha=0.3)
        plt.title(f'{encoded_data.shape[1]}D Latent Space (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
    
    plt.savefig(save_path)
    plt.show()

def visualize_digit_reconstruction(autoencoder, data, n=10, save_path="digit_reconstruction.png"):
    """
    Visualize original and reconstructed digits side by side.
    
    Args:
        autoencoder: Trained autoencoder model
        data: Input data to reconstruct
        n: Number of examples to show
        save_path: Path to save the visualization
    """
    # Normalize and reshape the data
    data = data.astype('float32') / 255.0
    data = np.expand_dims(data, axis=-1)
    
    # Choose random samples
    indices = np.random.choice(len(data), n, replace=False)
    sample_data = data[indices]
    
    # Get reconstructions
    reconstructions = autoencoder.predict(sample_data)
    
    # Plot original vs. reconstruction
    plt.figure(figsize=(20, 4))
    
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_data[i].reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_latent_space_interpolation(decoder, save_path="latent_interpolation.png"):
    """
    Visualize interpolation in the latent space.
    
    Args:
        decoder: Trained decoder model
        save_path: Path to save the visualization
    """
    # For 2D latent space
    if decoder.input_shape[1] == 2:
        # Create a grid in latent space
        n = 20  # Number of points in each dimension
        grid_x = np.linspace(-3, 3, n)
        grid_y = np.linspace(-3, 3, n)
        
        figure = np.zeros((28 * n, 28 * n))
        
        # Decode each point in the grid
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(28, 28)
                figure[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = digit
        
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gray')
        plt.title('Latent Space Interpolation (2D Grid)')
        plt.axis('off')
        
    # For higher-dimensional latent space, do linear interpolation between two points
    else:
        # Generate two random points in latent space
        z_dim = decoder.input_shape[1]
        z1 = np.random.normal(0, 1, size=(1, z_dim))
        z2 = np.random.normal(0, 1, size=(1, z_dim))
        
        # Decode the endpoints to see what digits they represent
        x_decoded_1 = decoder.predict(z1)
        x_decoded_2 = decoder.predict(z2)
        
        # Create interpolations
        n_interp = 10
        alpha_values = np.linspace(0, 1, n_interp)
        
        plt.figure(figsize=(15, 3))
        
        # Plot the starting point
        ax = plt.subplot(1, n_interp, 1)
        plt.imshow(x_decoded_1.reshape(28, 28), cmap='gray')
        plt.title('Start')
        plt.axis('off')
        
        # Plot interpolations
        for i in range(1, n_interp-1):
            alpha = alpha_values[i]
            z_interp = (1-alpha) * z1 + alpha * z2
            x_decoded_interp = decoder.predict(z_interp)
            
            ax = plt.subplot(1, n_interp, i+1)
            plt.imshow(x_decoded_interp.reshape(28, 28), cmap='gray')
            plt.title(f'α={alpha:.1f}')
            plt.axis('off')
        
        # Plot the ending point
        ax = plt.subplot(1, n_interp, n_interp)
        plt.imshow(x_decoded_2.reshape(28, 28), cmap='gray')
        plt.title('End')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def compare_dimensionality_reduction_techniques(data, labels, save_path="dim_reduction_comparison.png"):
    """
    Compare different dimensionality reduction techniques: PCA, t-SNE, and Autoencoder.
    
    Args:
        data: Input data
        labels: Labels for the input data
        save_path: Path to save the visualization
    """
    # Normalize and reshape the data
    normalized_data = data.astype('float32') / 255.0
    flattened_data = normalized_data.reshape(len(data), -1)
    
    plt.figure(figsize=(15, 5))
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flattened_data)
    
    plt.subplot(1, 3, 1)
    plt.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=labels, cmap='tab10', alpha=0.8, s=10
    )
    plt.colorbar()
    plt.grid(True, alpha=0.3)
    plt.title('PCA (Linear)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(flattened_data)
    
    plt.subplot(1, 3, 2)
    plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1],
        c=labels, cmap='tab10', alpha=0.8, s=10
    )
    plt.colorbar()
    plt.grid(True, alpha=0.3)
    plt.title('t-SNE (Non-linear)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # 3. Autoencoder latent space
    # Create and train a simple autoencoder
    autoencoder = SimpleAutoencoder(latent_dim=2)
    encoder, decoder, ae_model = autoencoder.build_models(input_shape=(28, 28, 1))
    
    # Train the autoencoder briefly to get a basic representation
    # In practice, you'd train longer, but this is for demonstration purposes
    autoencoder.train(data, epochs=5, batch_size=128)
    
    # Encode the data
    encoded_data = encoder.predict(np.expand_dims(normalized_data, axis=-1))
    
    plt.subplot(1, 3, 3)
    plt.scatter(
        encoded_data[:, 0], encoded_data[:, 1],
        c=labels, cmap='tab10', alpha=0.8, s=10
    )
    plt.colorbar()
    plt.grid(True, alpha=0.3)
    plt.title('Autoencoder (Non-linear)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def demonstrate_generative_capabilities(decoder, n_samples=10, save_path="generated_samples.png"):
    """
    Demonstrate the generative capabilities of the decoder by sampling from the latent space.
    
    Args:
        decoder: Trained decoder model
        n_samples: Number of samples to generate
        save_path: Path to save the visualization
    """
    # Sample random points from the latent space
    latent_dim = decoder.input_shape[1]
    
    # Option 1: Sample from standard normal distribution
    z_normal = np.random.normal(0, 1, size=(n_samples, latent_dim))
    
    # Option 2: Sample from uniform distribution
    z_uniform = np.random.uniform(-3, 3, size=(n_samples, latent_dim))
    
    # Decode the samples
    generated_normal = decoder.predict(z_normal)
    generated_uniform = decoder.predict(z_uniform)
    
    # Plot the generated samples
    plt.figure(figsize=(20, 4))
    
    # Plot samples from normal distribution
    for i in range(n_samples):
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(generated_normal[i].reshape(28, 28), cmap='gray')
        plt.title('Normal Sampling')
        plt.axis('off')
    
    # Plot samples from uniform distribution
    for i in range(n_samples):
        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(generated_uniform[i].reshape(28, 28), cmap='gray')
        plt.title('Uniform Sampling')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Take a subset of the data for faster execution
    n_samples = 5000
    x_train_subset = x_train[:n_samples]
    y_train_subset = y_train[:n_samples]
    x_test_subset = x_test[:1000]
    y_test_subset = y_test[:1000]
    
    # Create and train the autoencoder
    print("Building and training the autoencoder...")
    autoencoder = SimpleAutoencoder(latent_dim=2)
    encoder, decoder, ae_model = autoencoder.build_models()
    history = autoencoder.train(x_train_subset, epochs=10)
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("autoencoder_training.png")
    plt.show()
    
    # Visualize the latent space
    print("Visualizing the latent space...")
    visualize_latent_space(encoder, x_test_subset, y_test_subset)
    
    # Visualize digit reconstruction
    print("Visualizing digit reconstruction...")
    visualize_digit_reconstruction(ae_model, x_test_subset)
    
    # Visualize latent space interpolation
    print("Visualizing latent space interpolation...")
    visualize_latent_space_interpolation(decoder)
    
    # Compare dimensionality reduction techniques
    print("Comparing dimensionality reduction techniques...")
    compare_dimensionality_reduction_techniques(x_test_subset, y_test_subset)
    
    # Demonstrate generative capabilities
    print("Demonstrating generative capabilities...")
    demonstrate_generative_capabilities(decoder)
    
    print("Visualization complete! Check the output images:")
    print("- latent_space_2d.png")
    print("- digit_reconstruction.png")
    print("- latent_interpolation.png")
    print("- dim_reduction_comparison.png")
    print("- generated_samples.png")
