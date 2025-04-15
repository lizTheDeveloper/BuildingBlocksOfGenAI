"""
VAE Exercise - MNIST Image Generation and Reconstruction
Building Blocks of Generative AI Course - Day 1

This exercise guides students through implementing a Variational Autoencoder (VAE)
for image generation and reconstruction using the MNIST dataset.

Students will fill in the missing code to build the encoder and decoder networks
and implement the VAE loss function.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
LATENT_DIM = 2  # Dimensionality of the latent space
BATCH_SIZE = 128
EPOCHS = 10

class VAEMnist:
    def __init__(self, latent_dim=LATENT_DIM):
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.vae = None
    
    def build_encoder(self):
        """
        Build the encoder network for the VAE.
        
        The encoder takes an input image and outputs the mean and log variance
        of the latent distribution, as well as a sampled latent vector.
        
        Returns:
            The encoder model
        """
        # TODO: Implement the encoder network
        
        # Define encoder inputs (28x28 grayscale images)
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        
        # TODO: Add convolutional layers
        # Hint: Start with Conv2D layers with increasing filters, followed by
        # downsampling (using strides=2) and then flatten the output
        
        # TODO: Add dense layers for the latent space parameters
        # You need to create:
        # 1. z_mean - Dense layer for the mean of the latent distribution
        # 2. z_log_var - Dense layer for the log variance of the latent distribution
        
        # TODO: Implement the sampling function to sample from the latent distribution
        # Hint: Use the reparameterization trick to make the sampling differentiable
        
        # TODO: Create the encoder model
        
        return self.encoder
    
    def build_decoder(self):
        """
        Build the decoder network for the VAE.
        
        The decoder takes a latent vector and outputs a reconstructed image.
        
        Returns:
            The decoder model
        """
        # TODO: Implement the decoder network
        
        # Define decoder inputs (latent vectors)
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        
        # TODO: Add dense layers to transform the latent vector
        # Hint: You need to transform the latent vector to match the dimensions
        # needed for the first convolutional transpose layer
        
        # TODO: Add convolutional transpose layers to upsample and reconstruct the image
        # Hint: Use Conv2DTranspose layers with appropriate strides to upsample
        
        # TODO: Add the final layer with sigmoid activation
        # The output should be the same shape as the input images: (28, 28, 1)
        
        # TODO: Create the decoder model
        
        return self.decoder
    
    def build_vae(self):
        """
        Build the complete VAE by connecting the encoder and decoder.
        
        Also defines the VAE loss function that combines reconstruction loss
        and KL divergence loss.
        
        Returns:
            The complete VAE model
        """
        # Ensure encoder and decoder are built
        if self.encoder is None:
            self.build_encoder()
        if self.decoder is None:
            self.build_decoder()
        
        # TODO: Implement the VAE model
        
        # Define VAE inputs
        inputs = keras.Input(shape=(28, 28, 1))
        
        # TODO: Connect encoder and decoder
        # 1. Get z_mean, z_log_var, and z from the encoder
        # 2. Feed z into the decoder to get the reconstructed image
        
        # TODO: Define the VAE loss function
        # The loss should combine:
        # 1. Reconstruction loss (how well the image is reconstructed)
        # 2. KL divergence loss (to ensure the latent space has desired properties)
        
        # TODO: Create and compile the VAE model
        
        return self.vae
    
    def train(self, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """
        Train the VAE model.
        
        Args:
            x_train: Training data (images)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Ensure the VAE is built
        if self.vae is None:
            self.build_vae()
        
        # Reshape and normalize the data
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255
        
        # Train the model
        history = self.vae.fit(
            x_train,
            x_train,  # Input is the same as target for autoencoders
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
        )
        
        return history

def plot_latent_space(vae_model, n=30, figsize=15):
    """
    Plot images decoded from a grid of points in the latent space.
    
    Args:
        vae_model: The trained VAE model
        n: Number of grid points in each dimension
        figsize: Size of the figure
    """
    # Create a grid of points in the latent space
    figure = np.zeros((28 * n, 28 * n))
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
    # Decode each point in the grid
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae_model.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(28, 28)
            figure[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = digit
    
    # Plot the grid of decoded images
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap="Greys_r")
    plt.title("Latent Space Visualization")
    plt.axis("off")
    plt.savefig("vae_latent_space.png")
    plt.show()

def plot_reconstructions(vae_model, data, n=10):
    """
    Plot original and reconstructed images side by side.
    
    Args:
        vae_model: The trained VAE model
        data: Test data (images)
        n: Number of images to reconstruct
    """
    # Choose random samples from the data
    np.random.seed(42)
    sample_indices = np.random.choice(len(data), n, replace=False)
    sample_images = data[sample_indices]
    
    # Reshape and normalize
    sample_images = np.expand_dims(sample_images, -1).astype("float32") / 255
    
    # Predict reconstructions
    reconstructions = vae_model.vae.predict(sample_images)
    
    # Plot original vs reconstruction
    plt.figure(figsize=(20, 4))
    
    for i in range(n):
        # Original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap="Greys_r")
        plt.title("Original")
        plt.axis("off")
        
        # Reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap="Greys_r")
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("vae_reconstructions.png")
    plt.show()

def plot_random_generated_images(vae_model, n=10):
    """
    Generate and plot random images by sampling from the latent space.
    
    Args:
        vae_model: The trained VAE model
        n: Number of images to generate
    """
    # Sample random points from the latent space
    z_samples = np.random.normal(size=(n, vae_model.latent_dim))
    
    # Generate images by decoding the random points
    generated_images = vae_model.decoder.predict(z_samples)
    
    # Plot the generated images
    plt.figure(figsize=(20, 2))
    
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap="Greys_r")
        plt.title(f"Generated {i+1}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("vae_generated_images.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Loading MNIST dataset...")
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    
    print("Building and training VAE model...")
    vae_model = VAEMnist(latent_dim=LATENT_DIM)
    history = vae_model.train(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("Plotting latent space visualization...")
    plot_latent_space(vae_model, n=20, figsize=12)
    
    print("Plotting image reconstructions...")
    plot_reconstructions(vae_model, x_test, n=10)
    
    print("Generating random images from latent space...")
    plot_random_generated_images(vae_model, n=10)
    
    print("Exercise complete!")
