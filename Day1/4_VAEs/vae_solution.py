"""
VAE Solution - MNIST Image Generation and Reconstruction
Building Blocks of Generative AI Course - Day 1

This is the solution to the VAE exercise, implementing a Variational Autoencoder
for image generation and reconstruction using the MNIST dataset.
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
        # Define encoder inputs (28x28 grayscale images)
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        
        # Add convolutional layers
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        
        # Dense layers for the latent space parameters
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        # Sampling function using the reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Sampling layer
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()
        
        return self.encoder
    
    def build_decoder(self):
        """
        Build the decoder network for the VAE.
        
        The decoder takes a latent vector and outputs a reconstructed image.
        
        Returns:
            The decoder model
        """
        # Define decoder inputs (latent vectors)
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="z_sampling")
        
        # Dense layers to transform the latent vector
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        
        # Convolutional transpose layers to upsample and reconstruct the image
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        
        # Final layer with sigmoid activation to get pixel values between 0 and 1
        decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        
        # Create decoder model
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()
        
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
        
        # Define VAE inputs
        inputs = keras.Input(shape=(28, 28, 1))
        
        # Connect encoder and decoder
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        
        # Define the VAE loss function
        def vae_loss(inputs, reconstruction):
            # Reconstruction loss (binary crossentropy for normalized pixel values)
            reconstruction_loss = keras.losses.binary_crossentropy(
                K.flatten(inputs), K.flatten(reconstruction)
            )
            reconstruction_loss *= 28 * 28  # Scale by image dimensions
            
            # KL divergence loss (analytical formula for Gaussian case)
            kl_loss = -0.5 * K.sum(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
            )
            
            # Total loss
            return K.mean(reconstruction_loss + kl_loss)
        
        # Create and compile VAE model
        self.vae = keras.Model(inputs, reconstruction, name="vae")
        self.vae.add_loss(vae_loss(inputs, reconstruction))
        self.vae.compile(optimizer=keras.optimizers.Adam())
        
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

def plot_digit_classes_in_latent_space(vae_model, data, labels, figsize=12):
    """
    Plot the digit classes in the latent space to visualize clustering.
    
    Args:
        vae_model: The trained VAE model
        data: Test data (images)
        labels: Test labels (digit classes)
        figsize: Size of the figure
    """
    # Reshape and normalize
    data = np.expand_dims(data, -1).astype("float32") / 255
    
    # Get the latent space representation
    z_mean, _, _ = vae_model.encoder.predict(data)
    
    # Create a scatter plot colored by digit class
    plt.figure(figsize=(figsize, figsize))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, 
                         cmap='tab10', alpha=0.8, s=10)
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Digits")
    plt.add_artist(legend1)
    
    plt.title('Latent Space Representation by Digit Class')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.grid(True, alpha=0.3)
    plt.savefig("vae_latent_clusters.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print("Building and training VAE model...")
    vae_model = VAEMnist(latent_dim=LATENT_DIM)
    history = vae_model.train(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Plot the training history
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('VAE Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("vae_training_history.png")
    plt.show()
    
    print("Plotting latent space visualization...")
    plot_latent_space(vae_model, n=20, figsize=12)
    
    print("Plotting image reconstructions...")
    plot_reconstructions(vae_model, x_test, n=10)
    
    print("Generating random images from latent space...")
    plot_random_generated_images(vae_model, n=10)
    
    print("Plotting digit classes in latent space...")
    plot_digit_classes_in_latent_space(vae_model, x_test, y_test)
    
    print("Exercise complete!")
