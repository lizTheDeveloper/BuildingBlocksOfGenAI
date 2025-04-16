"""
Variational Autoencoder (VAE) for Image Generation and Reconstruction
Building Blocks of Generative AI Course - Day 1

This script implements a Variational Autoencoder for generating and reconstructing images
using the MNIST dataset as an example.
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
EPOCHS = 20

class VAEModelMNIST(keras.Model):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAEModelMNIST, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.build_encoder()
        self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def build_encoder(self):
        # Define encoder inputs
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        
        # Convolutional layers
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        
        # Mean and variance for the latent space
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Sampling layer
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()
    
    def build_decoder(self):
        # Define decoder inputs (the latent space)
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="z_sampling")
        
        # Dense layers to reconstruct the feature maps
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        
        # Transposed convolutions to upsample
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        
        # Output layer with sigmoid activation to get pixel values between 0 and 1
        decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        
        # Create decoder model
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

    def call(self, inputs):
        # Get outputs from encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # Get reconstruction from decoder
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Calculate losses
            # Flatten both input and reconstruction
            x_flat = tf.reshape(data, [-1, 784])  # 28*28 = 784
            reconstruction_flat = tf.reshape(reconstruction, [-1, 784])
            
            # Reconstruction loss (pixel-wise binary crossentropy)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(x_flat, reconstruction_flat)
            )
            reconstruction_loss *= 784  # Scale by image dimensions
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            
            total_loss = reconstruction_loss + kl_loss
        
        # Calculate gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            
        # Forward pass
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Calculate losses
        # Flatten both input and reconstruction
        x_flat = tf.reshape(data, [-1, 784])  # 28*28 = 784
        reconstruction_flat = tf.reshape(reconstruction, [-1, 784])
        
        # Reconstruction loss (pixel-wise binary crossentropy)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(x_flat, reconstruction_flat)
        )
        reconstruction_loss *= 784  # Scale by image dimensions
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        
        total_loss = reconstruction_loss + kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def train(self, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
        # Reshape data to include channel dimension and normalize
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255
        
        # Compile the model with a dummy loss since real loss is handled in train_step
        self.compile(optimizer=keras.optimizers.Adam())
        
        # Train the model
        history = self.fit(
            x_train,
            x_train,  # Provide same data as input and target
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
        )
        
        return history

def plot_latent_space(vae_model, n=30, figsize=15):
    """
    Plot images decoded from a grid of points in the latent space.
    """
    # Display a n*n grid of decoded digits from the 2D latent space
    figure = np.zeros((28 * n, 28 * n))
    
    # Create a grid of points in the latent space
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
    """
    # Choose random samples from the data
    np.random.seed(42)
    sample_indices = np.random.choice(len(data), n, replace=False)
    sample_images = data[sample_indices]
    
    # Reshape and normalize
    sample_images = np.expand_dims(sample_images, -1).astype("float32") / 255
    
    # Predict reconstructions
    reconstructions = vae_model(sample_images, training=False)
    
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
        plt.imshow(reconstructions[i].numpy().reshape(28, 28), cmap="Greys_r")
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("vae_reconstructions.png")
    plt.show()

def plot_random_generated_images(vae_model, n=10):
    """
    Generate and plot random images by sampling from the latent space.
    """
    # Random points from the latent space
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
    vae_model = VAEModelMNIST(latent_dim=LATENT_DIM)
    history = vae_model.train(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("Plotting latent space visualization...")
    plot_latent_space(vae_model, n=20, figsize=12)
    
    print("Plotting image reconstructions...")
    plot_reconstructions(vae_model, x_test, n=10)
    
    print("Generating random images from latent space...")
    plot_random_generated_images(vae_model, n=10)
    
    print("Exercise complete!")
