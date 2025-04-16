"""
VAE Exercise - Fashion MNIST Image Generation and Reconstruction
Building Blocks of Generative AI Course - Day 1

This exercise guides students through implementing a Variational Autoencoder (VAE)
for image generation and reconstruction using the Fashion MNIST dataset.

Students will fill in the missing code to build the encoder and decoder components,
implement the sampling function, and define the VAE loss function.
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

# Fashion MNIST class names for reference
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

class FashionVAE(keras.Model):
    """
    Variational Autoencoder (VAE) for Fashion MNIST dataset
    
    This class inherits from keras.Model to enable custom training logic
    and implements the VAE architecture with encoder, decoder, and custom loss function.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super(FashionVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.build_encoder()
        self.build_decoder()
        
        # Metrics trackers for monitoring training
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        """Define metrics to track during training"""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def build_encoder(self):
        """
        Build the encoder network
        
        The encoder maps input images to latent space parameters (mean and log variance)
        and samples from this distribution to produce latent vectors.
        """
        # TODO: Define encoder inputs for Fashion MNIST (28x28 grayscale images)
        # Create a Keras Input layer with the appropriate shape
        
        # TODO: Implement 2-3 convolutional layers
        # Each layer should increase the number of filters (e.g., 32, 64, 128)
        # Use strides=2 in at least one layer to downsample the spatial dimensions
        # Don't forget to add activation functions
        
        # TODO: Flatten the convolutional features
        
        # TODO: Add a dense layer to process the flattened features
        # This layer should have fewer units than the flattened layer (e.g., 16 units)
        
        # TODO: Create two separate dense layers for the mean and log variance
        # Both should output vectors of size self.latent_dim
        
        # TODO: Implement the sampling layer using a Lambda layer
        # 1. Define a sampling function that takes z_mean and z_log_var as inputs
        # 2. Generate random epsilon values from a normal distribution
        # 3. Apply the reparameterization trick: z = z_mean + exp(0.5 * z_log_var) * epsilon
        # 4. Return the sampled points z
        
        # TODO: Create the encoder model
        # The model should take the encoder_inputs and output [z_mean, z_log_var, z]
        
        # Print model summary
        print("Encoder Summary:")
        self.encoder.summary()
    
    def build_decoder(self):
        """
        Build the decoder network
        
        The decoder maps latent vectors back to the image space,
        reconstructing fashion images from their latent representation.
        """
        # TODO: Define decoder inputs (latent vectors of dimension self.latent_dim)
        
        # TODO: Create a dense layer to transform the latent vector
        # Calculate what dimensions you need based on your planned decoder architecture
        # The output should match the dimensions needed for reshaping
        # (typically the dimensions at the bottleneck of the encoder)
        
        # TODO: Reshape the dense layer output to 3D feature maps
        # Example shape might be (7, 7, 64) or similar depending on your architecture
        
        # TODO: Add 2-3 transposed convolution layers (Conv2DTranspose)
        # Start with higher number of filters and decrease
        # Use strides=2 to upsample the spatial dimensions
        # Your final feature map should be upsampled to 28x28 (Fashion MNIST dimensions)
        
        # TODO: Add a final Conv2D layer with sigmoid activation
        # This layer should have 1 filter (for grayscale) and output values in [0,1]
        
        # TODO: Create the decoder model
        # The model should take latent_inputs and output the reconstructed images
        
        # Print model summary
        print("Decoder Summary:")
        self.decoder.summary()
    
    def call(self, inputs):
        """Forward pass through the VAE model"""
        # Get outputs from encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # Get reconstruction from decoder
        reconstruction = self.decoder(z)
        return reconstruction
    
    def train_step(self, data):
        """
        Custom training step for the VAE
        
        Performs forward pass, calculates losses, and updates model weights.
        """
        if isinstance(data, tuple):
            data = data[0]
        
        with tf.GradientTape() as tape:
            # TODO: Pass the input data through the encoder
            # This should return z_mean, z_log_var, and z (the sampled latent vector)
            
            # TODO: Pass the sampled latent vector z through the decoder
            # This should return the reconstructed images
            
            # TODO: Prepare for loss calculation
            # 1. Flatten both input and reconstruction for pixel-wise comparison
            # 2. Original shape: [batch_size, 28, 28, 1]
            # 3. Flattened shape: [batch_size, 784]
            
            # TODO: Calculate reconstruction loss using binary_crossentropy
            # 1. Compare the flattened input and reconstruction pixel by pixel
            # 2. Use tf.reduce_mean to get the average loss across the batch
            # 3. Multiply by 784 (28*28) to scale the loss appropriately
            
            # TODO: Calculate KL divergence loss
            # 1. Implement the KL divergence formula:
            #    -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # 2. This penalizes distributions that differ from the standard normal distribution
            
            # TODO: Calculate the total loss by adding reconstruction and KL divergence losses
        
        # Calculate gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def test_step(self, data):
        """
        Custom test step for the VAE
        
        Calculates validation metrics without weight updates.
        """
        if isinstance(data, tuple):
            data = data[0]
        
        # TODO: Implement a test step that mirrors the train_step but without gradient tracking
        # This function should:
        # 1. Run the encoder to get z_mean, z_log_var, and z
        # 2. Run the decoder to get reconstructions
        # 3. Calculate reconstruction loss (similar to train_step)
        # 4. Calculate KL divergence loss (similar to train_step)
        # 5. Calculate total loss
        # 6. Update the metrics
        # 
        # Note: Unlike train_step, this function does not need a GradientTape
        # or any weight updates, as it's only for evaluation.
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train(self, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """
        Train the VAE model
        
        Args:
            x_train: Training data (images)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Reshape and normalize the data
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255
        
        # Compile the model with a dummy loss since real loss is handled in train_step
        self.compile(optimizer=keras.optimizers.Adam())
        
        # Train the model
        history = self.fit(
            x_train,
            x_train,  # Input is the same as target for autoencoders
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
        )
        
        return history

def plot_latent_space(vae_model, n=30, figsize=15, save_path=None):
    """
    Visualize the latent space by plotting a grid of decoded images
    
    Args:
        vae_model: Trained VAE model
        n: Number of points in each latent dimension
        figsize: Size of the figure
        save_path: Path to save the visualization (optional)
    """
    # Create a grid of latent space coordinates
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
    
    # Plot the decoded images
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap="Greys_r")
    plt.title("Fashion MNIST Latent Space")
    plt.axis("off")
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_reconstructions(vae_model, data, n=10, save_path=None):
    """
    Plot original and reconstructed fashion items side by side
    
    Args:
        vae_model: Trained VAE model
        data: Test data (images)
        n: Number of images to display
        save_path: Path to save the visualization (optional)
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
        plt.title(f"Original: {FASHION_MNIST_CLASSES[data[sample_indices][i][1]]}" 
                  if len(data[0]) > 1 else "Original")
        plt.axis("off")
        
        # Reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].numpy().reshape(28, 28), cmap="Greys_r")
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_random_generated_images(vae_model, n=10, save_path=None):
    """
    Generate and plot random fashion items by sampling from the latent space
    
    Args:
        vae_model: Trained VAE model
        n: Number of images to generate
        save_path: Path to save the visualization (optional)
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
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_latent_space_with_labels(vae_model, data, labels, n_samples=1000, figsize=(10, 8), save_path=None):
    """
    Visualize how different fashion categories are distributed in the latent space
    
    Args:
        vae_model: Trained VAE model
        data: Test data (images)
        labels: Test data labels
        n_samples: Number of samples to plot
        figsize: Size of the figure
        save_path: Path to save the visualization (optional)
    """
    # Choose random samples
    np.random.seed(42)
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        data_subset = data[indices]
        labels_subset = labels[indices]
    else:
        data_subset = data
        labels_subset = labels
    
    # Reshape and normalize
    data_subset = np.expand_dims(data_subset, -1).astype("float32") / 255
    
    # Get latent space representations
    z_mean, z_log_var, z = vae_model.encoder.predict(data_subset)
    
    # Create scatter plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels_subset, 
                         cmap='tab10', alpha=0.8, s=10)
    
    # Add legend with class names
    plt.colorbar(ticks=range(10))
    plt.title("Latent Space Visualization of Fashion MNIST Categories")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.grid(alpha=0.3)
    
    # Add class names as annotations
    for i, class_name in enumerate(FASHION_MNIST_CLASSES):
        # Find mean position for each class
        idx = labels_subset == i
        if np.any(idx):
            x = np.mean(z_mean[idx, 0])
            y = np.mean(z_mean[idx, 1])
            plt.annotate(class_name, (x, y), fontsize=12, 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    print("Building and training VAE model...")
    vae_model = FashionVAE(latent_dim=LATENT_DIM)
    history = vae_model.train(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("Plotting latent space visualization...")
    plot_latent_space(vae_model, n=20, figsize=12, save_path="fashion_vae_latent_space.png")
    
    print("Plotting image reconstructions...")
    plot_reconstructions(vae_model, x_test, n=10, save_path="fashion_vae_reconstructions.png")
    
    print("Generating random images from latent space...")
    plot_random_generated_images(vae_model, n=10, save_path="fashion_vae_generated_images.png")
    
    print("Visualizing latent space with class labels...")
    plot_latent_space_with_labels(vae_model, x_test, y_test, save_path="fashion_vae_latent_clusters.png")
    
    print("Exercise complete!")