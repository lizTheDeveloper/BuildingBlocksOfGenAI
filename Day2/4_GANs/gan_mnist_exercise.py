"""
GAN MNIST Exercise
Building Blocks of Generative AI Course - Day 2

This exercise helps students implement and train a GAN to generate MNIST digits.
Students will fill in the missing parts of the generator and discriminator networks,
along with the training loop.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
BATCH_SIZE = 128
EPOCHS = 20
LATENT_DIM = 100
IMAGE_SHAPE = (28, 28, 1)

class MNISTGAN:
    def __init__(self, latent_dim=LATENT_DIM):
        self.latent_dim = latent_dim
        self.image_shape = IMAGE_SHAPE
        self.generator = None
        self.discriminator = None
        self.gan = None
        
    def build_generator(self):
        """
        Build the generator model.
        
        The generator takes a random noise vector and transforms it into an image.
        
        Returns:
            The generator model
        """
        # TODO: Implement the generator network
        
        # Input is a random noise vector from the latent space
        noise = keras.Input(shape=(self.latent_dim,))
        
        # TODO: Add dense layers to transform the noise
        # Hint: You'll need to reshape to prepare for convolutional layers
        
        # TODO: Add upsampling layers (Conv2DTranspose) to generate the image
        # Hint: The final output should have the same shape as MNIST images: (28, 28, 1)
        
        # TODO: Add the final layer with tanh activation for pixel values in [-1, 1]
        # Remember to scale the output appropriately
        
        # TODO: Create the generator model
        
        return self.generator
    
    def build_discriminator(self):
        """
        Build the discriminator model.
        
        The discriminator takes an image and outputs the probability that the image is real.
        
        Returns:
            The discriminator model
        """
        # TODO: Implement the discriminator network
        
        # Input is an image
        image = keras.Input(shape=self.image_shape)
        
        # TODO: Add convolutional layers with downsampling
        # Hint: Use Conv2D with strides=2 for downsampling
        
        # TODO: Flatten the features
        
        # TODO: Add dense layers
        
        # TODO: Add the final layer with sigmoid activation for binary classification (real/fake)
        
        # TODO: Create and compile the discriminator model
        # Use binary crossentropy loss and an appropriate optimizer
        
        return self.discriminator
    
    def build_gan(self):
        """
        Build the GAN by connecting the generator and discriminator.
        
        Returns:
            The GAN model
        """
        # Ensure generator and discriminator are built
        if self.generator is None:
            self.build_generator()
        if self.discriminator is None:
            self.build_discriminator()
        
        # TODO: Implement the GAN model
        
        # For training the generator, we freeze the discriminator's weights
        self.discriminator.trainable = False
        
        # GAN input (noise) will produce generated images
        gan_input = keras.Input(shape=(self.latent_dim,))
        
        # TODO: Connect the generator and discriminator
        # 1. Generate images from noise
        # 2. Feed the generated images to the discriminator
        
        # TODO: Create and compile the GAN model
        # Use binary crossentropy loss and an appropriate optimizer
        
        return self.gan
    
    def preprocess_data(self, images):
        """
        Preprocess the images for GAN training.
        
        Args:
            images: The input images
            
        Returns:
            Preprocessed images
        """
        # TODO: Implement preprocessing
        # 1. Scale images to [-1, 1]
        # 2. Ensure correct shape for the discriminator
        images = images.astype(np.float32)
        images = images / 127.5 - 1  # Scale to [-1, 1]
        images = np.expand_dims(images, axis=-1)
        return images
    
    def train(self, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, save_interval=5):
        """
        Train the GAN model.
        
        Args:
            x_train: Training data (images)
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_interval: Interval for saving sample images
            
        Returns:
            Training history (discriminator and generator losses)
        """
        # Preprocess the data
        x_train = self.preprocess_data(x_train)
        
        # Create arrays to store loss history
        d_loss_history = []
        g_loss_history = []
        
        # Start training
        for epoch in range(epochs):
            start_time = time.time()
            
            # -------------------------
            # TODO: Implement the training loop
            # 1. Train the discriminator with real and fake images (half real, half fake)
            # 2. Train the generator to fool the discriminator
            # 3. Record losses for plotting
            # -------------------------
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Time: {time.time()-start_time:.2f}s")
            print(f"D Loss: {np.mean(d_loss_history[-batch_size:]):.4f}, G Loss: {np.mean(g_loss_history[-batch_size:]):.4f}")
            
            # Save sample images at specified intervals
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save_images(epoch + 1)
        
        # Return the loss history
        return {'d_loss': d_loss_history, 'g_loss': g_loss_history}
    
    def generate_images(self, num_images=25):
        """
        Generate random images from the trained generator.
        
        Args:
            num_images: Number of images to generate
            
        Returns:
            Generated images
        """
        # Generate random noise
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        
        # Generate images from noise
        generated_images = self.generator.predict(noise)
        
        # Rescale images from [-1, 1] to [0, 1]
        generated_images = (generated_images + 1) / 2.0
        
        return generated_images
    
    def save_images(self, epoch, num_images=25, grid_size=5):
        """
        Generate and save images at a given epoch.
        
        Args:
            epoch: Current epoch number
            num_images: Number of images to generate
            grid_size: Size of the grid to display images
        """
        # Generate images
        generated_images = self.generate_images(num_images)
        
        # Plot images in a grid
        plt.figure(figsize=(10, 10))
        
        for i in range(num_images):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"gan_mnist_epoch_{epoch}.png")
        plt.close()
    
    def plot_loss_history(self, history):
        """
        Plot discriminator and generator loss over training.
        
        Args:
            history: Dictionary with d_loss and g_loss arrays
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history['d_loss'], label='Discriminator Loss')
        plt.plot(history['g_loss'], label='Generator Loss')
        plt.title('GAN Loss During Training')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('gan_loss_history.png')
        plt.show()

# Main execution
if __name__ == "__main__":
    print("Loading MNIST dataset...")
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    
    print("Building and training GAN model...")
    gan_model = MNISTGAN(latent_dim=LATENT_DIM)
    history = gan_model.train(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, save_interval=5)
    
    print("Plotting loss history...")
    gan_model.plot_loss_history(history)
    
    print("Generating final sample of images...")
    gan_model.save_images(epoch=EPOCHS, num_images=25)
    
    print("Exercise complete!")
