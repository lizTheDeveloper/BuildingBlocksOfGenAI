"""
GAN MNIST Solution
Building Blocks of Generative AI Course - Day 2

Complete solution for the GAN MNIST exercise. This implements a full GAN to
generate MNIST digits, with proper generator and discriminator networks and training loop.
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
        # Input is a random noise vector from the latent space
        noise = keras.Input(shape=(self.latent_dim,))
        
        # First dense layer to get enough units for reshaping
        x = layers.Dense(7 * 7 * 128)(noise)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Reshape for convolutional layers
        x = layers.Reshape((7, 7, 128))(x)
        
        # Upsampling with Conv2DTranspose layers
        x = layers.Conv2DTranspose(128, kernel_size=4, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Final layer with tanh activation for pixel values in [-1, 1]
        x = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
        
        # Create the generator model
        self.generator = keras.Model(noise, x, name='generator')
        print(self.generator.summary())
        
        return self.generator
    
    def build_discriminator(self):
        """
        Build the discriminator model.
        
        The discriminator takes an image and outputs the probability that the image is real.
        
        Returns:
            The discriminator model
        """
        # Input is an image
        image = keras.Input(shape=self.image_shape)
        
        # Convolutional layers with downsampling
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(image)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Flatten the features
        x = layers.Flatten()(x)
        
        # Dense layers
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Final layer with sigmoid activation for binary classification
        x = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile the discriminator model
        self.discriminator = keras.Model(image, x, name='discriminator')
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        print(self.discriminator.summary())
        
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
        
        # GAN input (noise) will produce generated images
        gan_input = keras.Input(shape=(self.latent_dim,))
        
        # Connect the generator and discriminator
        # 1. Generate images from noise
        generated_images = self.generator(gan_input)
        # 2. Feed the generated images to the discriminator
        
        # For training the generator, we freeze the discriminator's weights
        self.discriminator.trainable = False
        gan_output = self.discriminator(generated_images)
        
        # Create and compile the GAN model
        self.gan = keras.Model(gan_input, gan_output, name='gan')
        self.gan.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
        print(self.gan.summary())
        
        return self.gan
    
    def preprocess_data(self, images):
        """
        Preprocess the images for GAN training.
        
        Args:
            images: The input images
            
        Returns:
            Preprocessed images
        """
        # Scale images to [-1, 1]
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
        # Ensure GAN model is built
        if self.gan is None:
            self.build_gan()
        
        # Preprocess the data
        x_train = self.preprocess_data(x_train)
        
        # Calculate the number of batches per epoch
        batch_count = x_train.shape[0] // batch_size
        
        # Create arrays to store loss history
        d_loss_history = []
        g_loss_history = []
        
        # Start training
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training loop for each batch
            for batch_idx in range(batch_count):
                # -----------------------
                # Train the discriminator
                # -----------------------
                
                # Set discriminator to trainable
                self.discriminator.trainable = True
                
                # Select a random batch of real images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_images = x_train[idx]
                
                # Generate a batch of fake images
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_images = self.generator.predict(noise, verbose=0)
                
                # Get labels for real and fake images
                real_labels = np.ones((batch_size, 1)) * 0.9  # Smoothed labels for stability
                fake_labels = np.zeros((batch_size, 1))
                
                # Train on real images
                d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
                
                # Train on fake images
                d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
                
                # Average discriminator loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_history.append(d_loss[0])
                
                # -----------------------
                # Train the generator
                # -----------------------
                
                # Set discriminator to non-trainable
                self.discriminator.trainable = False
                
                # Generate new noise for the generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # Generator wants the discriminator to label fake images as real
                valid_labels = np.ones((batch_size, 1))
                
                # Train the generator
                g_loss = self.gan.train_on_batch(noise, valid_labels)
                g_loss_history.append(g_loss)
                
                # Print progress
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{batch_count}")
                    print(f"D Loss: {d_loss[0]:.4f}, Acc: {100*d_loss[1]:.2f}%, G Loss: {g_loss:.4f}")
            
            # Print epoch results
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Time: {elapsed_time:.2f}s")
            print(f"D Loss: {np.mean(d_loss_history[-batch_count:]):.4f}, G Loss: {np.mean(g_loss_history[-batch_count:]):.4f}")
            
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
