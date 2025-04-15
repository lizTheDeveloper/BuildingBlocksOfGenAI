# Generative Adversarial Networks (GANs): Theory

## Introduction to GANs (15 minutes)

- GANs: introduced by Ian Goodfellow in 2014
- Two-network architecture: generator and discriminator
- Novel approach: networks "compete" in a minimax game
- Revolutionized image generation quality

## The GAN Game

- Generator: Creates fake data to fool the discriminator
- Discriminator: Tries to distinguish real data from fakes
- Zero-sum game with a Nash equilibrium 
- Training process is like a forger vs. detective competition

## GAN Architecture in Detail

```python
# Generator network
def build_generator(latent_dim):
    model = keras.Sequential([
        # Foundation for 7x7 feature maps
        layers.Dense(7*7*128, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        
        # Upsampling layers
        layers.Conv2DTranspose(128, kernel_size=4, strides=1, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        # Output layer with tanh activation
        layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])
    return model

# Discriminator network
def build_discriminator(input_shape):
    model = keras.Sequential([
        # Convolutional layers
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=input_shape),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        # Classification output
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

## GAN Training Dynamics

- Non-convergent training: oscillation rather than minimization
- Key challenges:
  - Mode collapse: generator produces limited variety
  - Vanishing gradients: when discriminator becomes too good
  - Training instability: difficult to find equilibrium

## GAN Loss Functions

```python
# Discriminator loss function
def discriminator_loss(real_output, fake_output):
    # Real examples should be classified as 1
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    # Fake examples should be classified as 0
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
    # Total loss is the sum
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss function
def generator_loss(fake_output):
    # Generator wants fake examples to be classified as 1
    return tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)
```

## GAN Variants and Applications

- Conditional GANs: incorporate class labels
- CycleGAN: unpaired image-to-image translation
- StyleGAN: controlled image generation with style mixing
- Applications:
  - Photorealistic image generation
  - Image-to-image translation
  - Super-resolution
  - Data augmentation
  - Art generation

## Practical Tips for GAN Training

- Use label smoothing (0.9 instead of 1.0 for real labels)
- Apply instance noise to inputs
- Implement spectral normalization
- Try alternative losses (Wasserstein, LSGAN)
- Schedule learning rates carefully
- Balance generator and discriminator updates

## Transition to Hands-on Exercise

- Questions before we implement our own GAN?
- We'll focus on MNIST digit generation
- You'll implement the generator and discriminator networks
- Goal: Generate convincing handwritten digits
