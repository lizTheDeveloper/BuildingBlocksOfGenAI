# GAN Exercise: MNIST Generation

## Exercise Overview (5 minutes)

- Build and train a GAN to generate MNIST digits
- Implement both generator and discriminator networks
- Monitor training progress via generated samples
- Analyze results and challenges encountered

## MNIST Dataset Preparation

```python
# Load MNIST dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()

# Preprocess the data
def preprocess_data(images):
    # Scale images to [-1, 1] range for tanh activation
    images = images.astype(np.float32)
    images = images / 127.5 - 1
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    return images

x_train = preprocess_data(x_train)
```

## Generator Architecture Requirements

- Input: Random noise vector from latent space (e.g., 100-dimensional)
- Hidden layers: Dense layer followed by convolutional transpose layers
- Output: 28×28×1 image with tanh activation (pixel values in [-1, 1])
- Upsampling strategy: Use Conv2DTranspose with strides=2
- Key components to include:
  - Batch normalization
  - LeakyReLU activation
  - Proper reshaping between dense and convolutional layers

## Discriminator Architecture Requirements

- Input: 28×28×1 MNIST image
- Hidden layers: Convolutional layers with downsampling
- Output: Single unit with sigmoid activation (probability of real)
- Downsampling strategy: Use Conv2D with strides=2
- Key components to include:
  - Dropout for regularization
  - LeakyReLU activation
  - Final dense layer for classification

## GAN Training Loop Implementation

```python
# Example training loop structure
@tf.function
def train_step(real_images):
    # Generate random noise for the generator
    noise = tf.random.normal([BATCH_SIZE, latent_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)
        
        # Get discriminator outputs
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss
```

## Evaluation and Visualization

- Generate sample images at regular intervals
- Create a grid of generated images for visual inspection
- Plot loss curves for generator and discriminator
- Analyze for mode collapse or training instability

## Exercise Challenges

- Balance generator and discriminator training
- Prevent mode collapse
- Recognize and address vanishing gradients
- Tune hyperparameters for stable training

## Starter Code Distribution

- We'll provide skeleton code with TODOs for you to complete
- Architecture guidelines and high-level structure are provided
- You'll implement:
  - Network architectures for generator and discriminator
  - Loss functions
  - Key parts of the training loop

## Success Metrics

- Generator produces recognizable digits
- No apparent mode collapse (variety of digits)
- Stable training (non-exploding loss curves)
- Improvements in image quality over training
