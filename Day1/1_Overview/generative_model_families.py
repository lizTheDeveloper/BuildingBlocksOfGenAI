"""
Generative Model Families
Building Blocks of Generative AI Course - Day 1

This script visualizes the different families of generative models including:
1. Autoregressive Models
2. Variational Autoencoders (VAEs)
3. Generative Adversarial Networks (GANs)
4. Diffusion Models
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

def visualize_autoregressive_model():
    """Visualize a simple autoregressive model on time series data"""
    # Create a 1D time series
    t = np.linspace(0, 4*np.pi, 100)
    signal = np.sin(t) + 0.2 * np.sin(3*t) + np.random.normal(0, 0.1, size=len(t))
    
    # Simulate an autoregressive prediction
    ar_signal = np.zeros_like(signal)
    ar_signal[0] = signal[0]
    for i in range(1, len(signal)):
        if i < 3:
            ar_signal[i] = signal[i]
        else:
            # Simple AR(3) model for demonstration
            ar_signal[i] = 0.8 * signal[i-1] + 0.1 * signal[i-2] + 0.05 * signal[i-3] + np.random.normal(0, 0.05)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, 'b-', label='Original Signal')
    plt.plot(t, ar_signal, 'r--', label='Autoregressive Prediction')
    plt.title('Autoregressive Model')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('autoregressive_model.png')
    plt.show()




print("Visualizing Autoregressive Model")
visualize_autoregressive_model()

def visualize_vae_concept():
    """Visualize the concept of a Variational Autoencoder (VAE)"""
    # Create a simple moon dataset
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # Apply a simple PCA to simulate the latent space and reconstruction
    pca = PCA(n_components=2)
    X_latent = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_latent)
    
    # Add Gaussian noise to the latent space to simulate VAE sampling
    X_latent_noisy = X_latent + np.random.normal(0, 0.1, size=X_latent.shape)
    X_gen = pca.inverse_transform(X_latent_noisy)
    
    plt.figure(figsize=(10, 8))
    
    # Plot the original data and latent space
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.7, label='Original Data')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Plot the latent space
    plt.subplot(2, 2, 2)
    plt.scatter(X_latent[:, 0], X_latent[:, 1], c='red', alpha=0.7, label='Latent Space')
    plt.title('Latent Space (Encoding)')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    
    # Plot the reconstructed data
    plt.subplot(2, 2, 3)
    plt.scatter(X_recon[:, 0], X_recon[:, 1], c='green', alpha=0.7, label='Reconstructed')
    plt.title('Reconstructed Data (Decoding)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Plot the generated data (from noisy latent)
    plt.subplot(2, 2, 4)
    plt.scatter(X_gen[:, 0], X_gen[:, 1], c='purple', alpha=0.7, label='Generated')
    plt.title('Generated Data (Sampling)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_concept.png')
    plt.show()

print("\nVisualizing Variational Autoencoder (VAE)")
visualize_vae_concept()

def visualize_gan_concept():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    # 1. Generate a toy real dataset (moons)
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    real_center = X.mean(axis=0)

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.close()  # We'll show it via the animation, so close the extra static figure

    # 3. "Generator" starts far from the real data
    gen_center = np.array([-1.5, 1.5])  # Starting point for the generator's distribution

    # A simple function to get generator samples around gen_center
    def get_generator_samples(center, num=300):
        return center + np.random.normal(0, 0.15, size=(num, 2))

    # 4. "Discriminator" boundary radius (circle) around real data center
    #    We'll let it adjust over iterations to illustrate changes
    initial_radius = 1.5
    final_radius   = np.sqrt(np.mean(np.sum((X - real_center)**2, axis=1))) * 1.2
    radius_values  = np.linspace(initial_radius, final_radius, num=30)

    # 5. Create scatter plots (blank for now), plus a placeholder circle
    real_scatter      = None
    gen_scatter       = None
    discriminator_circle = None

    def init():
        """Initialize the background of the animation."""
        ax.set_title('Toy GAN Animation: Generator vs. Discriminator')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.0, 2.5)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

        # Plot the real data (blue)
        global real_scatter
        real_scatter = ax.scatter([], [], c='blue', alpha=0.7, label='Real Data')

        # Plot the generator data (green)
        global gen_scatter
        gen_scatter = ax.scatter([], [], c='green', alpha=0.7, label='Generated Data')
        
        # Add a legend (we'll do it once)
        ax.legend(loc='upper right')
        return (real_scatter, gen_scatter,)

    def update(frame):
        """Update function called at each frame (epoch)."""
        global gen_center, discriminator_circle
        
        # 1) Update generator center to move fractionally closer to real_center
        #    This is just a toy version of "learning"
        alpha = 0.15  # how fast the generator moves toward real center
        gen_center = gen_center + alpha*(real_center - gen_center)
        
        # 2) Generate new points from generator
        gen_samples = get_generator_samples(gen_center, num=300)
        
        # 3) Update scatter plot data
        real_scatter.set_offsets(X)
        gen_scatter.set_offsets(gen_samples)
        
        # 4) Update the "discriminator" boundary (circle radius)
        radius = radius_values[frame]
        
        # If we already have a circle from previous frame, remove it
        if discriminator_circle is not None:
            discriminator_circle.remove()
        
        # Draw a new circle around the real data center
        discriminator_circle = plt.Circle(real_center, radius, color='red', fill=False, lw=2, 
                                        label='Discriminator Boundary')
        ax.add_patch(discriminator_circle)
        
        return (real_scatter, gen_scatter, discriminator_circle)

    # 6. Create the animation
    anim = FuncAnimation(fig, update, frames=len(radius_values), 
                        init_func=init, interval=400, blit=False, repeat=True)

    # 7. Display in Colab (JS-based animation)
    HTML(anim.to_jshtml())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_diffusion():
    # 1) Create a simple pattern (circle)
    t = np.linspace(0, 1, 100)
    x_pattern = np.sin(2 * np.pi * t)
    y_pattern = np.cos(2 * np.pi * t)
    
    # 2) Setup figure
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.close()  # We'll display via animation, so close the static figure
    
    # Original line (pattern) and "diffused" points
    (line,) = ax.plot([], [], 'b-', linewidth=3, label='Original Pattern')
    scatter = ax.scatter(x_pattern, y_pattern, c='r', alpha=0.5, s=20, label='Diffused Pattern')
    
    # 3) Noise schedule: from 0.0 up to 1.0 across N frames
    noise_levels = np.linspace(0, 1, 50)  # 50 frames for the animation
    
    def init():
        """Initialize background for the animation."""
        ax.set_title('Diffusion Model Concept (Animation)')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        
        # Plot original pattern once
        line.set_data(x_pattern, y_pattern)
        
        ax.legend(loc='lower left')
        return (line, scatter)
    
    def update(frame):
        """Update function called at each frame."""
        noise = noise_levels[frame]
        
        # Add Gaussian noise to the original pattern
        x_noisy = x_pattern + np.random.normal(0, noise, size=len(x_pattern))
        y_noisy = y_pattern + np.random.normal(0, noise, size=len(y_pattern))
        
        # Update the scatter points
        coords = np.column_stack((x_noisy, y_noisy))
        scatter.set_offsets(coords)
        
        return (line, scatter)
    
    # 4) Create the animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(noise_levels),
        init_func=init,
        blit=False,
        interval=250,
        repeat=True
    )
    
    return HTML(anim.to_jshtml())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_snowman_diffusion():
    #--------------------------
    # 1) Define a "snowman" shape in 2D by concatenating three circles
    #--------------------------
    def circle_points(cx, cy, r, n=100):
        """Return x, y arrays for a circle of radius r centered at (cx, cy)."""
        angles = np.linspace(0, 2*np.pi, n)
        x = cx + r * np.cos(angles)
        y = cy + r * np.sin(angles)
        return x, y
    
    # Top circle, middle circle, bottom circle
    top_x, top_y    = circle_points(0, 1.4, 0.3, 100)
    middle_x,mid_y  = circle_points(0, 0.7, 0.4, 100)
    bottom_x,bottom_y= circle_points(0, 0.0, 0.6, 100)
    
    # Combine into single shape arrays
    x_snowman = np.concatenate([top_x, middle_x, bottom_x])
    y_snowman = np.concatenate([top_y, mid_y, bottom_y])
    
    #--------------------------
    # 2) Setup figure
    #--------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.close()  # We'll display via animation, so close the static figure
    
    # Plot the original shape (thin black line)
    ax.plot(x_snowman, y_snowman, 'k-', linewidth=1, label='Snowman (original)')
    
    # Create a second "line" object to show the diffused version
    # Initialize it at the same coordinates, but we’ll overwrite later
    (diffused_line,) = ax.plot(x_snowman, y_snowman, 'r-', linewidth=2, label='Diffused Snowman')
    
    ax.set_title('Snowman Diffusion Model Animation')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.8, 2.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    
    #--------------------------
    # 3) Noise schedule
    #--------------------------
    # We'll add random Gaussian noise from 0.0 up to 1.0 across 50 frames
    noise_levels = np.linspace(0, 1.0, 50)
    
    #--------------------------
    # 4) Animation callbacks
    #--------------------------
    def init():
        # We don’t need to change anything for the background initialization,
        # we already plotted the original shape. 
        return (diffused_line,)
    
    def update(frame):
        noise = noise_levels[frame]
        
        # Add Gaussian noise based on the noise schedule
        x_noisy = x_snowman + np.random.normal(0, noise, size=len(x_snowman))
        y_noisy = y_snowman + np.random.normal(0, noise, size=len(y_snowman))
        
        # Update the diffused line data
        diffused_line.set_data(x_noisy, y_noisy)
        return (diffused_line,)
    
    #--------------------------
    # 5) Create the animation
    #--------------------------
    anim = FuncAnimation(
        fig,
        update,
        frames=len(noise_levels),
        init_func=init,
        interval=300,   # milliseconds per frame
        blit=False,
        repeat=True
    )
    
    #--------------------------
    # 6) Return HTML for inline display in a Colab/Jupyter cell
    #--------------------------
    return HTML(anim.to_jshtml())
import torch
import clip
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#------------------------------------------------------------------------------
# 1) Load CLIP Model and Preprocess
#------------------------------------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Encode our target text prompt: "a snowman"
text_prompt = ["a snowman"]
text_tokens = clip.tokenize(text_prompt).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

#------------------------------------------------------------------------------
# 2) Snowman Shape (2D)
#------------------------------------------------------------------------------
def circle_points(cx, cy, r, n=100):
    """Return x, y arrays for a circle of radius r centered at (cx, cy)."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return x, y

# Top, middle, bottom circles
top_x, top_y      = circle_points(0, 1.4, 0.3)
middle_x, middle_y= circle_points(0, 0.7, 0.4)
bottom_x, bottom_y= circle_points(0, 0.0, 0.6)

# Combine for a single shape
x_snowman = np.concatenate([top_x, middle_x, bottom_x])
y_snowman = np.concatenate([top_y, middle_y, bottom_y])
original_points = np.column_stack((x_snowman, y_snowman))

#------------------------------------------------------------------------------
# 3) Helper: Convert 2D Snowman Points -> PIL Image
#------------------------------------------------------------------------------
def points_to_image(points, img_size=256):
    """
    Renders the shape as a white background with black lines on top
    in a square PIL image of size=img_size x img_size.
    """
    # Shift/scale to fit nicely in the image
    # We'll assume the shape is in ~(-1.5..1.5) range.
    min_x, max_x = -1.5, 1.5
    min_y, max_y = -0.8, 2.0
    
    # Create blank image
    img = Image.new("RGB", (img_size, img_size), color="white")
    draw = ImageDraw.Draw(img)
    
    # Convert normalized coords to pixel coords
    def to_pixel(px, py):
        # Map x from [min_x..max_x] -> [0..img_size]
        nx = (px - min_x) / (max_x - min_x) * img_size
        # Map y from [max_y..min_y] because PIL's y=0 is top
        ny = (max_y - py) / (max_y - min_y) * img_size
        return (nx, ny)
    
    # We'll connect consecutive points as a polyline
    polyline = [to_pixel(pt[0], pt[1]) for pt in points]
    # Also connect the last point to the first
    polyline.append(to_pixel(points[0,0], points[0,1]))
    
    # Draw black line
    draw.line(polyline, fill="black", width=2)
    
    return img

#------------------------------------------------------------------------------
# 4) CLIP Scoring: shape -> image -> CLIP similarity to "a snowman"
#------------------------------------------------------------------------------
def clip_snowman_score(points):
    """
    1) Convert 2D points to an image
    2) Preprocess the image for CLIP
    3) Compute similarity with the text embedding for "a snowman"
    """
    img = points_to_image(points)
    with torch.no_grad():
        # Preprocess and encode
        img_input = preprocess(img).unsqueeze(0).to(device)
        image_embedding = model.encode_image(img_input)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        # Cosine similarity
        similarity = (image_embedding * text_embedding).sum()
    return similarity.item()

#------------------------------------------------------------------------------
# 5) Diffusion w/ CLIP Guidance
#------------------------------------------------------------------------------
# We'll do 50 steps, each time we add random Gaussian noise to the shape,
# try N_tries proposals, pick the best shape by CLIP similarity.
def animate_clip_guided_diffusion():
    steps = 50
    N_tries = 5
    noise_magnitude = 0.02

    current_points = original_points.copy()
    shapes_over_time = [current_points.copy()]
    for step in range(steps):
        best_candidate = None
        best_score = -9999
        
        # We generate a few random proposals & pick whichever has highest CLIP similarity
        for _ in range(N_tries):
            proposal = current_points + np.random.normal(0, noise_magnitude, size=current_points.shape)
            score = clip_snowman_score(proposal)
            if score > best_score:
                best_score = score
                best_candidate = proposal
        current_points = best_candidate
        shapes_over_time.append(current_points.copy())

    #------------------------------------------------------------------------------
    # 6) Animate with matplotlib
    #------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,6))
    plt.close()

    # Plot the original shape in black
    ax.plot(original_points[:,0], original_points[:,1], 'k-', linewidth=1, label="Original Snowman")
    (guided_line,) = ax.plot([], [], 'r-', linewidth=2, label="CLIP-Guided Shape")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.8, 2.0)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    title_text = ax.set_title("")
    ax.legend(loc='lower left')

    def init():
        guided_line.set_data([], [])
        title_text.set_text("Step 0")
        return (guided_line, title_text)

    def update(frame):
        points = shapes_over_time[frame]
        guided_line.set_data(points[:,0], points[:,1])
        # We'll show the CLIP similarity in the title
        clip_score = clip_snowman_score(points)
        title_text.set_text(f"Step {frame}, CLIP Score: {clip_score:.3f}")
        return (guided_line, title_text)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(shapes_over_time),
        init_func=init,
        interval=400,
        blit=False,
        repeat=True
    )

    return HTML(anim.to_jshtml())



if __name__ == "__main__":
    print("1. Visualizing Autoregressive Model concept...")
    visualize_autoregressive_model()
    
    print("\n2. Visualizing Variational Autoencoder (VAE) concept...")
    visualize_vae_concept()
    
    print("\n3. Visualizing Generative Adversarial Network (GAN) concept...")
    visualize_gan_concept()
    
    print("\n4. Visualizing Diffusion Model concept...")
    # Run it
    animate_diffusion()
    
    print("\n5. Visualizing Snowman Diffusion Model concept...")
    animate_snowman_diffusion()
    
    print("\n6. Visualizing CLIP-Guided Diffusion Model...")
    animate_clip_guided_diffusion()
    
    print("\nVisualization complete! Check the output images:")
    print("- autoregressive_model.png")
    print("- vae_concept.png")
    print("- gan_concept.png")
    print("- diffusion_concept.png")
