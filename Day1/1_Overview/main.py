"""
Overview of Generative AI - Main Script
Building Blocks of Generative AI Course - Day 1

This script provides an entry point to run all the visualization examples
for the Overview of Generative AI section.
"""

import time
import sys
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "numpy", 
        "matplotlib", 
        "scikit-learn", 
        "seaborn"
    ]
    
    missing_packages = []
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_visualization(name, function):
    """Run a visualization function and handle any errors"""
    print(f"\n{'='*80}")
    print(f"Running {name}...")
    print(f"{'-'*80}")
    
    try:
        # Run the visualization
        function()
        print(f"✓ {name} completed successfully!")
        
    except Exception as e:
        print(f"✗ Error running {name}:")
        print(f"  {type(e).__name__}: {e}")
        return False
    
    return True

def main():
    """Main function to run all visualizations"""
    print("Generative AI Overview - Visualization Examples")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install the required packages and try again.")
        return
    
    # Import visualization modules
    try:
        from generative_vs_discriminative import visualize_generative_vs_discriminative
        from latent_space_visualization import visualize_latent_space, visualize_latent_interpolation
        from generative_model_families import (
            visualize_autoregressive_model,
            visualize_vae_concept,
            visualize_gan_concept,
            visualize_diffusion_concept
        )
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all Python files are in the same directory.")
        return
    
    # Run all visualizations
    visualizations = [
        ("Generative vs Discriminative Models", visualize_generative_vs_discriminative),
        ("Latent Space Visualization", visualize_latent_space),
        ("Latent Space Interpolation", visualize_latent_interpolation),
        ("Autoregressive Model", visualize_autoregressive_model),
        ("Variational Autoencoder (VAE) Concept", visualize_vae_concept),
        ("Generative Adversarial Network (GAN) Concept", visualize_gan_concept),
        ("Diffusion Model Concept", visualize_diffusion_concept)
    ]
    
    successful = 0
    total = len(visualizations)
    
    for name, function in visualizations:
        if run_visualization(name, function):
            successful += 1
        
        # Pause between visualizations to avoid overwhelming the system
        if name != visualizations[-1][0]:  # If not the last visualization
            time.sleep(1)
    
    # Summary
    print("\n" + "="*80)
    print(f"Visualization Summary: {successful}/{total} completed successfully")
    print("Output images saved to the current directory.")
    print("="*80)

if __name__ == "__main__":
    main()
