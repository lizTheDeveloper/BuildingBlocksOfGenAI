# Overview of Generative AI

## What is Generative AI?

- **Definition**: AI systems that can generate new content
- **Content Types**: Text, images, audio, video, code, 3D models
- **Key Difference**: 
  - **Discriminative AI**: Classifies or predicts (Is this A or B?)
  - **Generative AI**: Creates new examples (Generate something like A)

## Evolution of Generative AI

![Timeline of Generative AI](./images/gen_ai_timeline.png)

- **Early Systems (1950s-2000s)**: Rule-based, limited capabilities
- **Statistical Models (2000s)**: Markov models, n-grams
- **Deep Learning Era (2014+)**: GANs, VAEs, autoregressive models
- **Large Foundation Models (2020+)**: GPT, DALL-E, Stable Diffusion

## Why Generative AI Matters

- **Creative Augmentation**: Enhance human creativity
- **Automation**: Generate content at scale
- **Personalization**: Tailor content to individual needs
- **Knowledge Work**: Summarize, draft, research assistance
- **Accessibility**: Make creation accessible to non-experts
- **Problem Solving**: Generate novel solutions

## Core Types of Generative Models

| Model Type | Key Mechanism | Examples | Strengths |
|------------|---------------|----------|-----------|
| **Autoencoders** | Encode-decode through bottleneck | VAEs, VQ-VAEs | Compact representations |
| **GANs** | Generator vs. discriminator | StyleGAN, CycleGAN | High-quality images |
| **Diffusion** | Gradual denoising | DALL-E, Stable Diffusion | Photorealism, control |
| **Autoregressive** | Next-token prediction | GPT, LLaMA | Text, reasoning |
| **Flow-based** | Invertible transformations | Glow, RealNVP | Exact density estimation |

## Generative vs. Discriminative Models

![Generative vs Discriminative](./images/gen_vs_disc.png)

- **Discriminative Models**:
  - Learn decision boundaries between classes
  - Model conditional probability: P(Y|X)
  - Typically simpler, more efficient
  - Examples: SVMs, Decision Trees, CNN Classifiers

- **Generative Models**:
  - Learn the full data distribution
  - Model joint probability: P(X,Y) or P(X)
  - More complex, but more versatile
  - Examples: GANs, VAEs, Language Models

## Applications of Generative AI

- **Creative Tools**: Art creation, music composition, design
- **Content Production**: Marketing copy, blog posts, scripts
- **Software Development**: Code generation, debugging assistance
- **Product Design**: 3D modeling, prototype generation
- **Synthetic Data**: Training data for other AI systems
- **Interactive Experiences**: Games, education, virtual worlds

## Current Capabilities

- **Text**: Long-form content, code, creative writing, summarization
- **Images**: Photorealistic images, style transfer, inpainting, editing
- **Audio**: Speech synthesis, music generation, sound effects
- **Video**: Short clips, editing, animation
- **3D**: Simple models, textures, environments
- **Multimodal**: Text-to-image, image-to-text, video understanding

## Limitations and Challenges

- **Factual Accuracy**: "Hallucinations" and false information
- **Control**: Balancing creativity vs. following instructions
- **Intellectual Property**: Copyright and ownership questions
- **Bias**: Reproducing or amplifying dataset biases
- **Misuse**: Potential for harmful content generation
- **Authenticity**: Detecting AI-generated content

## Building Blocks We'll Explore

![Building Blocks](./images/building_blocks.png)

- **Day 1**:
  - Deep Learning Foundations
  - Latent Space Representations
  - Variational Autoencoders

- **Day 2**:
  - Generative Adversarial Networks
  - Attention Mechanisms
  - Transformer Architecture

- **Day 3**:
  - Hugging Face Ecosystem
  - Fine-tuning Strategies
  - Building Applications

## Course Format and Expectations

- **Structure**: Lecture + Hands-on Implementation
- **Pace**: Fast but thorough coverage of fundamentals
- **Prerequisites**: Python, basic ML knowledge
- **Goal**: Understand core mechanisms + practical implementation
- **Outcome**: Ability to work with and adapt generative models

## Today's Learning Objectives

1. Understand core concepts of generative models
2. Review essential deep learning techniques
3. Explore latent space and its importance
4. Implement a simple generative model
5. Build and train Variational Autoencoders (VAEs)

## Let's Begin Our Generative AI Journey!

---

# Questions Before We Dive In?
