# Building Blocks of Generative Models

## Core Components of Generative Models

- **Probability Distributions**: Model the data space
- **Sampling Techniques**: Generate new examples
- **Latent Space**: Compact, meaningful representations
- **Decoder Architectures**: Transform latent variables to data 
- **Learning Objectives**: Define what makes a "good" generation

## Probability Distributions

- **What**: Mathematical functions that describe the likelihood of outcomes
- **Why**: Generative models learn to approximate data distributions
- **How**: Parametrize distributions with neural networks

![Probability Distributions](./images/probability_distributions.png)

## Common Distributions in Generative Models

| Distribution | Formula | Usage |
|--------------|---------|-------|
| **Gaussian (Normal)** | $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$ | Latent variables, noise modeling |
| **Bernoulli** | $f(k) = p^k(1-p)^{1-k}$ | Binary data (black/white pixels) |
| **Categorical** | $P(X=i) = p_i$ | Discrete choices (tokens, classes) |
| **Mixture of Gaussians** | $f(x) = \sum_{i=1}^{k} w_i \mathcal{N}(x|\mu_i, \sigma_i)$ | Complex, multi-modal distributions |

## Why Distributions Matter

- **Expressiveness**: Different distributions fit different data types
- **Sampling**: Generate new examples by sampling from learned distribution
- **Evaluation**: Compare generated and real data distributions
- **Manipulability**: Modify distribution parameters for controlled generation

## Sampling Techniques

- **Direct Sampling**: Draw directly from a known distribution
- **Rejection Sampling**: Propose and filter samples
- **MCMC**: Sequential sampling through Markov chains
- **Importance Sampling**: Weighted sampling from proposal distribution
- **Ancestral Sampling**: Sample step-by-step in directed models

## Direct Sampling Implementation

```python
# Sample from a Gaussian distribution
def sample_gaussian(mean, std, shape):
    # Create random normal samples
    samples = np.random.normal(0, 1, size=shape)
    # Scale and shift to target distribution
    samples = samples * std + mean
    return samples

# Sample from a mixture of Gaussians
def sample_gaussian_mixture(means, stds, weights, n_samples):
    # Choose which Gaussian to sample from
    component_indices = np.random.choice(
        len(weights), size=n_samples, p=weights
    )
    
    # Sample from selected Gaussians
    samples = np.zeros(n_samples)
    for i in range(n_samples):
        idx = component_indices[i]
        samples[i] = np.random.normal(means[idx], stds[idx])
        
    return samples
```

## The Latent Space

- **Definition**: Lower-dimensional space where data is represented
- **Purpose**: 
  - Compress high-dimensional data
  - Organize data meaningfully
  - Enable controlled generation

![Latent Space](./images/latent_space.png)

## Properties of Good Latent Spaces

- **Smoothness**: Similar points in latent space produce similar outputs
- **Disentanglement**: Individual dimensions control meaningful factors
- **Completeness**: All latent points decode to valid outputs
- **Interpolability**: Moving in latent space creates meaningful transitions
- **Structure**: Organization reflects semantic relationships

## Creating Latent Spaces

- **Autoencoders**: Learn compressed representations
- **Principal Component Analysis**: Linear dimensionality reduction
- **Variational Methods**: Learn probability distributions in latent space
- **Adversarial Learning**: Implicitly model high-dimensional distributions

## Latent Space Visualization

![Latent Space Visualization](./images/latent_visualization.png)

- **Dimensionality Reduction**: t-SNE, UMAP for visualization
- **Latent Traversals**: Vary one dimension while fixing others
- **Interpolation**: Move between two points in latent space
- **Clustering**: Group similar latent representations

## Decoder Architectures

- **Purpose**: Map from latent space to data space
- **Types**:
  - **MLPs**: Fully connected networks for simple data
  - **ConvNets**: For images and spatial data
  - **RNNs/Transformers**: For sequential data (text, audio)
  - **Specialized**: For 3D data, graphs, etc.

## Decoder Implementation

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        
        # Build decoder layers
        modules = []
        
        # First layer from latent space
        modules.append(nn.Linear(latent_dim, hidden_dims[0]))
        modules.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            modules.append(nn.ReLU())
        
        # Output layer
        modules.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # For images, might add sigmoid for [0,1] output
        modules.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        # z is a point in latent space
        return self.decoder(z)
```

## Learning Objectives

- **Maximum Likelihood**: Maximize probability of training data
- **Reconstruction Loss**: Minimize difference between original and generated
- **Adversarial Loss**: Make generated examples indistinguishable from real
- **Perceptual Loss**: Match features in perceptual space (e.g., VGG features)
- **KL Divergence**: Align distributions (e.g., latent space to prior)

## Common Loss Functions

| Loss | Formula | Usage |
|------|---------|-------|
| **MSE** | $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Continuous data (images) |
| **BCE** | $L = -\frac{1}{n}\sum_{i=1}^{n}y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)$ | Binary data |
| **KL** | $D_{KL}(P\|Q) = \sum_{x} P(x) \log\frac{P(x)}{Q(x)}$ | Distribution matching |
| **Perceptual** | $L = \|\phi(y) - \phi(\hat{y})\|^2$ | Feature matching |

## Simple Generative Model Example

```python
class SimpleGenerativeModel:
    def __init__(self, latent_dim, output_dim):
        # Define latent space dimension
        self.latent_dim = latent_dim
        
        # Decoder network: latent → hidden → output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # For image pixels in [0,1]
        )
    
    def sample(self, num_samples=1):
        # Sample from latent prior (standard normal)
        z = torch.randn(num_samples, self.latent_dim)
        
        # Decode to data space
        samples = self.decoder(z)
        return samples
    
    def train(self, dataloader, num_epochs=10, lr=1e-3):
        # Implement training loop
        # ...
```

## Evaluation Metrics

- **Log-Likelihood**: How probable is the generated data?
- **Inception Score/FID**: For images, compare feature distributions
- **BLEU/ROUGE/etc.**: For text, compare against references
- **Human Evaluation**: Subjective quality assessment
- **Task-Specific**: Performance on downstream tasks

## Applications of Basic Generative Models

- **Data Augmentation**: Generate additional training examples
- **Anomaly Detection**: Identify examples outside learned distribution
- **Data Compression**: Efficient representation through latent space
- **Feature Learning**: Unsupervised representation learning
- **Creative Applications**: Generate novel designs, art, etc.

## Today's Hands-On Exercise

1. Implement a simple latent variable model
2. Create a basic decoder architecture
3. Sample from latent space
4. Visualize the generative process
5. Evaluate the quality of generations

---

# Questions?
