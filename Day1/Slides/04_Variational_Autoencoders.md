# Variational Autoencoders (VAEs)

## From Autoencoders to Variational Autoencoders

![Autoencoder vs VAE](./images/ae_vs_vae.png)

- **Standard Autoencoder**:
  - Deterministic mapping to latent space
  - No control over latent space structure
  - Can't easily generate new samples

- **Variational Autoencoder (VAE)**:
  - Probabilistic mapping to latent space
  - Structured latent space (usually Gaussian)
  - Enables sampling and generation

## VAE Architecture

![VAE Architecture](./images/vae_architecture.png)

- **Encoder (Recognition Network)**:
  - Maps input x to distribution parameters μ, σ
  - Approximates posterior q(z|x)

- **Latent Space**:
  - Distribution instead of single point
  - Usually multivariate Gaussian

- **Decoder (Generator Network)**:
  - Reconstructs input from latent samples
  - Models likelihood p(x|z)

## The Reparameterization Trick

![Reparameterization Trick](./images/reparameterization.png)

- **Problem**: Sampling is not differentiable
- **Solution**: Reparameterize z = μ + σ * ε where ε ~ N(0,1)
- **Benefit**: Allows backpropagation through the sampling process

```python
def reparameterize(mu, logvar):
    """
    Sample from the latent distribution using the reparameterization trick
    """
    std = torch.exp(0.5 * logvar)  # Convert logvar to std
    eps = torch.randn_like(std)    # Random noise from N(0,1)
    z = mu + eps * std             # Reparameterized sample
    return z
```

## VAE Loss Function

- **Reconstruction Loss**: How well does the model reconstruct inputs?
  - For images: Mean squared error (MSE) or binary cross-entropy (BCE)
  - For discrete data: Cross-entropy or negative log-likelihood

- **KL Divergence**: How close is latent distribution to the prior?
  - Regularizes the latent space
  - Makes sampling possible
  - For Gaussian: Closed form solution

```python
def vae_loss(x, x_recon, mu, logvar):
    """
    VAE loss function: reconstruction loss + KL divergence
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence: D_KL(q(z|x) || p(z))
    # For Gaussian: -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + kl_loss
    
    return total_loss
```

## The ELBO Objective

- **Evidence Lower Bound (ELBO)**:
  - Mathematically: log p(x) ≥ E[log p(x|z)] - D_KL(q(z|x) || p(z))
  - Reconstruction term: Maximize log likelihood of data given latent
  - KL term: Minimize divergence between encoder output and prior

- **Interpretation**:
  - Reconstruction: "Make the decoded samples look like the input"
  - KL Divergence: "Make the latent distribution match the prior"
  - Balancing act between reconstruction quality and regularization

## VAE Implementation

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent mean and variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For images scaled to [0,1]
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
```

## Training a VAE

```python
# Initialize model, optimizer
model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten images
        data = data.view(data.size(0), -1)
        
        # Forward pass
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Calculate loss
        loss = vae_loss(data, recon_batch, mu, logvar)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Log progress
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
```

## Sampling from a VAE

```python
def generate_samples(model, num_samples=16):
    """
    Generate samples from the VAE by sampling from the latent space
    """
    # Sample from the prior p(z) ~ N(0, I)
    z = torch.randn(num_samples, model.latent_dim)
    
    # Decode the latent samples
    with torch.no_grad():
        samples = model.decode(z)
    
    # Reshape to images if needed
    samples = samples.view(num_samples, 1, 28, 28)
    
    return samples
```

## Properties of VAE Latent Space

- **Continuity**: Similar points in latent space produce similar outputs
- **Completeness**: Any point sampled from prior gives reasonable output
- **Disentanglement**: With proper regularization, dimensions can capture 
  meaningful factors of variation

## Latent Space Visualization and Exploration

![VAE Latent Space](./images/vae_latent_space.png)

```python
# Visualize latent space (2D projection with t-SNE)
def visualize_latent_space(model, data_loader):
    # Encode all test data
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.view(data.size(0), -1)
            mu, _ = model.encode(data)  # Just use means
            latent_vectors.append(mu)
            labels.append(label)
    
    # Concatenate batches
    latent_vectors = torch.cat(latent_vectors, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Use t-SNE for visualization
    tsne = TSNE(n_components=2)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                          c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of VAE Latent Space')
    plt.savefig('vae_latent_space.png')
```

## Latent Space Interpolation

![Latent Interpolation](./images/latent_interpolation.png)

```python
def interpolate_latent(model, img1, img2, steps=10):
    """
    Interpolate between two images in latent space
    """
    # Encode images
    with torch.no_grad():
        mu1, _ = model.encode(img1.view(1, -1))
        mu2, _ = model.encode(img2.view(1, -1))
    
    # Interpolate in latent space
    interpolations = []
    for alpha in torch.linspace(0, 1, steps):
        z = alpha * mu1 + (1 - alpha) * mu2
        # Decode interpolated latent vector
        recon = model.decode(z)
        interpolations.append(recon)
    
    return torch.cat(interpolations, dim=0)
```

## Conditional VAEs

![Conditional VAE](./images/conditional_vae.png)

- **Extension**: Include conditioning variable in encoder and decoder
- **Inputs**: Both x and condition c (e.g., class label)
- **Control**: Generate samples with specific attributes
- **Formula**: Model p(x|z,c) instead of just p(x|z)

```python
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder (now takes x and c)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            # Rest of encoder...
        )
        
        # Decoder (also takes z and c)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            # Rest of decoder...
        )
        
    def forward(self, x, c):
        # One-hot encode the condition
        c_onehot = F.one_hot(c, self.num_classes).float()
        
        # Concatenate input and condition
        x_c = torch.cat([x, c_onehot], dim=1)
        
        # Encode to get latent distribution
        mu, logvar = self.encode(x_c)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Concatenate latent and condition for decoder
        z_c = torch.cat([z, c_onehot], dim=1)
        
        # Decode
        x_recon = self.decode(z_c)
        
        return x_recon, mu, logvar
```

## Variational Autoencoder Variants

- **β-VAE**: Increased KL weight for better disentanglement
- **VQ-VAE**: Discrete latent space with vector quantization
- **VRNN**: Recurrent VAE for sequential data
- **VAE-GAN**: Combines VAE with GAN discriminator
- **NVAE**: Hierarchical VAE with normalizing flows
- **VD-VAE**: Very deep VAE with residual connections

## Practical Applications of VAEs

- **Image Generation**: Create novel images
- **Anomaly Detection**: Identify out-of-distribution samples
- **Data Augmentation**: Generate variations of training data
- **Feature Learning**: Unsupervised representation learning
- **Drug Discovery**: Generate novel molecular structures
- **Recommender Systems**: Collaborative filtering with uncertainty

## Limitations of VAEs

- **Image Quality**: Often blurry compared to GANs
- **Posterior Collapse**: Latent variables may be ignored
- **Disentanglement**: Hard to achieve without special techniques
- **Distribution Mismatch**: Gaussian assumptions may not fit data
- **Evaluation**: Hard to evaluate generation quality

## Today's VAE Exercise

1. Implement a VAE for image generation
2. Train on MNIST/Fashion-MNIST dataset
3. Visualize the latent space
4. Sample new images
5. Experiment with latent space interpolation

---

# Questions?
