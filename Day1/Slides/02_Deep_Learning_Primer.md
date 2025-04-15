# Deep Learning Primer

## Neural Networks: The Building Blocks

![Neural Network Architecture](./images/neural_network.png)

- **Neurons**: Units that receive inputs, apply activation functions, and produce outputs
- **Weights**: Parameters that determine the strength of connections
- **Layers**: Collections of neurons that process information hierarchically
- **Activation Functions**: Non-linear functions that enable complex pattern recognition

## Neural Network Components

```python
class SimpleNeuron:
    def __init__(self, weights, bias):
        self.weights = weights  # Connection strengths
        self.bias = bias        # Offset value
        
    def forward(self, inputs):
        # Weighted sum of inputs
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        # Apply activation function
        output = self.activation(z)
        return output
        
    def activation(self, z):
        # ReLU activation function
        return max(0, z)
```

## Key Activation Functions

| Function | Formula | Properties | Use Cases |
|----------|---------|------------|-----------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Range: [0,1], Smooth | Binary classification outputs |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Range: [-1,1], Zero-centered | Hidden layers (older networks) |
| **ReLU** | $\text{ReLU}(x) = \max(0, x)$ | Range: [0,∞), Fast | Hidden layers (modern default) |
| **Leaky ReLU** | $f(x) = \begin{cases} x, & \text{if}\ x > 0 \\ \alpha x, & \text{otherwise} \end{cases}$ | Prevents "dying ReLU" | Hidden layers (improved version) |
| **Softmax** | $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$ | Multi-class probabilities | Classification output layers |

## Multi-Layer Neural Networks

- **Input Layer**: Receives raw features
- **Hidden Layers**: Extract hierarchical features
  - Early layers: Simple patterns
  - Deep layers: Complex abstractions
- **Output Layer**: Produces predictions
- **Forward Propagation**: Information flow from input to output

## Forward Propagation

```python
def forward_propagation(inputs, network):
    # Start with raw inputs
    current_activations = inputs
    
    # Process through each layer
    for layer in network:
        next_activations = []
        # Process each neuron in the layer
        for neuron in layer:
            output = neuron.forward(current_activations)
            next_activations.append(output)
        # Outputs of this layer become inputs to next layer
        current_activations = next_activations
        
    # Final layer outputs are the predictions
    return current_activations
```

## Training Neural Networks

- **Loss Functions**: Measure prediction errors
  - MSE: Mean Squared Error (regression)
  - Cross-Entropy: Classification problems
  - Kullback-Leibler Divergence: Distribution matching

- **Optimization**: Process of finding optimal weights
  - Goal: Minimize the loss function
  - Challenge: High-dimensional, non-convex optimization

## Gradient Descent

![Gradient Descent](./images/gradient_descent.png)

- **Intuition**: Walk downhill in the error landscape
- **Process**:
  1. Calculate gradient (direction of steepest increase)
  2. Move in the opposite direction
  3. Repeat until convergence or stopping criterion

```python
def gradient_descent(params, learning_rate, grad_fn, iterations):
    for i in range(iterations):
        # Calculate gradient of loss w.r.t. parameters
        gradients = grad_fn(params)
        
        # Update each parameter in the opposite direction of its gradient
        for j in range(len(params)):
            params[j] -= learning_rate * gradients[j]
            
    return params
```

## Backpropagation

- **Key Insight**: Efficiently compute gradients through the chain rule
- **Process**:
  1. Forward pass: Compute outputs and cache intermediate values
  2. Compute output layer error
  3. Propagate error backwards, layer by layer
  4. Update weights based on their contribution to error

## Backpropagation Visualization

![Backpropagation](./images/backpropagation.png)

- **Forward Pass**: Compute activations (blue)
- **Backward Pass**: Compute gradients (red)
- **Weight Updates**: Proportional to:
  - How much a weight contributes to the error
  - Input activation to that weight

## Optimization Algorithms

| Algorithm | Description | Advantages | Challenges |
|-----------|-------------|------------|------------|
| **SGD** | Update with single sample gradient | Simple, low memory | Noisy updates, sensitive to scaling |
| **Momentum** | Accumulate gradients over time | Faster convergence, handles noise | Additional hyperparameter |
| **RMSProp** | Adaptive learning rates per parameter | Handles different scales | Requires more computation |
| **Adam** | Combines momentum and adaptive rates | Current standard, robust | Complex, can generalize poorly |

## Overfitting and Regularization

![Overfitting](./images/overfitting.png)

- **Overfitting**: Model learns noise in training data
  - Perfect performance on training data
  - Poor performance on new data
  
- **Regularization**: Techniques to prevent overfitting
  - L1/L2 regularization: Penalize large weights
  - Dropout: Randomly disable neurons during training
  - Early stopping: Stop when validation loss increases
  - Data augmentation: Create variations of training examples

## Hyperparameters

- **Learning Rate**: Size of gradient descent steps
  - Too high: Divergence or oscillation
  - Too low: Slow convergence or stuck in local minima
  
- **Batch Size**: Number of examples processed at once
  - Large: Stable but slow and memory-intensive
  - Small: Noisy but potentially faster convergence
  
- **Architecture**: Network structure
  - Depth: Number of layers
  - Width: Neurons per layer
  - Skip connections: Connect non-adjacent layers

## Training Workflow

```python
# Initialize model with random weights
model = NeuralNetwork()

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch.inputs)
        
        # Calculate loss
        loss = loss_fn(outputs, batch.targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    # Validation
    val_loss = evaluate(model, val_dataloader)
    print(f"Epoch {epoch}: Val Loss {val_loss:.4f}")
```

## Deep Learning Frameworks

| Framework | Key Features | Best For |
|-----------|-------------|----------|
| **PyTorch** | Dynamic computation graph, Pythonic | Research, rapid prototyping |
| **TensorFlow** | Static+dynamic graphs, production tools | Deployment, mobile/edge |
| **JAX** | Functional, GPU/TPU-optimized | High-performance computing |
| **Keras** | High-level API, simple interface | Beginners, quick experiments |

## Modern Neural Network Architectures

- **Convolutional Neural Networks (CNNs)**
  - Specialized for grid data (images)
  - Shared weights reduce parameters
  - Translation invariance

- **Recurrent Neural Networks (RNNs)**
  - Process sequential data
  - Maintain internal state
  - LSTM and GRU variants

- **Transformers**
  - Self-attention mechanism
  - Parallel processing of sequences
  - State-of-the-art for many tasks

## Why Deep Learning for Generative AI?

- **Representational Power**: Learn complex distributions
- **End-to-End Learning**: No manual feature engineering
- **Scalability**: Performance improves with more data and compute
- **Flexibility**: Same principles across data types (text, images, audio)
- **Transfer Learning**: Leverage knowledge across tasks

## Today's Hands-On Exercises

1. Implement a simple neural network for classification
2. Explore gradient descent optimization
3. Visualize the learning process
4. Apply regularization techniques
5. Build foundation for generative models

---

# Questions?
