"""
Fashion MNIST Classifier - Hands-On Exercise
Building Blocks of Generative AI Course - Day 1

This script guides students through building a simple neural network for Fashion MNIST
classification, emphasizing forward pass, loss computation, backpropagation, and parameter updates.

The Fashion MNIST dataset consists of 70,000 grayscale images of clothing items,
each 28x28 pixels, with 10 categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from time import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for the training and testing data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
])

# Load the Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                       download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                      download=True, transform=transform)

# Create data loaders for batching
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the class labels for Fashion MNIST
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Visualize some example images
def visualize_dataset_samples(dataset, num_samples=10, class_names=None):
    """Visualize random samples from the dataset"""
    fig = plt.figure(figsize=(12, 4))
    
    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]
        
        # Add subplot
        ax = fig.add_subplot(1, num_samples, i + 1)
        
        # Convert tensor to numpy and reshape for display
        img = img.numpy()[0]  # Get the first channel (grayscale)
        
        # Display the image
        ax.imshow(img, cmap='gray')
        
        # Set title to the class name if provided
        if class_names:
            ax.set_title(class_names[label])
        else:
            ax.set_title(f"Label: {label}")
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Visualize some examples
print("Displaying random samples from Fashion MNIST:")
visualize_dataset_samples(train_dataset, class_names=fashion_mnist_labels)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # TODO: Define the network architecture
        # Input size: 28x28 = 784 (flattened image)
        # Define at least 2 hidden layers
        # Output size: 10 (number of classes)
        
        # Example structure (uncomment to use):
        self.fc1 = nn.Linear(28*28, 128)  # Input to hidden layer 1
        self.fc2 = nn.Linear(128, 64)     # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(64, 10)      # Hidden layer 2 to output
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Remember to flatten the image first: x.view(-1, 28*28)
        # Apply activation functions between layers
        
        # Example implementation (uncomment to use):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)          # Output layer (logits)
        
        return x

# Define training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    """Train the model for one epoch"""
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # TODO: Implement the training steps
        # 1. Zero the gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        output = model(data)
        
        # 3. Calculate loss
        loss = criterion(output, target)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update running loss
        running_loss += loss.item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Accuracy: {100.*correct/total:.2f}% | '
                  f'Time: {time()-start_time:.2f}s')
    
    # Print epoch summary
    print(f'Epoch {epoch} completed | '
          f'Loss: {running_loss/len(train_loader):.4f} | '
          f'Accuracy: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader)

# Define testing function
def test(model, device, test_loader, criterion):
    """Evaluate the model on the test set"""
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            test_loss += criterion(output, target).item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # Calculate average loss and accuracy
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

# Function to visualize model predictions
def visualize_predictions(model, device, test_loader, class_names, num_samples=10):
    """Visualize model predictions on random test samples"""
    model.eval()  # Set the model to evaluation mode
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Select a subset of images
    indices = np.random.choice(len(images), num_samples, replace=False)
    images = images[indices].to(device)
    labels = labels[indices].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Plot the images and their predictions
    fig = plt.figure(figsize=(15, 6))
    
    for i in range(num_samples):
        # Plot original image
        ax = fig.add_subplot(2, num_samples, i + 1)
        img = images[i].cpu().numpy()[0]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {class_names[labels[i].item()]}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot prediction
        ax = fig.add_subplot(2, num_samples, i + 1 + num_samples)
        ax.imshow(img, cmap='gray')
        # Color the title based on correctness
        title_color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {class_names[predicted[i].item()]}", color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create model
    model = SimpleNN().to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Train the model
    epochs = 5
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        # Train for one epoch
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        
        # Test the model
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    
    # Plot training and testing loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'bo-', label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, 'ro-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accuracies, 'go-')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.show()
    
    # Visualize model predictions
    visualize_predictions(model, device, test_loader, fashion_mnist_labels)
