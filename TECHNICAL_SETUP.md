# Technical Setup for Building Blocks of Generative AI Course

This document provides detailed instructions for setting up your development environment for the "Building Blocks of Generative AI" course.

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+ recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB of free disk space
- **GPU**: For optimal performance, a CUDA-compatible GPU is recommended for training models (not required for small exercises)

## Python Environment Setup

We recommend using a virtual environment to avoid conflicts with other Python packages. You can use either Conda or venv.

### Option 1: Using Conda (Recommended)

1. **Install Miniconda or Anaconda**:
   - Download from: https://docs.conda.io/en/latest/miniconda.html

2. **Create a new environment**:
   ```bash
   conda create -n genai python=3.9
   ```

3. **Activate the environment**:
   ```bash
   conda activate genai
   ```

4. **Install the required packages**:
   ```bash
   conda install -c conda-forge tensorflow matplotlib numpy pandas scikit-learn seaborn
   conda install -c conda-forge notebook
   pip install tensorflow-probability
   ```

### Option 2: Using venv

1. **Create a new virtual environment**:
   ```bash
   python -m venv genai-env
   ```

2. **Activate the environment**:
   - On Windows:
     ```bash
     genai-env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source genai-env/bin/activate
     ```

3. **Install the required packages**:
   ```bash
   pip install tensorflow tensorflow-probability numpy matplotlib pandas scikit-learn seaborn notebook
   ```

## Additional Packages

Depending on the exercises, you may need to install additional packages:

```bash
# For Day 1 exercises
pip install umap-learn

# For Day 2 exercises (GANs and Transformers)
pip install torch torchvision

# For Day 3 exercises (Hugging Face)
pip install transformers datasets
```

## Jupyter Notebook Setup

Many exercises are provided as Jupyter notebooks. To set up Jupyter:

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install notebook
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open the exercise notebooks** from the Jupyter interface in your browser.

## Testing Your Setup

To verify your setup is working correctly, run the following test script:

```python
# Save this as test_setup.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Check TensorFlow
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Simple plot test
plt.figure(figsize=(10, 5))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Test Plot")
plt.savefig("test_plot.png")
plt.close()

print("Setup test completed successfully!")
```

Run the script:
```bash
python test_setup.py
```

## Troubleshooting

### Common Issues

1. **TensorFlow GPU issues**:
   - Make sure you have the appropriate CUDA and cuDNN versions installed for your TensorFlow version
   - See: https://www.tensorflow.org/install/gpu

2. **Package conflicts**:
   - If you encounter package conflicts, try creating a fresh virtual environment and follow the installation steps carefully

3. **Jupyter Notebook not showing in browser**:
   - Check the URL in the terminal output
   - Try using a different browser
   - Ensure no firewall is blocking the connection

### Getting Help

If you encounter any issues with the setup, please:
1. Check the course forum for similar issues and solutions
2. Post your specific error messages and environment details on the forum
3. Contact the course instructors for additional support

## Optional: Cloud-based Alternatives

If you have issues setting up a local environment or need more computational power, consider these cloud options:

- **Google Colab**: Free access to GPUs for notebooks
  - https://colab.research.google.com/

- **Kaggle Kernels**: Free GPU and TPU access
  - https://www.kaggle.com/code

- **Gradient by Paperspace**: Free and paid tiers with GPU support
  - https://gradient.paperspace.com/

---

Created by The Multiverse School
