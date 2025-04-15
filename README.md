# Building Blocks of Generative AI Course

This repository contains all materials for the three-day workshop on Generative AI and Large Language Models.

## Course Structure

The course is organized into three days:

### Day 1: Foundations and Autoencoders
- Overview of Generative AI
- Deep Learning Primer
- Building Blocks of Generative Models
- Variational Autoencoders (VAEs)

### Day 2: GANs, LLMs, and Transformers
- Generative Adversarial Networks (GANs)
- Introduction to LLMs
- Attention Mechanisms
- Transformers

### Day 3: Applications and Fine-tuning
- Hugging Face Ecosystem
- Training LLMs
- Fine-tuning Strategies
- Transfer Learning

## Materials Organization

Each day's materials are organized as follows:

```
BuildingBlocksOfGenAI/
├── Day1/
│   ├── 1_Overview/
│   │   ├── generative_vs_discriminative.py
│   │   ├── latent_space_visualization.py
│   │   ├── generative_model_families.py
│   │   ├── main.py
│   ├── 2_DeepLearningPrimer/
│   │   ├── deep_learning_basics.py
│   ├── 3_BuildingBlocks/
│   │   ├── probability_sampling.py
│   │   ├── latent_space_representation.py
│   │   ├── simple_generative_model.py
│   ├── 4_VAEs/
│   │   ├── vae_exercise_mnist.py
│   │   ├── vae_solution.py
├── Day2/
│   ├── [To be added]
├── Day3/
│   ├── [To be added]
```

## Running the Examples

Each Python file is designed to be self-contained and can be run individually. For example:

```bash
cd BuildingBlocksOfGenAI/Day1/1_Overview
python generative_vs_discriminative.py
```

Some files may require you to fill in missing code as part of exercises. Solutions are provided in separate files where appropriate.

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow Probability (for some examples)

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn tensorflow-probability
```

## Resources

The content is based on the course outline provided and includes hands-on exercises to reinforce the concepts covered in lectures.

Each code example includes detailed comments to help you understand what's happening at each step.
