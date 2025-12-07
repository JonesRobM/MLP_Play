# Multi-Layer Perceptron Background 🧠

A comprehensive guide to understanding the theory and mathematics behind Multi-Layer Perceptrons (MLPs).

## Table of Contents

1. [What is a Multi-Layer Perceptron?](#what-is-a-multi-layer-perceptron)
2. [Biological Inspiration](#biological-inspiration)
3. [The Perceptron: Building Block](#the-perceptron-building-block)
4. [From Perceptron to MLP](#from-perceptron-to-mlp)
5. [Forward Propagation](#forward-propagation)
6. [Activation Functions](#activation-functions)
7. [Loss Functions](#loss-functions)
8. [Backpropagation](#backpropagation)
9. [Optimisation and Gradient Descent](#optimisation-and-gradient-descent)
10. [Why MLPs Matter](#why-mlps-matter)

---

## What is a Multi-Layer Perceptron?

A **Multi-Layer Perceptron (MLP)** is a type of artificial neural network consisting of multiple layers of interconnected nodes (neurones). It's called "multi-layer" because it has:

1. An **input layer** that receives the data
2. One or more **hidden layers** that transform the data
3. An **output layer** that produces predictions

Unlike simple linear models, MLPs can learn **non-linear relationships** in data, making them powerful tools for complex classification and regression tasks.

### Key Characteristics

- **Feedforward**: Information flows in one direction (input → hidden → output)
- **Fully Connected**: Each neurone connects to all neurones in the next layer
- **Non-linear**: Uses activation functions to capture complex patterns
- **Trainable**: Learns optimal weights through backpropagation

---

## Biological Inspiration

Neural networks draw inspiration from the human brain:

| Biological Neuron | Artificial Neuron |
|-------------------|-------------------|
| Dendrites receive signals | Inputs receive features |
| Cell body sums signals | Weighted sum of inputs |
| Activation threshold | Activation function |
| Axon transmits output | Neurone output |
| Synaptic strength | Weight parameters |

However, artificial neural networks are **simplified abstractions**—they don't truly replicate biological brains but borrow the concept of interconnected processing units.

---

## The Perceptron: Building Block

The **perceptron** is the simplest neural network unit, invented by Frank Rosenblatt in 1958.

### Mathematical Formulation

For inputs **x₁, x₂, ..., xₙ** and weights **w₁, w₂, ..., wₙ**:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
y = activation(z)
```

Where:
- **z** is the weighted sum (also called pre-activation)
- **b** is the bias term (shifts the decision boundary)
- **y** is the output after applying an activation function

### The XOR Problem

A single perceptron can only learn **linearly separable** patterns. This limitation, known as the XOR problem, motivated the development of multi-layer networks:

- **Linearly Separable**: Can be separated by a straight line (e.g., OR, AND gates)
- **Non-linearly Separable**: Requires a curved boundary (e.g., XOR gate, moons dataset)

---

## From Perceptron to MLP

By stacking multiple perceptrons into layers and adding non-linear activation functions, we create an MLP that can learn arbitrarily complex patterns.

### Architecture

```
Input Layer    Hidden Layer(s)    Output Layer
    x₁ ──────→  h₁ ──────→  y₁
    x₂ ──────→  h₂ ──────→  y₂
    x₃ ──────→  h₃ ──────→
    ...         h₄
```

Each arrow represents a **weight** that gets multiplied by the input value. Each neurone in the hidden and output layers:
1. Computes a weighted sum of its inputs
2. Adds a bias term
3. Applies a non-linear activation function

---

## Forward Propagation

Forward propagation is the process of computing predictions by passing inputs through the network.

### Step-by-Step Process

**1. Input to Hidden Layer**

```
z₁ = W₁ · x + b₁
a₁ = σ(z₁)
```

Where:
- **W₁** is the weight matrix (input → hidden)
- **x** is the input vector
- **b₁** is the bias vector for the hidden layer
- **σ** is the activation function
- **a₁** is the activated hidden layer output

**2. Hidden to Output Layer**

```
z₂ = W₂ · a₁ + b₂
a₂ = σ(z₂)
```

Where:
- **W₂** is the weight matrix (hidden → output)
- **b₂** is the bias vector for the output layer
- **a₂** is the final network prediction

### Example: Moons Dataset

For our moons experiment (2 → 4 → 1):
- **Input**: 2 features (x, y coordinates)
- **Hidden**: 4 neurones with sigmoid activation
- **Output**: 1 neurone with sigmoid activation (probability of class 1)

---

## Activation Functions

Activation functions introduce **non-linearity**, allowing networks to learn complex patterns.

### Sigmoid

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Properties:**
- Output range: (0, 1)
- Smooth, differentiable
- Interpretable as probability
- **Gradient**: σ'(z) = σ(z) · (1 - σ(z))

**Used in:** Moons experiment (binary classification)

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

**Properties:**
- Output range: [0, ∞)
- Computationally efficient
- Avoids vanishing gradient problem
- **Gradient**: 1 if z > 0, else 0

**Used in:** Wine experiment (modern best practice)

### Tanh (Hyperbolic Tangent)

```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
```

**Properties:**
- Output range: (-1, 1)
- Zero-centred (unlike sigmoid)
- Stronger gradients than sigmoid
- **Gradient**: tanh'(z) = 1 - tanh²(z)

**Comparison:**

![Activation Functions Comparison](https://upload.wikimedia.org/wikipedia/commons/6/6f/Gjl-t%28x%29.svg)

---

## Loss Functions

The **loss function** measures how wrong the network's predictions are. During training, we minimise this loss.

### Mean Squared Error (MSE)

```
L = (1/m) Σ(ŷᵢ - yᵢ)²
```

**Used for:** Regression and binary classification (moons experiment)

**Properties:**
- Penalises large errors heavily (squared term)
- Simple gradient: dL/dŷ = 2(ŷ - y)
- Works well with sigmoid output

### Cross-Entropy Loss

For **binary classification:**
```
L = -(1/m) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

For **multi-class classification:**
```
L = -(1/m) Σ Σ yᵢⱼ log(ŷᵢⱼ)
```

**Used for:** Wine experiment (multi-class)

**Properties:**
- Probabilistically motivated
- Stronger gradients for confident wrong predictions
- Standard choice for classification

---

## Backpropagation

**Backpropagation** is the algorithm that computes gradients for all weights by propagating errors backward through the network using the **chain rule** from calculus.

### The Chain Rule

For a composition of functions f(g(x)), the derivative is:

```
df/dx = (df/dg) · (dg/dx)
```

In neural networks, we chain together many such derivatives to find how the loss changes with respect to each weight.

### Gradient Flow (2-Layer Network)

**Output Layer Gradients:**

```
dL/dz₂ = (a₂ - y)                    # Error at output
dL/dW₂ = a₁ᵀ · dL/dz₂                # Gradient for output weights
dL/db₂ = mean(dL/dz₂)                # Gradient for output bias
```

**Hidden Layer Gradients:**

```
dL/da₁ = dL/dz₂ · W₂ᵀ                # Propagate error backward
dL/dz₁ = dL/da₁ ⊙ σ'(z₁)             # Apply activation derivative
dL/dW₁ = xᵀ · dL/dz₁                 # Gradient for hidden weights
dL/db₁ = mean(dL/dz₁)                # Gradient for hidden bias
```

Where ⊙ represents element-wise multiplication (Hadamard product).

### Why It Works

Backpropagation efficiently computes all gradients in a **single backward pass** by reusing intermediate computations. This is far more efficient than computing each gradient independently.

---

## Optimisation and Gradient Descent

Once we have gradients, we update weights to minimise the loss.

### Vanilla Gradient Descent

```
W_new = W_old - η · (dL/dW)
```

Where **η** (eta) is the **learning rate**, a hyperparameter controlling step size.

### Variants

**Batch Gradient Descent:**
- Uses entire dataset for each update
- Stable but slow for large datasets

**Stochastic Gradient Descent (SGD):**
- Uses one sample at a time
- Fast but noisy updates

**Mini-batch Gradient Descent:**
- Uses small batches (e.g., 32, 64 samples)
- Good balance of speed and stability

### Adam Optimiser

**Adam** (Adaptive Moment Estimation) is the most popular modern optimiser:

```
m_t = β₁ · m_{t-1} + (1-β₁) · gradient
v_t = β₂ · v_{t-1} + (1-β₂) · gradient²
W_new = W_old - η · m_t / (√v_t + ε)
```

**Advantages:**
- Adapts learning rate per parameter
- Combines momentum with adaptive learning rates
- Works well with minimal tuning

**Used in:** Wine experiment

---

## Why MLPs Matter

### Universal Approximation Theorem

**Theorem:** A feedforward network with a single hidden layer containing a finite number of neurones can approximate any continuous function on compact subsets of ℝⁿ, under mild assumptions on the activation function.

**In plain English:** Given enough neurones, an MLP can learn virtually any pattern!

### Practical Applications

MLPs serve as the foundation for understanding:
- **Convolutional Neural Networks (CNNs)** - Image processing
- **Recurrent Neural Networks (RNNs)** - Sequential data
- **Transformers** - Natural language processing
- **Deep Learning** - All modern AI systems

### Historical Significance

- **1958:** Rosenblatt invents the perceptron
- **1969:** Minsky & Papert show perceptron limitations (XOR problem)
- **1986:** Rumelhart et al. popularise backpropagation
- **2012:** AlexNet wins ImageNet, sparking the deep learning revolution
- **Today:** MLPs remain fundamental building blocks

---

## Further Reading

### Mathematical Foundations
- 📘 *Deep Learning* by Goodfellow, Bengio, and Courville (Chapter 6)
- 📘 *Neural Networks and Deep Learning* by Michael Nielsen (Free online)
- 📘 *Pattern Recognition and Machine Learning* by Christopher Bishop

### Interactive Resources
- 🎮 [TensorFlow Playground](https://playground.tensorflow.org/) - Visualise MLP learning
- 🎥 [3Blue1Brown Neural Network Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- 📊 [Distill.pub](https://distill.pub/) - Interactive ML explanations

### Research Papers
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251-257.

---

**Ready to see MLPs in action?** Check out the [Experiments Guide](experiments.md) to explore our hands-on implementations!
