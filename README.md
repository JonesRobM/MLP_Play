# MLP_Play 🧠

**Learn how neural networks work from scratch!** This project implements a Multi-Layer Perceptron (MLP) from the ground up using PyTorch, with **manual backpropagation** to demonstrate the fundamental mechanics of how neural networks learn.

## 🎯 What You'll Learn

This educational notebook shows you how to:
- 🔢 Build a neural network **without** relying on automatic differentiation
- 📐 Implement forward propagation (how predictions are made)
- 🔄 Implement backward propagation (how networks learn from mistakes)
- 📊 Train a classifier on the classic "moons" dataset
- 🎨 Visualize decision boundaries to see what the network learned

## 🚀 Quick Start

### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or for GPU support (CUDA 12.6):
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
```

### Run the Notebook

```bash
jupyter lab notebooks/mlp.ipynb
```

## 📓 What's Inside

The notebook demonstrates a **simple 2-layer neural network** that learns to classify non-linearly separable data:

- **Input Layer**: 2 features (x, y coordinates)
- **Hidden Layer**: 4 neurons with sigmoid activation
- **Output Layer**: 1 neuron for binary classification

### Why Manual Backpropagation?

Most tutorials use frameworks that hide the math. This notebook **shows you the actual gradient calculations** so you understand:
- How errors propagate backward through layers 🔙
- How gradients are computed using the chain rule 🔗
- How weights are updated via gradient descent 📉

## 🎓 Educational Features

✨ **Heavily commented code** - Every line explains what and why
📊 **Visualizations** - See the training data, loss curves, and decision boundaries
🔬 **Mathematical explanations** - Gradient formulas and activation derivatives included
🎯 **96.60% accuracy** - Watch the network learn to separate the moons in just 5,000 epochs!

## 🌙 The Moons Dataset

The notebook uses scikit-learn's `make_moons` dataset - two interleaving half-circles that can't be separated by a straight line. This demonstrates why we need **non-linear activations** in neural networks.

![Moons Dataset](figures/moons_dataset.png)
*Training and testing data split - notice the two crescent shapes that interleave*

## 📈 Example Results

### Training Progress

Watch the network learn! The loss curve shows how the model improves over 5,000 epochs:

![Training Loss](figures/training_loss.png)
*Loss decreases exponentially as the network learns the pattern*

### Decision Boundary Visualization

See what the network learned! The decision boundary shows how the MLP separates the two classes:

![Decision Boundary](figures/decision_boundary.png)
*Left: MLP predictions | Right: Ground truth - The network achieves 96.60% accuracy!*

## 🚀 Further Learning

Now that you understand the fundamentals, here are ways to deepen your knowledge:

### 🔧 Experiments to Try

- **Add more layers** - Try a 2-8-4-1 architecture. How does depth affect learning?
- **Different activations** - Replace sigmoid with ReLU or tanh. What changes?
- **Change the dataset** - Try `make_circles` or `make_classification` from scikit-learn
- **Implement momentum** - Add momentum to gradient descent for faster convergence
- **Try different loss functions** - Implement Binary Cross-Entropy instead of MSE
- **Add regularization** - Implement L2 regularization to prevent overfitting

### 📚 Recommended Resources

**Video Series:**
- 🎥 [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Beautiful visualizations of how neural networks work
- 🎥 [Andrej Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0) - Build neural nets from scratch

**Interactive Learning:**
- 🎮 [TensorFlow Playground](https://playground.tensorflow.org/) - Interactive neural network visualization
- 📖 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online book

**Next Steps:**
- Learn about **Convolutional Neural Networks (CNNs)** for image processing
- Explore **Recurrent Neural Networks (RNNs)** for sequence data
- Study **PyTorch's autograd** to see how automatic differentiation works
- Dive into **optimization algorithms** (Adam, RMSprop, AdaGrad)

### 💡 Challenges

1. Can you achieve >97% accuracy by tuning hyperparameters or training longer?
2. Can you implement batch training instead of full-batch?
3. Can you add learning rate decay?
4. Can you visualize the hidden layer activations?

---

**Perfect for**: Students, developers learning ML fundamentals, or anyone curious about what's really happening inside a neural network! 🎉