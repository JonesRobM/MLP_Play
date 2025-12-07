# Introduction to MLPs 🧠 

**Learn how neural networks work from scratch!** This educational repository implements Multi-Layer Perceptrons (MLPs) from the ground up, demonstrating both **manual backpropagation** and modern PyTorch implementations to help you understand the fundamental mechanics of how neural networks learn.

## 🎯 Educational Purpose

This repository is designed for **learning and teaching** the fundamentals of neural networks. Through hands-on implementations and detailed visualisations, you'll gain deep insight into:

- 🔢 How neural networks make predictions (forward propagation)
- 🔄 How networks learn from their mistakes (backpropagation)
- 📊 What hidden layers actually "see" in the data
- 🎨 How decision boundaries emerge during training
- 📈 The difference between binary and multi-class classification

Perfect for students, educators, and developers who want to truly understand what's happening inside a neural network!

## 📚 Documentation

- **[MLP Background](docs/mlp_background.md)** - Theory and mathematics behind Multi-Layer Perceptrons
- **[Experiments Guide](docs/experiments.md)** - Detailed walkthrough of each notebook experiment

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

### Run the Notebooks

```bash
jupyter lab notebooks/moons.ipynb
# or
jupyter lab notebooks/wine.ipynb
```

## 📓 What's Inside

This repository contains two educational experiments demonstrating different aspects of MLPs:

### 🌙 Moons Classification (`moons.ipynb`)
**Binary Classification with Manual Backpropagation**

A from-scratch implementation that shows you the actual mathematics:
- **Architecture**: 2 inputs → 4 hidden neurones (sigmoid) → 1 output
- **Dataset**: Non-linearly separable "moons" (10,000 samples)
- **Special Feature**: Manual backpropagation with detailed gradient calculations
- **Accuracy**: ~96-97% on test set

![Moons Dataset](figures/moons_dataset.png)
*Two interleaving crescents that require non-linear separation*

### 🍷 Wine Classification (`wine.ipynb`)
**Multi-Class Classification with PyTorch**

A modern implementation demonstrating multi-class problems:
- **Architecture**: 13 inputs → 64 hidden neurones (ReLU) → 3 outputs
- **Dataset**: Wine cultivar classification (178 samples, 13 chemical features)
- **Special Feature**: Comprehensive visualisations including PCA, correlation matrices, and ROC curves
- **Accuracy**: ~98% on test set

![Wine PCA](figures/wine_pca.png)
*Three wine classes visualised in principal component space*

## 🎓 Key Educational Features

### Visualisations Across Both Notebooks

Both experiments include extensive visualisations to build intuition:

✨ **Dataset Exploration** - See the raw data and class distributions
📊 **Training Dynamics** - Loss curves showing learning progress
🎨 **Decision Boundaries** - See how the network carves up the feature space
🧠 **Hidden Layer Activations** - Understand what individual neurones learn
📈 **Performance Metrics** - Confusion matrices, ROC curves, and accuracy plots

### Code Quality

🔬 **Detailed Comments** - Every line explains the "what" and "why" in British English
📐 **Mathematical Explanations** - Gradient formulas and activation derivatives
🏗️ **Two Approaches** - Manual implementation (moons) vs. PyTorch best practices (wine)

## 🌙 Moons Experiment Results

### Training Progress

![Training Loss](figures/moon_training_loss.png)
*Loss decreases exponentially as the network learns the pattern*

### Decision Boundary

![Decision Boundary](figures/moon_decision_boundary.png)
*Left: MLP predictions | Right: Ground truth*

### Hidden Layer Activations

![Hidden Activations](figures/moon_hidden_activations.png)
*How each of the 4 neurones responds to different regions of input space*

## 🍷 Wine Experiment Results

### Feature Relationships

![Correlation Matrix](figures/wine_correlation_matrix.png)
*Understanding which chemical properties are related*

### Model Performance

![Confusion Matrix](figures/wine_confusion_matrix.png)
*Classification performance across all three wine classes*

![ROC Curves](figures/wine_roc_curves.png)
*Receiver Operating Characteristic curves showing excellent discrimination*

### What the Network Learnt

![Wine Activations](figures/wine_activation_heatmap.png)
*Different neurones specialise in detecting different wine types*

## 🔧 Experiments to Try

### Modify the Moons Notebook
- **Add more layers** - Try 2-8-4-1 architecture
- **Different activations** - Replace sigmoid with tanh
- **Change the dataset** - Try `make_circles` or `make_classification`
- **Implement momentum** - Add momentum to gradient descent

### Modify the Wine Notebook
- **Adjust architecture** - Try different hidden layer sizes
- **Add dropout** - Implement regularisation to prevent overfitting
- **Feature engineering** - Create interaction terms between features
- **Try other datasets** - Use iris, digits, or breast cancer datasets

## 📚 Learning Resources

### Video Series
- 🎥 [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Beautiful visualisations
- 🎥 [Andrej Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0) - Build from scratch

### Interactive Learning
- 🎮 [TensorFlow Playground](https://playground.tensorflow.org/) - Interactive neural network visualisation
- 📖 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online book

### Next Steps
- Learn about **Convolutional Neural Networks (CNNs)** for image processing
- Explore **Recurrent Neural Networks (RNNs)** for sequence data
- Study **PyTorch's autograd** to see how automatic differentiation works
- Dive into **optimisation algorithms** (Adam, RMSprop, AdaGrad)

## 💡 Challenge Questions

1. Why does the moons experiment use sigmoid activation while wine uses ReLU?
2. Can you explain why the wine experiment needs 64 hidden neurones whilst moons only needs 4?
3. What would happen if you trained the wine classifier without standardising the features?
4. Can you achieve >98% accuracy on the wine dataset by tuning hyperparameters?
5. How do the hidden layer activations differ between correctly and incorrectly classified samples?

## 📁 Repository Structure

```
MLP_Play/
├── notebooks/
│   ├── moons.ipynb       # Binary classification with manual backprop
│   └── wine.ipynb        # Multi-class classification with PyTorch
├── figures/              # All visualisations generated by notebooks
├── docs/
│   ├── mlp_background.md # Theory and mathematics of MLPs
│   └── experiments.md    # Detailed explanation of each experiment
├── requirements.txt      # Python dependencies
└── README.md            # You are here!
```

---

**Built for learning** 🎓 | **Open for exploration** 🔬 | **Perfect for teaching** 👨‍🏫

*This project uses British English spelling throughout all documentation and code comments.*
