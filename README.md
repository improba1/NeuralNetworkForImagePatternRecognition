# Neural Network from Scratch for Image Pattern Recognition

A from-scratch implementation of a fully connected Multi-Layer Perceptron (MLP) using only NumPy. The project's goal was to demystify the fundamental mechanics of neural networks by building one from the ground up to classify simple 4-pixel images into distinct pattern categories.

## 🧠 Project Overview

The neural network was designed to recognize patterns in 4-pixel images:
- **Horizontal lines**: Two top pixels of one color, two bottom of another
- **Vertical lines**: Two right pixels of one color, two left of another  
- **Diagonal lines**: Two diagonal pixels of one color, two others of another
- **Solid color**: All 4 pixels of the same color

## ⚙️ Key Features

- **Pure NumPy Implementation**: No high-level ML frameworks used
- **Custom Architecture**: 12-6-4 neuron structure with bias units
- **Activation Functions**: Implemented ReLU and Softmax from scratch
- **Manual Dataset Generation**: Created 112 training samples (28 per category)
- **Training Visualization**: Monitored loss convergence over 2500 epochs

## 🏗️ Network Architecture
Input Layer (12 neurons)
↓
Hidden Layer (6 neurons + bias) + ReLU activation
↓
Output Layer (4 neurons) + Softmax activation

## 🛠️ Technical Implementation

### Core Components
```python
# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Forward propagation
def forward_propagation(X, parameters):
    # Layer 1
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = relu(Z1)
    
    # Output layer
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = softmax(Z2)
    
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```
## Training Process
Loss Function: Categorical Cross-Entropy

Optimization: Gradient Descent

Epochs: 2500 iterations

Learning Rate: Manually tuned for convergence

## 📊 Results
The model successfully learned to classify patterns, showing clear convergence:

Loss decreased consistently over training epochs

Achieved accurate classification on test patterns

Demonstrated the effectiveness of gradient descent
