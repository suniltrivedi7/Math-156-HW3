"""
This script implements a binary logistic regression model trained using mini-batch SGD.
It minimizes the cross-entropy loss function to optimize the weights.

Hyperparameters:
- Batch size
- Fixed learning rate
- Maximum number of iterations

Usage:
- Modify hyperparameters as needed.
- Run the script to train the model on sample data.
"""

import numpy as np

# Sigmoid function
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss function
def CE_loss(y, t):
    """Compute binary cross-entropy loss."""
    return -np.mean(t * np.log(y + 1e-9) + (1 - t) * np.log(1 - y + 1e-9)) # Add 1e-9 to avoid numerical errors

# Mini-batch SGD for logistic regression
def mini_batch_sgd(X, t, batch_size=32, learning_rate=0.01, max_iter=100):
    """
    Train a logistic regression model using mini-batch SGD.

    Parameters:
    X : ndarray (N, D) - Input features
    t : ndarray (N, 1) - Target labels (0 or 1)
    batch_size : int - Number of samples per batch
    learning_rate : float - Learning rate for gradient descent
    max_iters : int - Number of training iterations

    Returns:
    w : ndarray (D, 1) - Optimized weight vector
    """
    N, d = X.shape # Number of samples and features
    w = np.random.randn(d) * 0.01 # Initialize weights with small random values
    
    for iteration in range(max_iter):
        indices = np.random.permutation(N) # Shuffle data
        X_shuffled, t_shuffled = X[indices], t[indices]
        
        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            t_batch = t_shuffled[i:i + batch_size]
            
            y_batch = sigmoid(X_batch @ w) # Compute predictions
            gradient = (X_batch.T @ (y_batch - t_batch)) / batch_size # Compute gradient
            w -= learning_rate * gradient # Update weights
            