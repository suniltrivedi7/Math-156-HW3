# Logistic Regression with Mini-Batch SGD - Write-Up

## Overview
This project implements **binary logistic regression** using **mini-batch stochastic gradient descent (SGD)**. The model is trained by minimizing the **cross-entropy loss function**, making it suitable for binary classification tasks.

## Mathematical Foundation

### Sigmoid Function
The logistic regression model maps inputs to probabilities using the sigmoid function:
$
\[
y = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}
\]
$
Where:
- \( w \) is the weight vector.
- \( x \) is the input feature vector.
- \( y \) is the predicted probability of class 1.

### Cross-Entropy Loss Function
To train the model, we minimize the cross-entropy loss:

\[
E(w) = - \sum_{n=1}^{N} \left[ t_n \log(y_n) + (1 - t_n) \log(1 - y_n) \right]
\]

Where:
- \( t_n \) is the true label (0 or 1) for sample \( n \).
- \( y_n \) is the predicted probability for sample \( n \).

### Mini-Batch SGD Optimization
The model updates weights using **mini-batch stochastic gradient descent**, computing gradients for small subsets (batches) of data instead of the entire dataset. The gradient of the loss function is given by:

\[
\nabla E(w) = \frac{1}{B} X^T (y - t)
\]

Where:
- \( B \) is the batch size.
- \( X \) is the batch feature matrix.
- \( y \) is the predicted probability vector.
- \( t \) is the true label vector.

Each iteration, the weights are updated using:

\[
w \leftarrow w - \eta \nabla E(w)
\]

Where \( \eta \) is the learning rate.

## Implementation Details

1. **Dataset Preparation**: The script generates synthetic data for training.
2. **Weight Initialization**: The weight vector is initialized to zeros.
3. **Training Loop**:
   - Shuffle the dataset at each iteration.
   - Divide the dataset into mini-batches.
   - Compute predictions using the sigmoid function.
   - Calculate the gradient of the loss function.
   - Update weights using the gradient.
4. **Convergence**: Training continues for a fixed number of iterations.

## Summary
This implementation efficiently trains a logistic regression model using mini-batch SGD. The use of **cross-entropy loss** ensures proper optimization, and **batch updates** improve efficiency over standard gradient descent.

### Future Improvements
- Implement adaptive learning rate methods like **Adam**.
- Add support for **L2 regularization**.
- Extend to multi-class classification using **softmax regression**.
