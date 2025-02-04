# Logistic Regression with Mini-Batch SGD

This project implements **Binary Logistic Regression** trained using **Mini-Batch Stochastic Gradient Descent (SGD)**. It optimizes weights by minimizing the **cross-entropy loss function**.

## Features

- Implements logistic regression from scratch.
- Uses mini-batch SGD for training.
- Includes configurable hyperparameters (batch size, learning rate, max iterations).
- Demonstrates usage with synthetic data.

## Installation

Ensure you have Python installed with NumPy:

```sh
pip install numpy
```

## Usage

Run the script to train the logistic regression model on synthetic data:

```sh
python logistic_regression_sgd.py
```

## Hyperparameters

Modify the following parameters in `mini_batch_sgd` function:

- `batch_size`: Number of samples per mini-batch (default: `32`).
- `learning_rate`: Step size for weight updates (default: `0.01`).
- `max_iters`: Maximum training iterations (default: `100`).

## Output

The script prints the optimized weight vector after training.

## License

This project is open-source and available for modification.

