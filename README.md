# MLP From Scratch

A Multi-Layer Perceptron implementation built entirely with NumPy. No PyTorch, TensorFlow, or sklearn models — just matrix math and backpropagation.

## Features

- Configurable architecture (any number of hidden layers)
- ReLU activation for hidden layers, Softmax output for multi-class classification
- Mini-batch Stochastic Gradient Descent
- He weight initialization
- Early stopping with patience
- Training history tracking (loss + accuracy)
- Reproducible results via `random_state`
- Input validation with clear error messages

## Quick Start

```python
from mlp import MLP

model = MLP(
    architecture=[16, 32, 16, 7],
    learning_rate=0.01,
    batch_size=256,
    random_state=42
)

model.summary()
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=1000, patience=15)
predictions = model.predict(X_test)
```

## Installation

Requires [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/appuk45/mlp-from-scratch.git
cd mlp-from-scratch
uv sync
```

To run the example notebook:

```bash
uv sync --extra example
```

## API

### `MLP(architecture, learning_rate=0.01, batch_size=128, random_state=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `architecture` | `list[int]` | Layer sizes, e.g. `[16, 32, 16, 7]` (input, hidden..., output) |
| `learning_rate` | `float` | Step size for gradient descent |
| `batch_size` | `int` | Samples per mini-batch |
| `random_state` | `int \| None` | Seed for reproducibility |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X, y, X_val, y_val, epochs, patience, verbose)` | `self` | Train the model with mini-batch SGD and optional early stopping |
| `predict(X)` | `np.ndarray` | Predicted class labels |
| `predict_proba(X)` | `np.ndarray` | Class probabilities |
| `summary()` | `None` | Print architecture and parameter count |

### Training History

After calling `fit()`, access `model.history`:

```python
model.history['train_loss']  # list of training losses per epoch
model.history['val_loss']    # list of validation losses per epoch
model.history['train_acc']   # list of training accuracies per epoch
model.history['val_acc']     # list of validation accuracies per epoch
```

## Example

See [`example.ipynb`](example.ipynb) for a full demo on the [Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (13,611 samples, 16 features, 7 classes). Achieves ~92% test accuracy.

## Limitations

- Multi-class classification only (softmax + categorical cross-entropy)
- SGD optimizer only (no Adam, RMSprop)
- No regularization (dropout, L2)
- CPU only

## Roadmap

- [ ] Binary classification and regression support
- [ ] Configurable hidden layer activations (sigmoid, tanh)
- [ ] Additional optimizers (Adam)
- [ ] L2 regularization
- [ ] Save/load model
