# MLP: Multilayer Perceptron

A built from scratch python multilayer perceptron for classification. Achieves 96-100% accuracy on scikit-learn datasets.

## How it works

The data is split into training and validation then formed into random batches in [`dataloader.py`](dataloader.py), then fed through the network. The network uses `ReLu` for nonlinear activations and `Softmax` for the final output layer and a cross entropy loss function. The model is trained with stochastic gradient descent using backpropagation to compute the gradient. The weights and biases are stored in the central class `NeuralNetwork`. Metrics like accuracy per epoch, precision, recall, and Fscore are printed to the terminal after training.  

| File | Responsibility |
|---|---|
| [`activations.py`](activations.py) | Contains `ReLu` and `Softmax`, and their gradients |
| [`dataloader.py`](dataloader.py) | formats data, splits into validation and training, and creates batches for SGD|
| [`losses.py`](losses.py) | Contains `categorical_cross_entropy` loss function |
| [`network.py`](network.py) | Contains the central `NeuralNetwork` class with the `gradient_descent`, `feed_forward`, `back_prop`, and  functions to evaluate the performance of the network|
| [`train.py`](train.py) | Contains the main function `train` that trains the network |
| [`test_backprop.py`](test_backprop.py) | tests backpropagation function against finite difference method |
| [`test_network.py`](test_network.py) | tests the network on scikit-learn datasets |

## Usage

```python
    import numpy as np
    import dataloader as dl
    from sklearn.datasets import load_breast_cancer

    X, Y = dl.format_data(load_breast_cancer()) #call format_data in dataloader to load the data
    layer_sizes = [X.shape[1], 20, 20, Y.shape[1]] #initialize hidden layers in the list (example: two hidden layers of 20 neurons each)

    nn = NeuralNetwork(layer_sizes, 32) #initialize neural network with batch_size = 32
    train.train(X, Y, nn) #train
```

## Testing

```bash
source venv/bin/activate
python3 -m pytest test_network.py -v -s
```
