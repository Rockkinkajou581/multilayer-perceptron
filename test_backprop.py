import numpy as np
import pytest

from network import NeuralNetwork
from losses import categorical_cross_entropy


def gradient_check(network, X, Y_true, epsilon=1e-5, tolerance=1e-4):
    """
    Compares analytical gradients (from network.back_prop) against
    numerical finite-difference gradients, for every weight and bias.
    Prints a relative error per parameter; anything above `tolerance`
    likely indicates a bug in back_prop.
    """
    # Run the analytical backward pass once, to get network.grad_W / grad_b
    network.feed_forward(X)
    network.back_prop(X, Y_true)
    analytical_grad_W = [g.copy() for g in network.grad_W]
    analytical_grad_b = [g.copy() for g in network.grad_b]

    all_passed = True

    # Check every weight matrix
    for l in range(len(network.weights)):
        W = network.weights[l]
        numerical_grad = np.zeros_like(W)

        # iterate over every entry of this weight matrix (fine for small test networks)
        it = np.nditer(W, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            original_value = W[idx]

            W[idx] = original_value + epsilon
            y_hat_plus = network.feed_forward(X)
            loss_plus = categorical_cross_entropy(network.activation[-1], Y_true)

            W[idx] = original_value - epsilon
            y_hat_minus = network.feed_forward(X)
            loss_minus = categorical_cross_entropy(network.activation[-1], Y_true)

            W[idx] = original_value  # restore

            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        analytical = analytical_grad_W[l]
        rel_error = np.linalg.norm(numerical_grad - analytical) / (
            np.linalg.norm(numerical_grad) + np.linalg.norm(analytical) + 1e-12
        )
        status = "OK" if rel_error < tolerance else "MISMATCH"
        print(f"Layer {l} weights: relative error = {rel_error:.2e}  [{status}]")
        if rel_error >= tolerance:
            all_passed = False

    # Check every bias vector
    for l in range(len(network.bias)):
        b = network.bias[l]
        numerical_grad = np.zeros_like(b)

        it = np.nditer(b, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            original_value = b[idx]

            b[idx] = original_value + epsilon
            y_hat_plus = network.feed_forward(X)
            loss_plus = categorical_cross_entropy(network.activation[-1], Y_true)

            b[idx] = original_value - epsilon
            y_hat_minus = network.feed_forward(X)
            loss_minus = categorical_cross_entropy(network.activation[-1], Y_true)

            b[idx] = original_value  # restore

            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        analytical = analytical_grad_b[l]
        rel_error = np.linalg.norm(numerical_grad - analytical) / (
            np.linalg.norm(numerical_grad) + np.linalg.norm(analytical) + 1e-12
        )
        status = "OK" if rel_error < tolerance else "MISMATCH"
        print(f"Layer {l} biases:  relative error = {rel_error:.2e}  [{status}]")
        if rel_error >= tolerance:
            all_passed = False

    print("\nALL CHECKS PASSED" if all_passed else "\nSOME CHECKS FAILED — see mismatches above")
    return all_passed


@pytest.fixture
def small_network_and_data():
    np.random.seed(0)

    # tiny network: small enough that checking every single parameter is fast
    layer_sizes = [4, 5, 3]  # 4 input features, 5 hidden units, 3 classes
    batch_size = 6
    net = NeuralNetwork(layer_sizes, batch_size)

    X = np.random.randn(batch_size, layer_sizes[0])

    # random one-hot labels
    Y_true = np.zeros((batch_size, layer_sizes[-1]))
    labels = np.random.randint(0, layer_sizes[-1], size=batch_size)
    Y_true[np.arange(batch_size), labels] = 1

    return net, X, Y_true


def test_backprop_matches_numerical_gradient(small_network_and_data):
    net, X, Y_true = small_network_and_data
    assert gradient_check(net, X, Y_true)


def test_backprop_output_shapes(small_network_and_data):
    net, X, Y_true = small_network_and_data
    net.back_prop(X, Y_true)

    for W, grad_W in zip(net.weights, net.grad_W):
        assert grad_W.shape == W.shape

    for b, grad_b in zip(net.bias, net.grad_b):
        assert grad_b.shape == b.shape
