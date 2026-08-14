import pytest
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import train
import dataloader as dl
from network import NeuralNetwork
import numpy as np


def test_on_iris():
    X, Y = dl.format_data(load_iris())
    layer_sizes = [X.shape[1], 10, 10, Y.shape[1]]

    nn = NeuralNetwork(layer_sizes, 32)
    train.train(X, Y, nn)


def test_on_wine():
    X, Y = dl.format_data(load_wine())
    layer_sizes = [X.shape[1], 20, 20, Y.shape[1]]

    nn = NeuralNetwork(layer_sizes, 32)
    train.train(X, Y, nn)


def test_on_breast_cancer():
    X, Y = dl.format_data(load_breast_cancer())
    layer_sizes = [X.shape[1], 20, 20, Y.shape[1]]

    nn = NeuralNetwork(layer_sizes, 32)
    train.train(X, Y, nn)


def test_on_digits():
    data = load_digits()
    X, Y = dl.format_data(load_digits())
    layer_sizes = [X.shape[1], 20, 20, Y.shape[1]]

    nn = NeuralNetwork(layer_sizes, 32)
    train.train(X, Y, nn)
