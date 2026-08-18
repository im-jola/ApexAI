"""
Custom Feedforward Neural Network Implementation.
Supports ReLU and Sigmoid activations, mutation, crossover, and serialization.
"""

import numpy as np


class NeuralNetwork:
    """Feedforward Neural Network with 1 hidden layer."""

    def __init__(self, input_size=7, hidden_size=6, output_size=4):
        # Initialize random weights and biases scaled [-1, 1]
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = np.random.uniform(-1, 1, (1, hidden_size))

        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b2 = np.random.uniform(-1, 1, (1, output_size))

        # Cached activations for telemetry visualizer
        self.last_inputs = np.zeros(input_size)
        self.last_hidden = np.zeros(hidden_size)
        self.last_outputs = np.zeros(output_size)

    @staticmethod
    def relu(x):
        """Rectified Linear Unit activation function."""
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function bounded [-500, 500] to prevent overflow."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs):
        """Executes forward propagation pass."""
        X = np.array(inputs).reshape(1, -1)
        self.last_inputs = np.array(inputs)

        h_raw = np.dot(X, self.w1) + self.b1
        self.last_hidden = self.relu(h_raw)[0]

        out_raw = np.dot(self.last_hidden.reshape(1, -1), self.w2) + self.b2
        self.last_outputs = self.sigmoid(out_raw)[0]

        return self.last_outputs

    def mutate(self, rate=0.1, scale=0.2):
        """Applies random Gaussian mutation to weights and biases."""
        for matrix in [self.w1, self.b1, self.w2, self.b2]:
            mask = np.random.rand(*matrix.shape) < rate
            matrix += mask * np.random.normal(0, scale, size=matrix.shape)

    def save(self, filename="best_car.npz"):
        """Saves weights and biases to compressed numpy archive."""
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(self, filename="best_car.npz"):
        """Loads weights and biases from file."""
        data = np.load(filename)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]

    def crossover(self, other):
        """Executes uniform crossover with another parent network."""
        child = NeuralNetwork()
        for attr in ["w1", "b1", "w2", "b2"]:
            mask = np.random.rand(*getattr(self, attr).shape) > 0.5
            setattr(child, attr, np.where(mask, getattr(self, attr), getattr(other, attr)))
        return child