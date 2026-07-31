"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.

Adapted from Michael Nielsen's Neural Networks and Deep Learning.
Modified to record mean squared error and test accuracy after each epoch.
"""

# Standard library
import random
import time

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in each layer.

        Example:
            [16, 8, 2]

        means:
            16 input neurons
            8 hidden neurons
            2 output neurons
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [
            np.random.randn(y, 1)
            for y in sizes[1:]
        ]

        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a):
        """Return the output of the network for input ``a``."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        test_data=None
    ):
        """
        Train the network using mini-batch stochastic gradient descent.

        Returns:
            mse_history:
                Training mean squared error after each epoch.

            accuracy_history:
                Test accuracy after each epoch.
        """

        if not training_data:
            raise ValueError("training_data cannot be empty.")

        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be greater than zero.")

        if eta <= 0:
            raise ValueError("eta must be greater than zero.")

        n = len(training_data)

        if test_data:
            n_test = len(test_data)
        else:
            n_test = 0

        mse_history = []
        accuracy_history = []

        for epoch in range(epochs):
            start_time = time.time()

            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            training_mse = self.mean_squared_error(
                training_data,
                labels_are_integers=False
            )

            mse_history.append(training_mse)

            elapsed_time = time.time() - start_time

            if test_data:
                correct = self.evaluate(test_data)
                accuracy = correct / n_test
                accuracy_history.append(accuracy)

                print(
                    "Epoch {0}: "
                    "{1} / {2} correct, "
                    "Accuracy: {3:.2%}, "
                    "MSE: {4:.6f}, "
                    "Time: {5:.2f} seconds".format(
                        epoch + 1,
                        correct,
                        n_test,
                        accuracy,
                        training_mse,
                        elapsed_time
                    )
                )

            else:
                print(
                    "Epoch {0}: "
                    "MSE: {1:.6f}, "
                    "Time: {2:.2f} seconds".format(
                        epoch + 1,
                        training_mse,
                        elapsed_time
                    )
                )

        return mse_history, accuracy_history

    def update_mini_batch(self, mini_batch, eta):
        """
        Update weights and biases using gradient descent
        on one mini-batch.
        """

        if not mini_batch:
            return

        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [
                nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)
            ]

            nabla_w = [
                nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)
            ]

        learning_scale = eta / len(mini_batch)

        self.weights = [
            w - learning_scale * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - learning_scale * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Return gradients for the cost function.

        Returns:
            nabla_b:
                Gradients for biases.

            nabla_w:
                Gradients for weights.
        """

        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        # Feedforward
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # Output-layer backward pass
        delta = (
            self.cost_derivative(activations[-1], y)
            * sigmoid_prime(zs[-1])
        )

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(
            delta,
            activations[-2].transpose()
        )

        # Hidden-layer backward pass
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)

            delta = (
                np.dot(
                    self.weights[-layer + 1].transpose(),
                    delta
                )
                * sp
            )

            nabla_b[-layer] = delta

            nabla_w[-layer] = np.dot(
                delta,
                activations[-layer - 1].transpose()
            )

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Return the number of correctly classified test examples.

        Test labels must be integers:
            0 = undirected
            1 = directed
        """

        test_results = [
            (
                int(np.argmax(self.feedforward(x))),
                int(y)
            )
            for x, y in test_data
        ]

        return sum(
            int(prediction == actual)
            for prediction, actual in test_results
        )

    def mean_squared_error(
        self,
        data,
        labels_are_integers=False
    ):
        """
        Return the average mean squared error for a dataset.

        Training labels are normally one-hot vectors.
        Test labels are normally integer class values.
        """

        if not data:
            raise ValueError("Cannot calculate MSE on an empty dataset.")

        total_error = 0.0

        for x, y in data:
            if labels_are_integers:
                target = self.vectorized_result(int(y))
            else:
                target = y

            prediction = self.feedforward(x)

            total_error += np.mean(
                (prediction - target) ** 2
            )

        return total_error / len(data)

    def predict(self, x):
        """Return the predicted class for one input vector."""
        output = self.feedforward(x)
        return int(np.argmax(output))

    def predict_with_scores(self, x):
        """
        Return the predicted class and the two output activations.
        """
        output = self.feedforward(x)

        predicted_class = int(np.argmax(output))

        return predicted_class, output

    def vectorized_result(self, class_id):
        """
        Convert an integer class label into a one-hot column vector.

        0 becomes:
            [[1.0],
             [0.0]]

        1 becomes:
            [[0.0],
             [1.0]]
        """

        if class_id < 0 or class_id >= self.sizes[-1]:
            raise ValueError(
                "class_id must match one of the output neurons."
            )

        result = np.zeros((self.sizes[-1], 1))
        result[class_id] = 1.0

        return result

    def cost_derivative(self, output_activations, y):
        """
        Return the derivative of the quadratic cost function.
        """
        return output_activations - y


def sigmoid(z):
    """The sigmoid activation function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid activation function."""
    s = sigmoid(z)
    return s * (1.0 - s)