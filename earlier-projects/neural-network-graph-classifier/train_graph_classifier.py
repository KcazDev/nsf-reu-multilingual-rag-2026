"""
train_graph_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Trains a feedforward neural network to classify graphs as either
undirected or directed using flattened adjacency matrices.

Class labels:
    0 = undirected
    1 = directed
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import network
from graph_dataset import (
    DIRECTED,
    UNDIRECTED,
    create_datasets,
    generate_directed_matrix,
    generate_undirected_matrix,
    save_all_datasets_to_csv,
)



# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

NUMBER_OF_NODES = 4

TRAINING_SIZE = 200
VALIDATION_SIZE = 50
TEST_SIZE = 50

HIDDEN_NEURONS = 16

EPOCHS = 50
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0

EDGE_PROBABILITY = 0.5
RANDOM_SEED = 42

FIGURES_FOLDER = Path("figures")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def class_name(class_id: int) -> str:
    """Convert an integer class label into readable text."""

    if class_id == UNDIRECTED:
        return "Undirected"

    if class_id == DIRECTED:
        return "Directed"

    return "Unknown"


def plot_mse(mse_history: list[float]) -> None:
    """Create and save the mean squared error graph."""

    FIGURES_FOLDER.mkdir(exist_ok=True)

    epochs = range(1, len(mse_history) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, mse_history, marker="o", markersize=3)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Mean Squared Error by Epoch")
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_FOLDER / "training_mse.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"MSE graph saved to: {output_path}")


def plot_accuracy(accuracy_history: list[float]) -> None:
    """Create and save the test accuracy graph."""

    if not accuracy_history:
        print("No accuracy history was recorded.")
        return

    FIGURES_FOLDER.mkdir(exist_ok=True)

    epochs = range(1, len(accuracy_history) + 1)
    percentages = [
        accuracy * 100
        for accuracy in accuracy_history
    ]

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, percentages, marker="o", markersize=3)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy by Epoch")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_FOLDER / "test_accuracy.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Accuracy graph saved to: {output_path}")


def print_dataset_summary(
    training_data,
    validation_data,
    test_data
) -> None:
    """Print basic information about the generated dataset."""

    print("\nDataset Summary")
    print("---------------")
    print(f"Number of nodes: {NUMBER_OF_NODES}")
    print(f"Input neurons: {NUMBER_OF_NODES ** 2}")
    print(f"Training examples: {len(training_data)}")
    print(f"Validation examples: {len(validation_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Output neurons: 2")


def test_custom_examples(net: network.Network) -> None:
    """
    Generate new graphs and show the network's predictions.

    These examples are created after training and are separate from
    the original training data.
    """

    print("\nNew Graph Predictions")
    print("---------------------")

    examples = [
        (
            generate_undirected_matrix(
                NUMBER_OF_NODES,
                EDGE_PROBABILITY
            ),
            UNDIRECTED
        ),
        (
            generate_directed_matrix(
                NUMBER_OF_NODES,
                EDGE_PROBABILITY
            ),
            DIRECTED
        ),
        (
            generate_undirected_matrix(
                NUMBER_OF_NODES,
                EDGE_PROBABILITY
            ),
            UNDIRECTED
        ),
        (
            generate_directed_matrix(
                NUMBER_OF_NODES,
                EDGE_PROBABILITY
            ),
            DIRECTED
        ),
    ]

    for example_number, (matrix, actual_class) in enumerate(
        examples,
        start=1
    ):
        input_vector = matrix.reshape(
            NUMBER_OF_NODES ** 2,
            1
        )

        prediction, scores = net.predict_with_scores(input_vector)

        print(f"\nExample {example_number}")
        print(matrix.astype(int))

        print(f"Actual class: {class_name(actual_class)}")
        print(f"Predicted class: {class_name(prediction)}")

        print(
            "Output scores: "
            f"Undirected = {scores[0, 0]:.4f}, "
            f"Directed = {scores[1, 0]:.4f}"
        )


def print_final_results(
    net: network.Network,
    test_data,
    mse_history: list[float],
    accuracy_history: list[float]
) -> None:
    """Print the final training and testing results."""

    correct_predictions = net.evaluate(test_data)
    total_predictions = len(test_data)

    final_accuracy = (
        correct_predictions / total_predictions
    )

    test_mse = net.mean_squared_error(
        test_data,
        labels_are_integers=True
    )

    print("\nFinal Results")
    print("-------------")

    print(
        f"Correct test predictions: "
        f"{correct_predictions} / {total_predictions}"
    )

    print(f"Final test accuracy: {final_accuracy:.2%}")
    print(f"Final training MSE: {mse_history[-1]:.6f}")
    print(f"Final test MSE: {test_mse:.6f}")

    if accuracy_history:
        best_accuracy = max(accuracy_history)
        best_epoch = accuracy_history.index(best_accuracy) + 1

        print(
            f"Best test accuracy: "
            f"{best_accuracy:.2%} at epoch {best_epoch}"
        )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:
    """Generate data, train the network, and display results."""

    print("Neural Network Graph Classifier")
    print("===============================")

    print("\nCreating graph datasets...")

    training_data, validation_data, test_data = create_datasets(
        number_of_nodes=NUMBER_OF_NODES,
        training_size=TRAINING_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        edge_probability=EDGE_PROBABILITY,
        random_seed=RANDOM_SEED
    )
    save_all_datasets_to_csv(
        training_data=training_data,
        validation_data=validation_data,
        test_data=test_data,
        number_of_nodes=NUMBER_OF_NODES,
        output_folder="data"
    )

    print_dataset_summary(
        training_data,
        validation_data,
        test_data
    )

    input_neurons = NUMBER_OF_NODES ** 2

    architecture = [
        input_neurons,
        HIDDEN_NEURONS,
        2
    ]

    print("\nNetwork Settings")
    print("----------------")
    print(f"Architecture: {architecture}")
    print(f"Epochs: {EPOCHS}")
    print(f"Mini-batch size: {MINI_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    np.random.seed(RANDOM_SEED)

    net = network.Network(architecture)

    print("\nBeginning training...\n")

    mse_history, accuracy_history = net.SGD(
        training_data=training_data,
        epochs=EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        eta=LEARNING_RATE,
        test_data=test_data
    )

    print_final_results(
        net,
        test_data,
        mse_history,
        accuracy_history
    )

    plot_mse(mse_history)
    plot_accuracy(accuracy_history)

    test_custom_examples(net)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()