"""
graph_dataset.py
~~~~~~~~~~~~~~~~

Creates training, validation, and test datasets made from
directed and undirected adjacency matrices.

Class labels:
    0 = undirected
    1 = directed

The generated datasets can also be saved as CSV files.
"""

import csv
import random
from pathlib import Path

import numpy as np


UNDIRECTED = 0
DIRECTED = 1


def generate_undirected_matrix(
    number_of_nodes: int,
    edge_probability: float = 0.5
) -> np.ndarray:
    """
    Generate a random undirected adjacency matrix.

    An undirected adjacency matrix is symmetric:
        A[i][j] = A[j][i]
    """

    matrix = np.zeros(
        (number_of_nodes, number_of_nodes),
        dtype=float
    )

    for row in range(number_of_nodes):
        for column in range(row + 1, number_of_nodes):
            edge_exists = (
                1.0
                if random.random() < edge_probability
                else 0.0
            )

            matrix[row, column] = edge_exists
            matrix[column, row] = edge_exists

    return matrix


def generate_directed_matrix(
    number_of_nodes: int,
    edge_probability: float = 0.5
) -> np.ndarray:
    """
    Generate a random directed adjacency matrix.

    Each possible direction is generated independently.
    Symmetric matrices are rejected so directed examples do not
    accidentally look like undirected examples.
    """

    while True:
        matrix = np.zeros(
            (number_of_nodes, number_of_nodes),
            dtype=float
        )

        for row in range(number_of_nodes):
            for column in range(number_of_nodes):
                if row != column:
                    matrix[row, column] = (
                        1.0
                        if random.random() < edge_probability
                        else 0.0
                    )

        if not np.array_equal(matrix, matrix.transpose()):
            return matrix


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a square adjacency matrix into a column vector.

    A 4 x 4 matrix becomes a 16 x 1 vector.
    """

    return matrix.reshape(matrix.size, 1)


def vectorized_label(class_id: int) -> np.ndarray:
    """
    Convert a class ID into a two-output one-hot vector.

    0 = undirected = [[1.0], [0.0]]
    1 = directed   = [[0.0], [1.0]]
    """

    if class_id not in (UNDIRECTED, DIRECTED):
        raise ValueError(
            "class_id must be 0 for undirected or 1 for directed."
        )

    label = np.zeros((2, 1))
    label[class_id] = 1.0

    return label


def create_example(
    number_of_nodes: int,
    class_id: int,
    training_format: bool,
    edge_probability: float = 0.5
):
    """
    Create one graph-classification example.

    Training data uses one-hot labels.
    Validation and test data use integer labels.
    """

    if class_id == UNDIRECTED:
        matrix = generate_undirected_matrix(
            number_of_nodes,
            edge_probability
        )

    elif class_id == DIRECTED:
        matrix = generate_directed_matrix(
            number_of_nodes,
            edge_probability
        )

    else:
        raise ValueError(
            "class_id must be 0 for undirected or 1 for directed."
        )

    input_vector = flatten_matrix(matrix)

    if training_format:
        label = vectorized_label(class_id)
    else:
        label = class_id

    return input_vector, label


def create_balanced_dataset(
    number_of_nodes: int,
    dataset_size: int,
    training_format: bool,
    edge_probability: float = 0.5
):
    """
    Create a balanced dataset with approximately equal numbers
    of directed and undirected graphs.
    """

    if number_of_nodes < 2:
        raise ValueError(
            "number_of_nodes must be at least 2."
        )

    if dataset_size < 2:
        raise ValueError(
            "dataset_size must be at least 2."
        )

    examples = []

    for index in range(dataset_size):
        class_id = (
            UNDIRECTED
            if index % 2 == 0
            else DIRECTED
        )

        example = create_example(
            number_of_nodes=number_of_nodes,
            class_id=class_id,
            training_format=training_format,
            edge_probability=edge_probability
        )

        examples.append(example)

    random.shuffle(examples)

    return examples


def create_datasets(
    number_of_nodes: int = 4,
    training_size: int = 200,
    validation_size: int = 50,
    test_size: int = 50,
    edge_probability: float = 0.5,
    random_seed: int | None = 42
):
    """
    Create training, validation, and test datasets.

    Returns:
        training_data:
            List of (input_vector, one_hot_label)

        validation_data:
            List of (input_vector, integer_label)

        test_data:
            List of (input_vector, integer_label)
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    training_data = create_balanced_dataset(
        number_of_nodes=number_of_nodes,
        dataset_size=training_size,
        training_format=True,
        edge_probability=edge_probability
    )

    validation_data = create_balanced_dataset(
        number_of_nodes=number_of_nodes,
        dataset_size=validation_size,
        training_format=False,
        edge_probability=edge_probability
    )

    test_data = create_balanced_dataset(
        number_of_nodes=number_of_nodes,
        dataset_size=test_size,
        training_format=False,
        edge_probability=edge_probability
    )

    return training_data, validation_data, test_data


def get_integer_label(label) -> int:
    """
    Convert either a one-hot label or integer label
    into a regular integer class ID.
    """

    if isinstance(label, np.ndarray):
        return int(np.argmax(label))

    return int(label)


def class_name(class_id: int) -> str:
    """Convert a class ID into readable text."""

    if class_id == UNDIRECTED:
        return "undirected"

    if class_id == DIRECTED:
        return "directed"

    return "unknown"


def save_dataset_to_csv(
    dataset,
    filename: str | Path,
    number_of_nodes: int
) -> None:
    """
    Save one dataset to a CSV file.

    Each row contains:
        example_id
        matrix values
        numeric label
        readable graph type
    """

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    matrix_headers = [
        f"A_{row}_{column}"
        for row in range(number_of_nodes)
        for column in range(number_of_nodes)
    ]

    headers = [
        "example_id",
        *matrix_headers,
        "label",
        "graph_type"
    ]

    with filename.open(
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for example_id, (input_vector, label) in enumerate(
            dataset,
            start=1
        ):
            flattened_values = (
                input_vector
                .reshape(-1)
                .astype(int)
                .tolist()
            )

            integer_label = get_integer_label(label)

            writer.writerow([
                example_id,
                *flattened_values,
                integer_label,
                class_name(integer_label)
            ])

    print(f"Dataset saved to: {filename}")


def save_all_datasets_to_csv(
    training_data,
    validation_data,
    test_data,
    number_of_nodes: int,
    output_folder: str | Path = "data"
) -> None:
    """
    Save the training, validation, and test datasets
    as three separate CSV files.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    save_dataset_to_csv(
        dataset=training_data,
        filename=output_folder / "training_dataset.csv",
        number_of_nodes=number_of_nodes
    )

    save_dataset_to_csv(
        dataset=validation_data,
        filename=output_folder / "validation_dataset.csv",
        number_of_nodes=number_of_nodes
    )

    save_dataset_to_csv(
        dataset=test_data,
        filename=output_folder / "test_dataset.csv",
        number_of_nodes=number_of_nodes
    )


def print_sample_matrices(
    number_of_nodes: int = 4,
    edge_probability: float = 0.5
):
    """Print example directed and undirected matrices."""

    undirected_matrix = generate_undirected_matrix(
        number_of_nodes,
        edge_probability
    )

    directed_matrix = generate_directed_matrix(
        number_of_nodes,
        edge_probability
    )

    print("\nSample undirected adjacency matrix:")
    print(undirected_matrix.astype(int))

    print("\nSymmetric:")
    print(
        np.array_equal(
            undirected_matrix,
            undirected_matrix.transpose()
        )
    )

    print("\nSample directed adjacency matrix:")
    print(directed_matrix.astype(int))

    print("\nSymmetric:")
    print(
        np.array_equal(
            directed_matrix,
            directed_matrix.transpose()
        )
    )


if __name__ == "__main__":
    print_sample_matrices()

    training_data, validation_data, test_data = create_datasets()

    print("\nDataset Summary")
    print("----------------")
    print(f"Training examples: {len(training_data)}")
    print(f"Validation examples: {len(validation_data)}")
    print(f"Test examples: {len(test_data)}")

    save_all_datasets_to_csv(
        training_data=training_data,
        validation_data=validation_data,
        test_data=test_data,
        number_of_nodes=4,
        output_folder="data"
    )