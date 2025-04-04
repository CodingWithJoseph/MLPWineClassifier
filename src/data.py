import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine


def fetch_dataset():
    # Load the Wine dataset as a pandas DataFrame
    return load_wine(as_frame=True).frame


def test_train_split(dataset, test_percent=0.2, random_state=42):
    # Set seed for reproducible shuffling
    np.random.seed(random_state)

    # Shuffle the dataset randomly
    shuffled_dataset = np.random.permutation(dataset)
    num_examples = shuffled_dataset.shape[0]

    # Calculate number of test samples
    test_size = math.ceil(num_examples * test_percent)

    # Features for test set
    X_test = shuffled_dataset[:test_size, : -1]

    # Labels for test set
    y_test = shuffled_dataset[:test_size, -1:].astype(int).reshape((-1,))

    # Features for training set
    X = shuffled_dataset[test_size:, : -1]

    # Labels for training set
    y = shuffled_dataset[test_size:, -1:].astype(int).reshape((-1,))

    return X, X_test, y, y_test


if __name__ == '__main__':
    data = fetch_dataset()
    Xtr, Xte, ytr, yte = test_train_split(data)
    print(f"Training Data Shape: {Xtr.shape}")
    print(f"Training Labels Shape: {ytr.shape}")
    print(f"Test Data Shape: {Xte.shape}")
    print(f"Test Labels Shape: {yte.shape}")

    sns.pairplot(data, hue='target', palette='viridis')
    plt.show()
