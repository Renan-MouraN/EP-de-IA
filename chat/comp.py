import os
import csv
import logging
from itertools import product

import numpy as np

# Logger configuration
def setup_logger(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

# ===== Configuration =====
CONFIG = {
    "X_PATH": "chat/X.npy",             # Features file (N, 10, 12, 1)
    "Y_PATH": "chat/Y_classe.npy",      # Labels file (N, 26)
    "OUTPUT_DIR": "output_comp",        # Output directory
    "TEST_RATIO": 0.2,             # Proportion for test split
    "SEED": 42,                    # Random seed
    "CV_FOLDS": 5,                 # Cross-validation folds
    
    # Grid search parameters
    "GRID_HIDDEN": [75, 100],
    "GRID_LR": [0.01, 0.1],
    "GRID_EPOCHS": [150],
    "GRID_PATIENCE": [15]
}
# =========================

class MLP:
    """
    Simple two-layer neural network with tanh hidden activation and softmax output.
    Supports training by gradient descent with optional early stopping.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, learning_rate: float = 0.01, seed: int = None, patience: int = None, max_epochs: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = learning_rate
        self.patience = patience
        self.max_epochs = max_epochs

        # Initialize weights including bias term
        self.W1 = np.random.randn(n_hidden, n_inputs + 1) * 0.1
        self.W2 = np.random.randn(n_outputs, n_hidden + 1) * 0.1

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _tanh_derivative(y: np.ndarray) -> np.ndarray:
        return 1.0 - np.square(y)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        # Hidden layer
        X_bias = np.append(X, 1.0)
        z = self._tanh(self.W1.dot(X_bias))

        # Output layer
        z_bias = np.append(z, 1.0)
        y = self._softmax(self.W2.dot(z_bias))

        # Cache for backprop
        self._cache = {"X_bias": X_bias, "z": z, "z_bias": z_bias, "y": y}
        return y

    def _compute_gradients(self, T: np.ndarray) -> tuple:
        # Retrieve forward pass cache
        Xb = self._cache["X_bias"]
        z = self._cache["z"]
        Zb = self._cache["z_bias"]
        Y = self._cache["y"]

        # Output error (softmax + cross-entropy)
        delta_out = Y - T

        # Hidden error
        grad_z = self._tanh_derivative(z)
        delta_hidden = grad_z * (self.W2[:, :-1].T.dot(delta_out))

        # Weight gradients
        grad_W2 = np.outer(delta_out, Zb)
        grad_W1 = np.outer(delta_hidden, Xb)

        return grad_W1, grad_W2

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray = None, Y_val: np.ndarray = None) -> tuple:
        """
        Train the network with gradient descent.
        Returns final training loss and validation loss.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            for X, T in zip(X_train, Y_train):
                # Forward
                self._forward(X)
                # Compute gradients
                grad_W1, grad_W2 = self._compute_gradients(T)
                # Update weights
                self.W1 -= self.lr * grad_W1
                self.W2 -= self.lr * grad_W2
                # Loss (cross-entropy)
                total_loss += -np.sum(T * np.log(self._cache['y'] + 1e-15))

            # Early stopping on validation set
            if X_val is not None:
                val_loss = self.evaluate_loss(X_val, Y_val)
                if self.patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_weights = (self.W1.copy(), self.W2.copy())
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        self.W1, self.W2 = best_weights
                        break
            else:
                val_loss = None

        return total_loss, val_loss

    def evaluate_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute average cross-entropy loss on given data."""
        losses = []
        for x, t in zip(X, Y):
            y = self._forward(x)
            losses.append(-np.sum(t * np.log(y + 1e-15)))
        return float(np.mean(losses))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one-hot encoded class vector."""
        y = self._forward(X)
        pred = np.zeros_like(y)
        pred[np.argmax(y)] = 1
        return pred

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in X])


# === Data utilities ===

def load_and_preprocess(x_path: str, y_path: str) -> tuple:
    """Load .npy arrays, flatten features, normalize, encode labels to {-1, 1}."""
    X = np.load(x_path)  # shape (N, 10, 12, 1)
    Y = np.load(y_path)  # shape (N, 26)

    N = X.shape[0]
    X = X.reshape(N, -1)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Add bias term in feature matrix
    X = np.hstack([np.ones((N, 1)), X])

    # Encode labels {>0: 1, else: -1}
    Y = np.where(Y > 0, 1, -1)
    return X, Y


def train_test_split(X: np.ndarray, Y: np.ndarray, test_ratio: float, seed: int) -> tuple:
    """Split data into train and test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def evaluate_and_save(model: MLP, X_test: np.ndarray, Y_test: np.ndarray, prefix: str) -> tuple:
    """Compute confusion matrix, accuracy, and save results to files."""
    predictions = model.predict_batch(X_test)
    n_classes = Y_test.shape[1]
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_vec, pred_vec in zip(Y_test, predictions):
        true_idx = np.argmax(true_vec)
        pred_idx = np.argmax(pred_vec)
        cm[true_idx, pred_idx] += 1

    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    np.savetxt(f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    accuracy = np.mean((predictions == Y_test).all(axis=1))
    with open(f"{prefix}_accuracy.txt", "w") as f:
        f.write(f"accuracy: {accuracy:.4f}\n")

    return cm, accuracy


def cross_validate(X: np.ndarray, Y: np.ndarray, params: dict, folds: int, seed: int) -> float:
    """Manual k-fold cross-validation returning average accuracy."""
    N = len(X)
    indices = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(indices)

    fold_sizes = np.full(folds, N // folds, dtype=int)
    fold_sizes[: N % folds] += 1

    current = 0
    accuracies = []
    for size in fold_sizes:
        start, stop = current, current + size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))

        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        model = MLP(n_inputs=X.shape[1], n_hidden=params['hidden'], n_outputs=Y.shape[1], learning_rate=params['lr'], seed=seed, patience=params['patience'], max_epochs=params['epochs'])
        model.train(X_tr, Y_tr, X_val, Y_val)
        preds = model.predict_batch(X_val)
        acc = np.mean((preds == Y_val).all(axis=1))
        accuracies.append(acc)
        current = stop

    return float(np.mean(accuracies))


def grid_search(X: np.ndarray, Y: np.ndarray, config: dict) -> dict:
    """Perform grid search over hyperparameters and save results."""
    results = []
    param_grid = product(config['GRID_HIDDEN'], config['GRID_LR'], config['GRID_EPOCHS'], config['GRID_PATIENCE'])

    for hidden, lr, epochs, patience in param_grid:
        params = {'hidden': hidden, 'lr': lr, 'epochs': epochs, 'patience': patience}
        cv_score = cross_validate(X, Y, params, config['CV_FOLDS'], config['SEED'])
        results.append({**params, 'cv_accuracy': cv_score})

    # Save grid results
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    keys = results[0].keys()
    with open(os.path.join(config['OUTPUT_DIR'], 'grid_search.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    best = max(results, key=lambda r: r['cv_accuracy'])
    logging.info(f"Best CV configuration: {best}")
    return best


def main():
    setup_logger()

    # Load and preprocess data
    X, Y = load_and_preprocess(CONFIG['X_PATH'], CONFIG['Y_PATH'])

    # Reserve last samples for test (consistent split)
    X_train, Y_train = X[:-130], Y[:-130]
    X_test, Y_test = X[-130:], Y[-130:]

    # Hyperparameter tuning
    best_cfg = grid_search(X_train, Y_train, CONFIG)

    # Final training with best parameters
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    prefix = os.path.join(CONFIG['OUTPUT_DIR'], 'dataset_best')

    model = MLP(n_inputs=X_train.shape[1], n_hidden=best_cfg['hidden'], n_outputs=Y_train.shape[1], learning_rate=best_cfg['lr'], seed=CONFIG['SEED'], patience=best_cfg['patience'], max_epochs=best_cfg['epochs'])

    # Save initial weights
    np.save(f"{prefix}_weights_initial_W1.npy", model.W1)
    np.save(f"{prefix}_weights_initial_W2.npy", model.W2)

    # Train model
    model.train(X_train, Y_train)

    # Save final weights
    np.save(f"{prefix}_weights_final_W1.npy", model.W1)
    np.save(f"{prefix}_weights_final_W2.npy", model.W2)

    # Evaluation on test set
    cm, accuracy = evaluate_and_save(model, X_test, Y_test, prefix)
    logging.info(f"Test accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
