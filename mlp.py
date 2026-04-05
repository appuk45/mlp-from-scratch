import numpy as np


class ActivationFunctions:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class MLP:
    def __init__(
        self,
        architecture: list[int],
        learning_rate: float = 0.01,
        batch_size: int = 128,
        random_state: int | None = None,
    ) -> None:
        self.architecture = architecture
        self.num_layers = len(architecture) - 1
        self.n_classes = self.architecture[-1]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parameters = {}
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)  # local rng
        self._initialize_parameters()

    def _initialize_parameters(self):
        for l in range(1, self.num_layers + 1):
            n_in = self.architecture[l - 1]
            n_out = self.architecture[l]

            # He initialization for weights
            self.parameters[f"W{l}"] = self._rng.randn(n_in, n_out) * np.sqrt(
                2.0 / n_in
            )

            # zero initialization for biases
            self.parameters[f"b{l}"] = np.zeros((1, n_out))

    def _forward_propagation(self, X):
        cache = {"A0": X}
        A = X

        for l in range(1, self.num_layers):
            Z = A @ self.parameters[f"W{l}"] + self.parameters[f"b{l}"]
            A = ActivationFunctions.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        Z_out = (
            A @ self.parameters[f"W{self.num_layers}"]
            + self.parameters[f"b{self.num_layers}"]
        )
        A_out = ActivationFunctions.softmax(Z_out)
        cache[f"Z{self.num_layers}"] = Z_out
        cache[f"A{self.num_layers}"] = A_out

        return A_out, cache

    def _compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        eps = 1e-8

        # clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        loss = -np.sum((y_true * np.log(y_pred_clipped))) / m

        return loss

    def _backward_propagation(self, cache, y_true):
        m = y_true.shape[0]
        gradients = {}

        # Output layer gradient (softmax + cross-entropy)
        # Simplified gradient: dZ = A - y_true (when using softmax + CE)
        dZ = cache[f"A{self.num_layers}"] - y_true

        # Backpropagate through layers
        for l in range(self.num_layers, 0, -1):
            A_prev = cache[f"A{l - 1}"]

            # Compute gradients for current layer
            gradients[f"dW{l}"] = (A_prev.T @ dZ) / m
            gradients[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m

            # Compute gradient for previous layer (if not input layer)
            if l > 1:
                dA_prev = dZ @ self.parameters[f"W{l}"].T
                dZ = dA_prev * ActivationFunctions.relu_derivative(cache[f"Z{l - 1}"])

        return gradients

    def _update_parameters(self, gradients):
        for l in range(1, self.num_layers + 1):
            self.parameters[f"W{l}"] -= self.learning_rate * gradients[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * gradients[f"db{l}"]

    def _compute_accuracy(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        patience: int | None = 15,
        verbose: int = 10,
    ) -> "MLP":
        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got {X.ndim}D")
        if X.shape[1] != self.architecture[0]:
            raise ValueError(
                f"Expected X_train with {self.architecture[0]} features, got {X.shape[1]}"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_train has {X.shape[0]} samples but y_train has {y.shape[0]}"
            )
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError(f"X_train contains Nan or inf values.")
        m = X.shape[0]
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle training data
            indices = self._rng.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            epoch_loss = 0
            n_batches = m // self.batch_size

            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward propagation
                y_pred, cache = self._forward_propagation(X_batch)

                # Compute loss
                batch_loss = self._compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss

                # Backward propagation
                gradients = self._backward_propagation(cache, y_batch)

                # Update parameters
                self._update_parameters(gradients)

            # Compute epoch metrics
            train_loss = epoch_loss / n_batches
            y_train_pred, _ = self._forward_propagation(X)
            train_acc = self._compute_accuracy(y_train_pred, y)

            y_val_pred, _ = self._forward_propagation(X_val)
            val_loss = self._compute_loss(y_val_pred, y_val)
            val_acc = self._compute_accuracy(y_val_pred, y_val)

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # Print progress
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_pred, _ = self._forward_propagation(X)
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            Input features

        Returns
        --------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if X.shape[1] != self.architecture[0]:
            raise ValueError(
                f"Expected X with {self.architecture[0]} features, got {X.shape[1]}"
            )
        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1)

    def summary(self) -> None:
        print(f"MLP Architecture {self.architecture}")
        print("-" * 50)
        total_params = 0
        for l in range(1, self.num_layers + 1):
            n_in = self.architecture[l - 1]
            n_out = self.architecture[l]
            params = n_in * n_out + n_out  # weights + biases
            total_params += params

            if l == self.num_layers:
                act = "Softmax"
            else:
                act = "ReLU"

            print(f"Layer {l}: {n_in} -> {n_out} ({act}) | Params: {params}")
        print("-" * 50)
        print(f"Total params: {total_params:,}")
