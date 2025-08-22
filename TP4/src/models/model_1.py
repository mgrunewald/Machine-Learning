#El modelo base para todo el trabajo.
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLP(object):

    def __init__(self, layers=[4, 5, 1], activations=["relu", "sigmoid"], verbose=True, plot=False) -> None:
        """
        Initializes the MLP with specified layers, activations, and optional verbosity/plotting settings.
        Inputs:
            layers: List of integers representing the number of nodes in each layer.
            activations: List of activation functions for each layer.
            verbose: Boolean flag for logging output.
            plot: Boolean flag for plotting learning curves.
        """
        assert len(layers) == len(activations) + 1, "Number of layers and activations mismatch"
        self.layers = layers
        self.num_layers = len(layers)
        self.activations = activations
        self.verbose = verbose
        self.plot = plot

        # Initialize weights and biases randomly
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def forward_pass(self, x):
        """
        Performs forward propagation of input data through the MLP.
        Inputs:
            x: Features vector (input data).
        Returns:
            a: List of preactivations for each layer.
            z: List of activations for each layer.
        """
        z = [np.array(x).reshape(-1, 1)] 
        a = []  

        for l in range(1, self.num_layers):
            a_l = np.dot(self.weights[l - 1], z[l - 1]) + self.biases[l - 1]
            a.append(np.copy(a_l))
            h = self.getActivationFunction(self.activations[l - 1])
            z_l = h(a_l)
            z.append(np.copy(z_l))

        return a, z
    
    def backward_pass(self, a, z, y):
        """
        Performs backward propagation to compute gradients of the loss with respect to weights and biases.
        Inputs:
            a: List of preactivations from forward pass.
            z: List of activations from forward pass.
            y: True target values.
        Returns:
            nabla_b: List of gradients for biases.
            nabla_w: List of gradients for weights.
            loss: Calculated loss value.
        """
        delta = [np.zeros(w.shape) for w in self.weights]
        h_prime = self.getDerivitiveActivationFunction(self.activations[-1])
        output = z[-1]
        delta[-1] = (output - y)  # Derivative of MSE

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = delta[-1]
        nabla_w[-1] = np.dot(delta[-1], z[-2].T)

        for l in reversed(range(1, len(delta))):
            h_prime = self.getDerivitiveActivationFunction(self.activations[l - 1])
            delta[l - 1] = np.dot(self.weights[l].T, delta[l]) * h_prime(a[l - 1])
            nabla_b[l - 1] = delta[l - 1]
            nabla_w[l - 1] = np.dot(delta[l - 1], z[l - 1].T)

        loss = np.mean((output - y) ** 2)  # MSE loss
        return nabla_b, nabla_w, loss

    def update_mini_batch(self, mini_batch, lr):
        """
        Updates model weights and biases using gradients computed from a mini-batch.
        Inputs:
            mini_batch: List of training samples (features and targets).
            lr: Learning rate for gradient updates.
        Returns:
            Average loss for the mini-batch.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        total_loss = 0

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, loss = self.backward_pass(*self.forward_pass(x), y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            total_loss += loss

        self.weights = [w - lr * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - lr * nb for b, nb in zip(self.biases, nabla_b)]
        return total_loss / len(mini_batch)

    def fit(self, training_data, epochs, mini_batch_size, lr, val_data=None, verbose=0):
        """
        Trains the MLP using the provided training data, with options for validation and verbosity.
        Inputs:
            training_data: List of tuples (features, targets) for training.
            epochs: Number of epochs to train.
            mini_batch_size: Number of samples per mini-batch.
            lr: Learning rate.
            val_data: Optional validation data for performance monitoring.
            verbose: Verbosity level for progress output.
        Returns:
            train_losses: List of training loss values per epoch.
            val_losses: List of validation loss values per epoch (if validation data is provided).
        """
        train_losses = []
        val_losses = []
        n = len(training_data)
        
        use_tqdm = verbose == 0 or verbose == 2
        print_detailed = verbose == 1 or verbose == 2
        progress_bar = tqdm(total=epochs, desc="Training Epochs") if use_tqdm else None

        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[i:i + mini_batch_size] for i in range(0, n, mini_batch_size)]
            
            epoch_train_losses = []

            for mini_batch in mini_batches:
                train_loss = self.update_mini_batch(mini_batch, lr)
                epoch_train_losses.append(train_loss)

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            if val_data:
                val_loss = self.evaluate(val_data)
                val_losses.append(val_loss)
            
            if print_detailed:
                if val_data:
                    print(f"Epoch {e + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {e + 1}: Train Loss: {avg_train_loss:.4f}")

            if use_tqdm:
                progress_bar.update(1)

        if use_tqdm:
            progress_bar.close()

        if self.plot: 
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            if val_losses:
                plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Curves')
            plt.legend()
            plt.grid()
            plt.show()

        return train_losses, val_losses
    
    def evaluate(self, test_data):
        """
        Evaluates the model on a given test dataset.
        Inputs:
            test_data: List of tuples (features, targets) for evaluation.
        Returns:
            Average binary cross-entropy loss on the test data.
        """
        sum_loss = 0
        for x, y in test_data:
            prediction = self.forward_pass(x)[-1][-1]
            sum_loss += np.mean((prediction - y) ** 2)  # MSE loss
        return sum_loss / len(test_data)

    def predict(self, X):
        """
        Predicts output labels for input data.
        Inputs:
            X: Array-like input data for prediction.
        Returns:
            Predictions as a numpy array.
        """
        predictions = []
        for x in X:
            prediction = self.forward_pass(x)[-1][-1].flatten()
            predictions.append(prediction)
        return np.array(predictions)

    @staticmethod
    def getActivationFunction(name):
        """
        Returns the activation function based on the provided name.
        Inputs:
            name: String representing the activation function ('sigmoid' or 'relu').
        Returns:
            Activation function corresponding to the name.
        """
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(x, 0)
        else:
            print('Unknown activation function. Using linear by default.')
            return lambda x: x

    @staticmethod
    def getDerivitiveActivationFunction(name):
        """
        Returns the derivative of the activation function based on the provided name.
        Inputs:
            name: String representing the activation function ('sigmoid' or 'relu').
        Returns:
            Derivative of the activation function.
        """
        if name == 'sigmoid':
            sig = lambda x: 1 / (1 + np.exp(-x))
            return lambda x: sig(x) * (1 - sig(x))
        elif name == 'relu':
            def relu_diff(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y
            return relu_diff
        else:
            print('Unknown activation function. Using linear by default.')
            return lambda x: 1