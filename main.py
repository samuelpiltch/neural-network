import numpy as np

class NeuralNetwork:

    def __init__(self, layer_nodes):
        self.layer_nodes = layer_nodes # A list of the number of nodes in each layer.
        self.weights = []
        self.biases = []

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Initialize each the weights between each layer to random values between -1 and 1.
    # Weights are stored in a matrix with size: layer_current x layer_next. Each matrix is then appended to a list used when feeding the data forward.
    # Initialize bias with 0 for each node.
    def init_weights(self):
        for layer in range(0, len(self.layer_nodes)-1):
            layer_current = self.layer_nodes[layer]
            layer_next = self.layer_nodes[layer + 1]
            weights = np.random.random((layer_current, layer_next)) * 2 - 1
            self.weights.append(weights)

            biases = np.zeros((1, layer_next))
            self.biases.append(biases)

    # Pass the input layer through each layer.
    # Dot multiply the input layer by the weights and add the bias. Apply the activation function to the sum.
    # Take the resulting matrix and add it to the layers array. Repeat these steps for each output until the final layer has been reached.
    # Return the predicted output (last layer).
    def feed_forward(self, input_layer):
        layers = [np.array(input_layer)]
        for layer in range(0, len(self.layer_nodes)-1):
            next_layer = np.dot(layers[layer], self.weights[layer])
            next_layer = np.add(next_layer, self.biases[layer])
            next_layer = self.sigmoid(next_layer)

            layers.append(next_layer)

        return layers[-1]
