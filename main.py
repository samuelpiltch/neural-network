import numpy as np

class NeuralNetwork:

    def __init__(self, layer_nodes):
        self.layer_nodes = layer_nodes # A list of the number of nodes in each layer.
        self.weights = []
        self.biases = []
        self.layers = []

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid function used to back propagate
    def sigmoid_derivative(self, x):
        return x * (1 - x)

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
    # Take the resulting matrix and add it to the self.layers array. Repeat these steps for each output until the final layer has been reached.
    # Return the predicted output (last layer).
    def feed_forward(self, input_layer):
        self.layers = [np.array(input_layer)]
        for layer in range(0, len(self.layer_nodes)-1):
            next_layer = np.dot(self.layers[layer], self.weights[layer])
            next_layer = np.add(next_layer, self.biases[layer])
            next_layer = self.sigmoid(next_layer)

            self.layers.append(next_layer)

        return self.layers[-1]

    # Train the neural network by backpropagating using gradient descent.
    def back_propagate(self, input_layer, output_layer):
        errors = []
        deltas = []

        for layer in reversed(range(1, len(self.layer_nodes))):
            if (layer == len(self.layer_nodes)-1):
                error = output_layer - self.layers[-1]
                errors.append(error)
            else:
                error = np.dot(deltas[-1], self.weights[layer].T)
                errors.append(error)

            delta = errors[-1] * self.sigmoid_derivative(self.layers[layer])
            deltas.append(delta)

        for weight in range(0, len(self.layer_nodes)-1):
            self.weights[weight] += self.layers[weight].T.dot(deltas[len(deltas)-weight-1])

    def train(self, input_layer, output_layer, iterations):
        for iteration in range(iterations):
            self.feed_forward(input_layer)
            self.back_propagate(input_layer, output_layer)

            if (iteration == iterations-1):
                print "Input: \n" + str(input_layer)
                print "Actual Output: \n" + str(output_layer)
                print "Predicted Output: \n" + str(self.feed_forward(input_layer))
                print "Loss: \n" + str(np.mean(np.square(output_layer - self.feed_forward(input_layer)))) # mean sum squared loss
                print "Weights: \n" + str(self.weights)

net = NeuralNetwork([2, 3, 1])
net.init_weights()

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# Scale units
X = X / np.amax(X, axis=0)
y = y / 100

net.train(X, y, 100000)
