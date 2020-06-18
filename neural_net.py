import numpy as np


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)
        self.weights = {}  # create dict to hold weights
        self.adjustments = {}  # create dict to hold adjusts
        self.biases = {}
        self.num_layers = 1

    def add_layer(self, shape):
        # Create weights with shape specified + biases
        self.weights[self.num_layers] = np.vstack(
            (2 * np.random.random(shape) - 1, 2 * np.random.random((1, shape[1])) - 1))
        # Initialize the adjustments for these weights to zero
        self.adjustments[self.num_layers] = np.zeros(shape)
        self.num_layers += 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def predict(self, data):
        # Pass data through pre-trained network
        for layer in range(2, self.num_layers + 1):
            dot_data_weights = np.dot(data, self.weights[layer - 1][:, :-1])
            extra_weight = self.weights[layer - 1][:, -1]  # + self.biases[layer]
            data = dot_data_weights + extra_weight
            data = self.sigmoid(data)
        return data

    def forward_propagate(self, data):
        # Propagate through network and hold values for use in back-propagation
        activation_values = {}
        activation_values[1] = data
        print("\n")
        print(data)
        for layer in range(2, self.num_layers + 1):
            dot_data_weights = np.dot(data.T, self.weights[layer - 1][:-1, :])
            extra_weight = self.weights[layer - 1][-1, :].T # + self.biases[layer]
            data = dot_data_weights + extra_weight
            # print(self.weights[layer - 1][:-1, :])
            # print(dot_data_weights)
            # print(extra_weight)
            # print(extra_weight.T)
            # print(data)
            # print("\n")
            data = self.sigmoid(data).T
            activation_values[layer] = data
        return activation_values

    @staticmethod
    def simple_error(outputs, targets):
        return targets - outputs

    @staticmethod
    def sum_squared_error(outputs, targets):
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def back_propagate(self, output, target):
        # Delta of output Layer (dictionary)
        deltas = {}

        deltas[self.num_layers] = output[self.num_layers] - target
        # Delta of hidden Layers
        for layer in reversed(range(2, self.num_layers)):  # All layers except input/output
            a_val_sig_deriv = self.sigmoid_derivative(output[layer])
            weights = self.weights[layer][:-1, :]
            prev_deltas = deltas[layer + 1]
            deltas[layer] = np.multiply(np.dot(weights, prev_deltas), a_val_sig_deriv)
        # Calculate total adjustments based on deltas
        for layer in range(1, self.num_layers):
            # forward analysis after updating back steps
            self.adjustments[layer] += np.dot(deltas[layer + 1], output[layer].T).T

    def gradient_descent(self, batch_size, learning_rate):
        # Calculate partial derivative and take a step in that direction
        for layer in range(1, self.num_layers):
            neg_partial_d = - (1 / batch_size) * self.adjustments[layer]
            self.weights[layer][:-1, :] += learning_rate * neg_partial_d
            self.weights[layer][-1, :] += learning_rate * 1e-3 * neg_partial_d[-1, :]

    def train(self, inputs, targets, num_epochs, learning_rate=1, stop_accuracy=1e-5):
        offset = []
        for epoch in range(num_epochs):
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]
                # Pass the training set through our neural network
                output = self.forward_propagate(x)
                # Calculate the error
                loss = self.sum_squared_error(output[self.num_layers], y)
                offset.append(loss)
                # Calculate Adjustments
                self.back_propagate(output, y)
            # update weights by gradient descent
            self.gradient_descent(len(inputs) - 1, learning_rate)
            # Check if accuracy criterion is satisfied
            if np.mean(offset[-(len(inputs)):]) < stop_accuracy and epoch > 0:
                break
        return np.asarray(offset), epoch + 1

    # def train(self, training_inputs, training_outputs, training_iterations, biases):
    #     # self.synoptic_weights += biases
    #     for i in range(training_iterations):
    #         output = self.think(training_inputs)
    #         output += biases
    #         error = self.simple_error(output, training_outputs)
    #         # biases *= self.sigmoid(error)
    #         adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
    #         print(adjustments)
    #         print("\n")
    #         self.synoptic_weights += adjustments

    def think(self, inputs):
        return self.sigmoid(np.dot(inputs.astype(float), self.weights))


if __name__ == "__main__":
    # ----------- XOR Function -----------------

    # Create instance of a neural network
    nn = NeuralNetwork()

    # Add Layers (Input layer is created by default)
    nn.add_layer((2, 9))
    nn.add_layer((9, 1))

    # XOR function
    training_data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1)
    print(training_data, len(training_data))
    print([[0, 1]])
    training_labels = np.asarray([[0], [1], [1], [0]])

    error, iteration = nn.train(training_data, training_labels, 5000)
    print('Error = ', np.mean(error[-4:]))
    print('Epoches needed to train = ', iteration)
    # for layer in range(2, nn.num_layers + 1):
    #     print(np.dot(np.asarray([[0, 0]]).reshape(2, 1), nn.weights[layer - 1][:, :-1]) + nn.weights[layer - 1][:, -1])

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    print("New situation: input data = ", A, B)
    print('Outputs after training: ')

    nn.predict(np.array([A, B]).reshape(2, 1))

    # neural_network = NeuralNetwork()
    #
    # print('Random syn weights:')
    # print(neural_network.synoptic_weights)
    # print("\n")
    # training_inputs = np.array([[0, 0, 1],
    #                             [1, 1, 1],
    #                             [0, 0, 0],
    #                             [1, 0, 1],
    #                             [0, 1, 1]])
    #
    # training_outputs = np.array([[0, 1, 0, 1, 0]]).T
    #
    # training_biases = np.array([[0], [0], [-.5], [0], [0]])
    #
    # neural_network.train(training_inputs, training_outputs, 10000, biases=training_biases)
    #
    # print('Syn weights after training: ')
    # print(neural_network.synoptic_weights)
    #
    # A = str(input("Input 1: "))
    # B = str(input("Input 2: "))
    # C = str(input("Input 3: "))
    #
    # print("New situation: input data = ", A, B, C)
    # print('Outputs after training: ')
    # print(neural_network.think(np.array([A, B, C])))
