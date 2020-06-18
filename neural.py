import sys
import numpy as np
import matplotlib

#batching input sets to get more robust outcome, high batch size may lead to over-correlations
X = [[1, 2, 3, 2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = .1* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,512)
layer2 = Layer_Dense(512,2)
layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

# weights = [[.2, .8, -.5, 1.0],
#            [.5, -.91, .26, -.5],
#            [-.26, -.27, .17, .87]]
# #
# bias = [2, 3, .5]
#
# weights2 = [[.1,-.14,.5],
#            [-.5, .12, -.33],
#            [-.26, -.27, .17, .87]]
# #
# bias2 = [-1, 2, -.5]

# output = np.dot(X, np.array(weights).T) + bias
# print(output)

# layer_outputs = []  # output of current layer
# for neuron_weights, neuron_bias in zip(weights, bias):
#     neuron_output = 0  # Output of given neuron
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
#
# print(layer_outputs)
# output = [bias1, bias2, bias3]
# for i in range(4):
#     for j in range(3):
#         output += inputs[i] * weights1[j]
#         output += inputs[i] * weights2[j]
#         output += inputs[i] * weights3[j]

# print(output)
# print("Python:", sys.version)
# print("Numpy:", np.__version__)
# print("Matplotlib:", matplotlib.__version__)
