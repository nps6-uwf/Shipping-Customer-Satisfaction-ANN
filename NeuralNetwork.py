# Author: Nick Sebasco
# Date: 4/22/2021
# Version: V1
# Overview:  Implementing a ANN from scratch to classify.
import numpy as np
from random import shuffle

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        np.random.seed(1) # seed the prng :)
        self.network = [
            # hidden layer
            [{'weights':[np.random.rand() for i in range(n_inputs + 1)]} for i in range(n_hidden)],
            # output layer
            [{'weights':[np.random.rand() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        ]
    # 1. Feed forward proagation methods:
    # Calculate neural activation for inputs.
    def activate(self, weights, inputs):
        return weights[0] * inputs[0] + self.activate(weights[1:], inputs[1:]) if len(weights) > 1 else weights[0]

    # Transfer neuron activation
    def transfer(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))
    
    # Forward propagate input to a network output
    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    # 2. Feed backward propagation methods:
    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)
    
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])
    
    # Update network weights with error
    def update_weights(self, network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']
    
    # calculate sum square error, used to train network.  hoping to minimize this.
    def sum_squared_error(self,expected, outputs):
        return sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])

    # Train a network for a fixed number of epochs
    def train_network(self,network, train, l_rate, n_epoch, n_outputs, log=True):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[int(row[-1])] = 1

                sum_error += self.sum_squared_error(expected, outputs)
                self.backward_propagate_error(network, expected)
                self.update_weights(network, row, l_rate)
            if log:
                print(f'->epoch:{epoch}, error:{sum_error:.4f}')

    # Make classification prediction.
    def predict(self, network, test_vector):
        outputs = self.forward_propagate(network, test_vector)
        return outputs.index(max(outputs))



