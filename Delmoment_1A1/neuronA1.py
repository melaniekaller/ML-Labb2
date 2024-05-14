import math
import random

class Neuron:
    def __init__(self, num_inputs, bias):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = bias

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    
    def activate(self, inputs):
        z = self.bias
    
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
            return self.sigmoid(z)

num_inputs = 4
bias = 1
neuron = Neuron(num_inputs, bias)
output = neuron.activate([6, 2, 9, 4])
print("Output:", output)