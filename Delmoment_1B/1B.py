import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer:
    def __init__(self, input_size, num_neurons):
        self.weights = 2 * np.random.rand(input_size, num_neurons) - 1
        self.biases = 2 * np.random.rand(num_neurons) - 1

    def matrix_multi(self, input_vector):
        layer_outputs = np.dot(input_vector, self.weights) + self.biases
        activated_outputs = [sigmoid(g) for g in layer_outputs]
        return np.array(activated_outputs)

input_vector = np.array([1.0, 2.0, 3.0])
layer = Layer(3,2)
output = layer.matrix_multi(input_vector)
print(output)