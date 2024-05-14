import numpy as np

def sigmoid(x:float) -> float:
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, num_inputs:int):
        self.weights:np.array = 2*np.random.rand(num_inputs) - np.ones(num_inputs)
        self.bias:float = float(np.random.rand())

    def forward(self, x_vector:np.array) -> float:
        z:float = np.dot(x_vector, self.weights) + self.bias
        y:float = sigmoid(z)
        return y

neuron = Neuron(3)
inputs = np.array([1, 0.2, 0.9])
output = neuron.forward(inputs)
print(output)