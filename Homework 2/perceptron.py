import numpy as np
import math
import random
from homework2 import sigmoid, sigmoidprime


class Perceptron:
    def __init__(self, weights, input_units = random.randint(1,10), bias = random.uniform(0,1), alpha = 1):
        self.input_units = input_units
        self.bias = bias
        self.alpha = alpha
        if weights:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.input_units)
    
    #Calculates forward step using the sigmoid function
    def forward_step(self, inputs): return [sigmoid(input) for input in inputs]

    def update(self, delta):
        #Compute gradients

        #Update parameters
        

        pass


class MLP:
    def __init__(self, *layers):
        self.layers = layers


if __name__ == "__main__":
    inputs = [4,2,6,8,4,3]
