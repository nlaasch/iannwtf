import numpy as np
import random
from homework2 import sigmoid, sigmoidprime, generate_data
import matplotlib as plt

input_dataset = np.array([[0,0],[0,1],[1,0],[1,1]])
label_and = np.array([0,0,0,1])
label_or = np.array([0,1,1,1])
label_nand = np.array([1,1,1,0])
label_nor = np.array([1,0,0,0])
label_xor = np.array([0,1,1,0])


class Perceptron:
    def __init__(self, input_units):
        self.input_units = input_units
        self.bias = random.uniform(0,1)
        self.alpha = 1
        self.drive = None
        self.weights = np.random.rand(self.input_units) 
    
    #Calculates forward step using the sigmoid function
    def forward_step(self, inputs):
        self.drive = inputs @ self.weights + self.bias
        #print("Drive: " + str(self.drive))
        #print("Calculated forward step: " + str(sigmoid(self.drive)))
        return sigmoid(self.drive) 

    def update(self, delta):
        #Compute gradients
        gradients = self.weights * delta

        #Update parameters
        print("Previous Weights: " + str(self.weights))
        self.weights = self.weights - (self.alpha * gradients)
        print("Updated Weights: " + str(self.weights))


  


class MLP:
    def __init__(self):
        i = 0
        self.hiddenlayer = [Perceptron(2) for i in range(4)]
        self.outputneuron = Perceptron(4)
        self.output = []
        self.data = None

    
    def forward_step(self):
        self.data = next(generate_data("xor"))
        calculated = [perceptron.forward_step(self.data[:2]) for perceptron in self.hiddenlayer]
        self.output.append([self.outputneuron.forward_step(calculated), self.data[2]])
        return [self.outputneuron.forward_step(calculated), self.data[2]]

    def backprop_step(self):
        delta = (self.output[-1][1] - self.output[-1][0]) * sigmoidprime(self.outputneuron.drive)
        self.outputneuron.update(delta)
        self.outputneuron.bias = self.outputneuron.alpha * (self.output[-1][1] - self.output[-1][0])

        for perceptron in self.hiddenlayer:
            #Calculate hidden delta
            delta_hidden = sigmoidprime(perceptron.drive) * delta * self.outputneuron.weights[self.hiddenlayer.index(perceptron)]
            #Update current perceptron with hidden delta
            perceptron.update(delta_hidden)
            #update bias from perceptron
            perceptron.bias = perceptron.alpha * (self.output[-1][1] - self.output[-1][0])
    


    def train(self):
        

        #Loss 
        loss = 0

        
        self.forward_step()
        self.backprop_step()

        loss += (self.data[2] - self.output[-1][0])**2

        original_output = self.output[-1][0]

        if self.output[-1][0] >= 0.5:
            self.output[-1][0] = 1
        else:
            self.output[-1][0] = 0
        
        return [loss, original_output, self.output[-1]]







if __name__ == "__main__":


    mlp = MLP()
    accuracy_sum = 0
    for i in range(1000):
        training_output = mlp.train()
        #Calculate accuracy
        if training_output[2][0] == training_output[2][1]:
            accuracy_sum += 1


        
        print("\nEpoch: " + str(i+1))
        print("Accuracy: " + str(accuracy_sum/(i+1)))
        print("Loss: " + str(training_output[0]))
        print("Original Output: " + str(training_output[1]))
        print("Actual output: " + str(training_output[2]))
