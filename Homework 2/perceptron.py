import numpy as np
import random
from homework2 import sigmoid, sigmoidprime, generate_data
import matplotlib as plt
import time

input_dataset = np.array([[0,0],[0,1],[1,0],[1,1]])
label_and = np.array([0,0,0,1])
label_or = np.array([0,1,1,1])
label_nand = np.array([1,1,1,0])
label_nor = np.array([1,0,0,0])
label_xor = np.array([0,1,1,0])


class Perceptron:
    def __init__(self, input_units):
        self.input_units = input_units
        self.bias = np.random.randn()
        self.alpha = 1
        self.drive = 0
        self.inputs = 0
        self.weights = np.random.randn(input_units) 
    
    #Calculates forward step using the sigmoid function
    def forward_step(self, inputs):
        #print("Data Input: " + str(inputs))
        self.drive = self.weights @ inputs + self.bias
        self.inputs = inputs
        ##print("Drive: " + str(self.drive))
        ##print("Calculated forward step: " + str(sigmoid(self.drive)))
        return sigmoid(self.drive) 

    def update(self, delta):
        #Compute gradients
        gradients = delta * self.inputs
    
        #print("Gradients: " + str(gradients))

        #Update parameters
        #print("Previous Weights: " + str(self.weights))
        self.bias -= self.alpha * delta
        self.weights -= self.alpha * gradients
        #print("Updated Weights: " + str(self.weights))


  


class MLP:
    def __init__(self):
        self.hiddenlayer = [Perceptron(2) for i in range(4)]
        self.outputneuron = Perceptron(4)
        self.output = 0
        self.accuracy_sum = 0
        self.loss = []
        self.accuracies = []
        self.start = time.time()

    
    def forward_step(self, inputs):
        calculated = np.array([perceptron.forward_step(inputs) for perceptron in self.hiddenlayer])
        self.output = self.outputneuron.forward_step(calculated)

    def backprop_step(self, target):
        delta = -(target - self.output) * sigmoidprime(self.outputneuron.drive)

        self.outputneuron.update(delta)
        

        for perceptron in self.hiddenlayer:
            #Calculate hidden delta
            delta_hidden =  delta * self.outputneuron.weights[self.hiddenlayer.index(perceptron)] * sigmoidprime(perceptron.drive)
            #Update current perceptron with hidden delta
            perceptron.update(delta_hidden)

    


    def train(self, epoch):
        

        #Loss 
        error = 0
        outputs = []
        
        
        
        for datapoint in range(input_dataset.shape[0]):
            self.forward_step(input_dataset[datapoint])
            self.backprop_step(label_xor[datapoint])
            error += (label_xor[datapoint] - self.output)
            output = -1
            if self.output >= 0.5:
                output = 1
            else:
                output = 0
            if (output == label_xor[datapoint]):
                self.accuracy_sum += 1
            outputs.append(output)
            

        self.accuracies.append(round(self.accuracy_sum/((epoch * 4) + 1), 4))
        self.loss.append(error)

        elapsed = (time.time() - self.start)
        
        
        return [outputs, elapsed, epoch, self.accuracies[-1], self.loss[-1]]







if __name__ == "__main__":


    mlp = MLP()
    training_output = None

    for epoch in range(1000):
        training_output = mlp.train(epoch)
        print(training_output)

