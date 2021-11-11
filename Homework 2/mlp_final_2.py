import numpy as np
from matplotlib import pyplot as plt
import time




# Possible choices for binary input
input_dataset = np.array([[0,0],[0,1],[1,0],[1,1]])

# List of bools as labels for respective logical operator
label_and = np.array([0,0,0,1])
label_or = np.array([0,1,1,1])
label_nand = np.array([1,1,1,0])
label_nor = np.array([1,0,0,0])
label_xor = np.array([0,1,1,0])




def sigmoid(x): 
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))


def sigmoidprime(x): 
    """
    Derivative of Sigmoid activation function
    """
    return sigmoid(x)*(1-sigmoid(x))




class Perceptron:
    """
    Instantiates a single perceptron with own weights and bias.

    Parameters:
    input_units(int): Length of the input array. Weights get created accordingly.
    """


    def __init__(self, input_units):
        """
        Saves all important variables for the perceptron. 
        Also initializes its weights and random bias.

        Parameters:
        input_units(int): Length of the input array. Weights get created accordingly.
        """

        self.input_units = input_units
        self.bias = np.random.randn()

        # Learning rate
        self.alpha = 1
        # To be accessed for backpropagation
        self.drive = 0
        # Saves input
        self.inputs = 0

        # Random weight for each input
        self.weights = np.random.randn(input_units)


    def forward_step(self, inputs):
        """
        Calculates forward step using sigmoid activation function. 
        Passes inputs through perceptron.

        Parameters:
        inputs(arr/list): Size 2 with binary boolean input representing logical statement.
        """

        # Multiply weights with given inputs
        self.drive = self.weights @ inputs + self.bias
        # Store it
        self.inputs = inputs

        # Perceptron output
        return sigmoid(self.drive)


    def update(self, delta):
        """
        Calculated error-signal from backpropagation is used to update weights and bias.

        Parameters:
        delta(float): error signal computed in MLP -> backprop_step
        """

        #Compute gradients
        gradients = delta * self.inputs

        #Update parameters 
        self.bias -= self.alpha * delta
        self.weights -= self.alpha * gradients




class MLP:
    """
    Contains multiple "layers" of perceptron objects.
    Passes inputs through the network and computes the error-signal with backpropagation.
    Also trains the network
    """


    def __init__(self):
        """
        Initializes one hidden & one output layer.
        Stores important variables.
        """
        # Initializes layers expecting specific input sizes
        self.hiddenlayer = [Perceptron(2) for i in range(4)]
        self.outputneuron = Perceptron(4)

        # Network output
        self.output = 0

        # For measuring network performance
        self.accuracy_sum = 0
        self.loss = []
        self.accuracies = []
        self.start = time.time()


    def forward_step(self, inputs):
        """
        Pushes given inputs through each perceptron.

        Parameters:
        inputs(arr/list): Size 2 with binary boolean input representing logical statement.
        """

        # Call every perceptron in hl with given inputs
        calculated = np.array([perceptron.forward_step(inputs) for perceptron in self.hiddenlayer])
        # Pass hl output to output layer
        self.output = self.outputneuron.forward_step(calculated)


    def backprop_step(self, target):
        """
        Compute error-signal with backpropagation.

        Parameters:
        target: The ground truth. Output the network is supposed to have.
        """

        # Applies function to compute error signal for output layer
        delta = -(target - self.output) * sigmoidprime(self.outputneuron.drive)

        # Updates neuron in output layer
        self.outputneuron.update(delta)

        for perceptron in self.hiddenlayer:
            #Calculate hidden delta
            delta_hidden =  delta * self.outputneuron.weights[self.hiddenlayer.index(perceptron)] * sigmoidprime(perceptron.drive)
            #Update current perceptron with hidden delta
            perceptron.update(delta_hidden)


    def train(self, epoch):
        """
        Trains network.

        Parameters:
        epoch(int): Training repetitions.
        """

        # Variables for loss
        error = 0
        outputs = []


        # Iterates over input
        for datapoint in range(input_dataset.shape[0]):

            # Performs forward and backward step for each datapoint in input
            self.forward_step(input_dataset[datapoint])
            self.backprop_step(label_xor[datapoint])
            
            # Computes Mean Squared Error
            error += (label_xor[datapoint] - self.output)**2

            # Loss
            output = -1

            # Network output >1 is enough to be classified as TRUE
            if self.output >= 0.5:
                output = 1
            else:
                output = 0

            # Keeps track of effiency
            if (output == label_xor[datapoint]):
                self.accuracy_sum += 1

            # Save network output
            outputs.append(output)


        # Measure performance
        self.accuracies.append(round(self.accuracy_sum/((epoch * 4) + 1), 4))
        self.loss.append(error)
        elapsed = (time.time() - self.start)

        # Contains all valueable information to evaluate network performance
        return [outputs, elapsed, epoch + 1, self.accuracies[-1], self.loss[-1]]




def plot(epochs, loss, accuracy):
    """
    Creates plots for given lists.

    Parameters:
    loss(list): Error of the network.
    accuracy(list): Accuracy of the network. 

    """

    # Both plots in one figure
    fig, (ax1, ax2) = plt.subplots(2,1,figsize = (14,12))

    # Set titles
    fig.suptitle("Evolution of networks prediction", fontsize=16)
    ax1.set(
        title= "Average loss per epoch",
        xlabel= "Epoch",
        ylabel= "Loss",
    )
    ax2.set(
        title= "Average accuracy per epoch",
        xlabel= "Epoch",
        ylabel= "Accuracy",
        ylim=(0,1)
    )

    # Feed data into plots
    ax1.plot(epochs, loss)
    ax2.plot(epochs, accuracy)

    # Start plotting
    plt.show()




# Runs the network
# Check wether library or program
if __name__ == "__main__":

    # Create MLP object
    mlp = MLP()

    # Store performance for plotting
    loss = []
    accuracy = []
    epochs = []

    # Iterate training steps
    for epoch in range(10000):
        # One training step (starts the magic)
        training_output = mlp.train(epoch)

        # Keep track of performance
        loss.append(training_output[4])
        epochs.append(epoch)
        accuracy.append(training_output[3])

    # Once trained, plot results
    plot(epochs, loss, accuracy)