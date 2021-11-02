import numpy as np
import math
import random

#Sigmoid function
def sigmoid(x): return 1 / 1 + math.exp(-x)

#Derivative of the sigmoid function
def sigmoidprime(x): return sigmoid(x)*(1-sigmoid(x))

#Training data set
#Inputs should be in the format: (x,y), and should be boolean values
def generate_data(operator):
    while True:
        a = not not random.getrandbits(1)
        b = not not random.getrandbits(1)
        if operator == "and":
            yield [a, b, bool(a and b)]
        elif operator == "or":
            yield [a, b, bool(a or b)]
        elif operator == "nand":
            yield [a, b, not bool(a and b)]
        elif operator == "nor":
            yield [a, b, not bool(a or b)]
        elif operator == "xor":
            yield [a, b, bool(a^b)]
        else:
            raise ValueError("Not an operator!")










if __name__ == "__main__":
    i = 0
    for i in range(10):
        print(next(generate_data("nand")))




#and

#or

#not and

#not or

#xor (exclusive or)
