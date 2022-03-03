import math

class Activation:
    @staticmethod
    def linear(x) :
        return x

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def relu(x):    
        return max(0.0, x)

    @staticmethod
    def softmax(x):
        return x
    
    @staticmethod
    def active(x, activation_function):
        if (activation_function == 'linear'):
            return Activation.linear(x)
        elif (activation_function == 'sigmoid'):
            return Activation.sigmoid(x)
        elif (activation_function == 'relu'):
            return Activation.relu(x)
        elif (activation_function == 'softmax'):
            return Activation.softmax(x)