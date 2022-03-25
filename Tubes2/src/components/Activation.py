import math

class Activation:
    LINEAR_ACTIVATION = "linear"
    SIGMOID_ACTIVATION = "sigmoid"
    RELU_ACTIVATION = "relu"
    SOFTMAX_ACTIVATION = "softmax"
    
    @staticmethod
    def linear(x) :
        return round(x)

    @staticmethod
    def sigmoid(x):
        value = 1 / (1 + math.exp(x*(-1)))
        return value

    @staticmethod
    def relu(x):    
        return max(0.0, x)

    @staticmethod
    def softmax(x):
        return x
    
    @staticmethod
    def active(x, activation_function):
        if (activation_function == Activation.LINEAR_ACTIVATION):
            return Activation.linear(x)
        elif (activation_function == Activation.SIGMOID_ACTIVATION):
            return Activation.sigmoid(x)
        elif (activation_function == Activation.RELU_ACTIVATION):
            return Activation.relu(x)
        elif (activation_function == Activation.SOFTMAX_ACTIVATION):
            return Activation.softmax(x)