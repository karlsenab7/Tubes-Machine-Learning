class Layer :
    def __init__(self, activation, numOfNode, weights, value) :
        self.activation : str = activation
        self.numOfNode : int = numOfNode
        self.weights : list[list[int]] = weights
        self.values : list[int] = value