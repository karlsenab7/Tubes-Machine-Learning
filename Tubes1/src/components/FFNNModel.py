import pandas as pd
from components.Activation import Activation
from components.Layer import Layer

class FFNNModel:
    def __init__(self, layers):
        self.layers : list[Layer] = layers
        pass

    def setInitialInput(self, input: list[int]):
        if (len(self.layers) >= 1):
            self.layers[0].values = input

    def solve(self):
        for i in range(len(self.layers)-1):
            idxLayerInput = i
            idxLayerTarget = i+1
            targetValues = []
            for nodeTarget in range(self.layers[idxLayerTarget].numOfNode):
                sigma = self.getSigma(idxLayerTarget, self.layers[idxLayerInput].values, nodeTarget)
                print(sigma)
                tempVal = Activation.active(sigma, self.layers[idxLayerTarget].activation)
                targetValues.append(tempVal)

            self.layers[idxLayerTarget].values = targetValues
            # print(targetValues)
        return self.layers[len(self.layers)-1].values

    
    def getSigma(self, idxLayer, input, node):
        temp = 0
        for idx, w in enumerate(self.layers[idxLayer].weights[node]):
            if (idx != 0):
                temp += w * input[idx-1]
            else:
                temp += w
        return temp

    @staticmethod
    def getModelFromFile (path):
        layers: list[Layer] = []
        df = pd.read_json(path)	
        for layer in df.model.layers :
            layer_ = Layer(layer['activation'], layer['numOfNode'], layer['weights'], layer['values'])
            layers.append(layer_)

        return FFNNModel(layers)