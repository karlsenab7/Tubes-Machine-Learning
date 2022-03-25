import math
import numpy as np
from components.Activation import Activation
import components.Layer as ly


class MiniBatchGradientMLP:
    def __init__(self, nodesArr, batchSize):
        self.batchSize = batchSize
        print("testtttttt")
        self.layers = [ly.Layer(nbNodes, Activation.LINEAR_ACTIVATION) for nbNodes in nodesArr]
        # np.random.seed(2)
        # ditambah 1 untuk biasnya
        self.weights = [np.random.rand(
            nodesArr[i]+1, nodesArr[i+1])-0.5 for i in range(len(nodesArr)-1)]

    def train(self, trainingData, trainingTarget, epochs, minCumulativeError, learningRate):
        trainingHistory = {
            "Epochs": epochs,
            "MinCumulativeError": minCumulativeError,
            "LearningRate": learningRate,
            "CumulativeError": [],
            "Accuracy": []
        }

        nbData = len(trainingData)
        nbFeature = len(trainingData[0])

        for i in range(epochs):
            accuracy = 0
            cumulativeError = 0
            count = 0
            globalDeltaW = [np.zeros(
                (self.layers[i].nbNodes+1, self.layers[i+1].nbNodes)) for i in range(len(self.layers)-1)]

            # iterasi tiap training data
            for j in range(nbData):
                inputs = trainingData[j]
                target = trainingTarget[j]

                output = self.feedForward(inputs)
                totalError = self.calcTotalError(output, target)
                cumulativeError += totalError

                localDeltaW = self.backwardProp(target, learningRate)

                # akumulasi deltaWeight
                for k in range(len(localDeltaW)):
                    globalDeltaW[k] += localDeltaW[k]

                # update akurasi
                if np.argmax(output) == np.argmax(target):
                    accuracy += 1

                # update weights jika sudah batch size
                if count == self.batchSize or i == epochs-1:
                    count = 0
                    for k in range(len(self.weights)):
                        self.weights[k] += globalDeltaW[k]
                    globalDeltaW = [np.zeros(
                        (self.layers[i].nbNodes+1, self.layers[i+1].nbNodes)) for i in range(len(self.layers)-1)]

                count += 1

            # update training history
            trainingHistory["CumulativeError"].append(cumulativeError)
            trainingHistory["Accuracy"].append(accuracy/nbData)
            print("Epoch : {:d} Cumulative Error : {:f} Accuracy : {:f}".format(
                i+1, cumulativeError, accuracy/nbData))

            # jika cumulative error nya udah lebih kecil dari minCumulativeError
            if cumulativeError <= minCumulativeError:
                break

        return trainingHistory

    def predict(self, instances):
        results = []
        for instance in instances:
            res = self.feedForward(instance)
            idx = np.argmax(res)
            results.append(idx)

        return results  # list of numpy array

    def calcTotalError(self, outputs, targets):
        errors = 0
        for i in range(len(targets)):
            errors += math.pow((targets[i]-outputs[i]), 2)

        return errors*0.5

    def feedForward(self, inputs, pr=False):
        if len(inputs) != self.layers[0].nbNodes:
            raise ValueError("Shape of input ({:d}) does not match number of nodes ({:d})".format(
                len(instance), self.layers[0].nbNodes))
        else:
            self.layers[0].outputs = np.array(inputs)
            instance = inputs
            for i, layer in enumerate(self.layers[1:]):
                instance = np.append(instance, 1)  # biasnya
                val = np.dot(instance, self.weights[i])
                if pr:
                    print(val, i)
                instance = layer.compute(val)

        return instance

    def backwardProp(self, targets, learningRate):
        deltaWeights = [np.zeros((self.layers[i].nbNodes+1, self.layers[i+1].nbNodes))
                        for i in range(len(self.layers)-1)]

        # cari deltaNode untuk tiap node di tiap layer (kecuali input)
        for i in range(len(self.layers)-1, 0, -1):
            currLayer = self.layers[i]

            # hitung untuk output layer
            if i == len(self.layers)-1:
                for j in range(currLayer.nbNodes):
                    outputK = currLayer.outputs[j]
                    self.layers[i].deltas[j] = outputK * \
                        (1-outputK)*(targets[j]-outputK)

            # hitung untuk hidden layer
            else:
                for j in range(currLayer.nbNodes):
                    outputH = currLayer.outputs[j]
                    self.layers[i].deltas[j] = outputH * \
                        (1-outputH) * \
                        np.dot(self.weights[i][j], self.layers[i+1].deltas)

        # cari deltaW nya untuk tiap weight
        for n in range(len(deltaWeights)):
            outputs = self.layers[n].outputs
            deltas = self.layers[n+1].deltas

            partialRes = []
            for output in outputs:
                for delta in deltas:
                    partialRes.append(output*delta)

            for delta in deltas:
                partialRes.append(delta)

            i = 0
            for row in range(len(deltaWeights[n])):
                for col in range(len(deltaWeights[n][row])):
                    deltaWeights[n][row][col] = partialRes[i]*learningRate
                    i += 1

        return deltaWeights
